package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"agi-assistant/internal/domain/rag"
	"github.com/DATA-DOG/go-sqlmock"
)

type fakeRetriever struct {
	rows   []rag.HybridResult
	err    error
	calls  int
	cancel context.CancelFunc
}

func TestExpectedBuildAndModelCheckedBeforeExecution(t *testing.T) {
	model := "rerank-model"
	backend := backendConfig{EmbeddingModel: "embed-model", RerankModel: model, RerankPreviewLen: 200}
	r := request{Mode: "hybrid_rerank", Expected: &expectedConfig{Revision: "source-sha", EmbeddingModel: "embed-model", Reranker: &model, RerankPreviewLen: 200}}
	if err := checkExpected(r, backend, "source-sha"); err != nil {
		t.Fatal(err)
	}
	if err := checkExpected(r, backend, "different-sha"); err == nil {
		t.Fatal("accepted stale binary")
	}
	backend.EmbeddingModel = "wrong-model"
	if err := checkExpected(r, backend, "source-sha"); err == nil {
		t.Fatal("accepted different model")
	}
	backend.EmbeddingModel = "embed-model"
	r.Mode = "vector"
	if err := checkExpected(r, backend, "source-sha"); err == nil {
		t.Fatal("vector arm accepted reranker")
	}
	r.Expected.Reranker = nil
	if err := checkExpected(r, backend, "source-sha"); err != nil {
		t.Fatal(err)
	}
}

func (f *fakeRetriever) RetrieveForEvaluation(ctx context.Context, q, mode string, k int) ([]rag.HybridResult, error) {
	f.calls++
	if f.cancel != nil {
		f.cancel()
	}
	return f.rows, f.err
}

func TestExplicitOptInPrecedesSettingsAndInput(t *testing.T) {
	for _, args := range [][]string{nil, {"--allow-backend-read"}, {"--allow-model-calls"}, {"--snapshot"}} {
		err := run(args, nil, io.Discard)
		if err == nil || !strings.Contains(err.Error(), "opt-in") {
			t.Fatalf("args=%v err=%v", args, err)
		}
	}
}

func TestInvalidRequestDoesNotReachRetriever(t *testing.T) {
	f := &fakeRetriever{}
	for _, r := range []request{{"q", "fallback", 20, 60, nil}, {"q", "vector", 4, 60, nil}, {"", "hybrid", 20, 60, nil}, {"q", "hybrid", 20, 0, nil}} {
		if _, err := retrieve(context.Background(), r, f); err == nil {
			t.Fatal("accepted invalid request")
		}
	}
	if f.calls != 0 {
		t.Fatal("called backend")
	}
}

func TestExportUsesChildIDAndContentNotParent(t *testing.T) {
	f := &fakeRetriever{rows: []rag.HybridResult{{PGID: 987, Chunk: rag.Chunk{ID: 0, Content: "child"}, Parent: "larger parent"}}}
	hits, err := retrieve(context.Background(), request{"q", "hybrid", 20, 60, nil}, f)
	if err != nil || len(hits) != 1 || hits[0].PGID != 987 || hits[0].ContentSHA256 != fmt.Sprintf("%x", sha256.Sum256([]byte("child"))) {
		t.Fatalf("%+v %v", hits, err)
	}
}

func TestExportRejectsBadHitsFailureAndLateCancel(t *testing.T) {
	valid := rag.HybridResult{PGID: 1, Chunk: rag.Chunk{Content: "child"}}
	for _, rows := range [][]rag.HybridResult{{valid, valid}, {{PGID: 0, Chunk: valid.Chunk}}, {{PGID: 1}}, {valid, valid, valid, valid, valid, valid}} {
		if _, err := retrieve(context.Background(), request{"q", "vector", 20, 60, nil}, &fakeRetriever{rows: rows}); err == nil {
			t.Fatal("accepted bad hits")
		}
	}
	want := errors.New("backend failure")
	if hits, err := retrieve(context.Background(), request{"q", "vector", 20, 60, nil}, &fakeRetriever{err: want}); hits != nil || !errors.Is(err, want) {
		t.Fatalf("%+v %v", hits, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if hits, err := retrieve(ctx, request{"q", "vector", 20, 60, nil}, &fakeRetriever{rows: []rag.HybridResult{valid}, cancel: cancel}); hits != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("%+v %v", hits, err)
	}
}

func TestSnapshotSelectOnlyAndNoPartialSuccess(t *testing.T) {
	for _, broken := range []bool{false, true} {
		db, mock, err := sqlmock.New()
		if err != nil {
			t.Fatal(err)
		}
		mock.ExpectBegin()
		rows := sqlmock.NewRows([]string{"id", "doc_hash", "chunk_idx", "content"}).AddRow(11, "ingested-doc", 3, "original child")
		if broken {
			rows.RowError(0, errors.New("read failed"))
		}
		mock.ExpectQuery("SELECT id, doc_hash, chunk_idx, content FROM rag_chunks ORDER BY id").WillReturnRows(rows)
		if broken {
			mock.ExpectRollback()
		} else {
			mock.ExpectCommit()
		}
		result, err := snapshot(context.Background(), db)
		if broken && (err == nil || result != nil) {
			t.Fatalf("partial success: %+v %v", result, err)
		}
		if !broken && (err != nil || len(result) != 1 || result[0].ChunkIdx != 3 || result[0].Content != "original child") {
			t.Fatalf("%+v %v", result, err)
		}
		if err := mock.ExpectationsWereMet(); err != nil {
			t.Fatal(err)
		}
		db.Close()
	}
}

func TestLiveIdentityLookupDetectsConcurrentChanges(t *testing.T) {
	for _, content := range []string{"original", "changed after retrieval"} {
		db, mock, err := sqlmock.New()
		if err != nil {
			t.Fatal(err)
		}
		mock.ExpectQuery("SELECT id, doc_hash, chunk_idx, content FROM rag_chunks WHERE id = ANY").WithArgs("{10}").
			WillReturnRows(sqlmock.NewRows([]string{"id", "doc_hash", "chunk_idx", "content"}).AddRow(10, "actual-doc", 2, content))
		hits := []hit{{PGID: 10, ContentSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte("original")))}}
		err = loadIdentities(context.Background(), db, hits)
		if content == "original" {
			if err != nil || hits[0].DocHash != "actual-doc" || hits[0].ChunkIdx != 2 {
				t.Fatalf("%+v %v", hits, err)
			}
		} else if err == nil {
			t.Fatal("accepted mutated live content")
		}
		if err := mock.ExpectationsWereMet(); err != nil {
			t.Fatal(err)
		}
		db.Close()
	}
}

func TestRequestGarbageRejectedBeforeConfigRead(t *testing.T) {
	for _, input := range []string{`{"question":"q","mode":"vector","candidate_k":20,"rrf_k":60} {}`, `{"extra":true}`, `not-json`} {
		var out bytes.Buffer
		err := run([]string{"--allow-backend-read", "--allow-model-calls", "--backend-config", "does-not-exist"}, strings.NewReader(input), &out)
		if err == nil || strings.Contains(err.Error(), "cannot read") || out.Len() != 0 {
			t.Fatalf("%v output=%s", err, out.String())
		}
	}
}
