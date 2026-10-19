package rag

import (
	"context"
	"errors"
	"testing"
	"time"

	"agi-assistant/config"
	"agi-assistant/internal/infrastructure/persistence/ragchunk"
)

type multiFailureRepo struct {
	queryRepo
	failure error
}

func (r multiFailureRepo) SearchES(ctx context.Context, q string, k int) ([]ragchunk.ESHit, error) {
	if q == "failed variant" {
		return nil, r.failure
	}
	return r.queryRepo.SearchES(ctx, q, k)
}

func TestMultiQueryFailureDiscardsPartialEvidence(t *testing.T) {
	want := errors.New("one rewritten query failed")
	e := NewEngine(&config.APIConfig{RAGConfig: config.RAGConfig{ChunkSize: 200, TopK: 5}}, multiFailureRepo{failure: want}, nil)
	e.Loaded = true
	e.SetRewriter(NewLLMRewriter(func(context.Context, string, string) (string, error) {
		return `{"queries":["valid variant","failed variant"]}`, nil
	}, 2))
	e.SetGenerateFn(func(context.Context, string, string) (string, error) {
		t.Error("generated from partial retrieval")
		return "wrong", nil
	})
	answer, results, err := e.QueryWithHistory(context.Background(), "question", nil)
	if answer != "" || results != nil || !errors.Is(err, want) {
		t.Fatalf("answer=%q results=%v err=%v", answer, results, err)
	}
}

func TestGenerationDeadlineNeverReturnsAnswer(t *testing.T) {
	e := NewEngine(&config.APIConfig{RAGConfig: config.RAGConfig{ChunkSize: 200, TopK: 5}}, queryRepo{}, nil)
	e.Loaded = true
	e.SetGenerateFn(func(ctx context.Context, _, _ string) (string, error) { <-ctx.Done(); return "late content", nil })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	answer, _, err := e.QueryWithHistory(ctx, "question", nil)
	if answer != "" || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("answer=%q err=%v", answer, err)
	}
}

type invalidEvidenceRepo struct {
	queryRepo
	rows []ragchunk.Row
}

func (r invalidEvidenceRepo) LoadByIDs(context.Context, []int64) ([]ragchunk.Row, error) {
	return r.rows, nil
}

func TestRetrievalRejectsInvalidAuthoritativeRows(t *testing.T) {
	for _, rows := range [][]ragchunk.Row{
		{{ID: 1, Content: "one"}},
		{{ID: 1, Content: "one"}, {ID: 1, Content: "duplicate"}},
		{{ID: 1, Content: "one"}, {ID: 3, Content: "unexpected"}},
	} {
		e := NewEngine(&config.APIConfig{RAGConfig: config.RAGConfig{ChunkSize: 200, TopK: 5}}, invalidEvidenceRepo{rows: rows}, nil)
		e.Loaded = true
		answer, _, err := e.QueryWithHistory(context.Background(), "question", nil)
		if answer != "" || err == nil {
			t.Fatalf("answer=%q err=%v", answer, err)
		}
	}
}
