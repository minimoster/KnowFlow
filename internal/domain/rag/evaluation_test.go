package rag

import (
	"context"
	"errors"
	"testing"

	"agi-assistant/config"
	"agi-assistant/internal/infrastructure/persistence/ragchunk"
)

type evalRepo struct {
	queryRepo
	esErr                error
	esCalls, vectorCalls int
}

func (*evalRepo) MilvusAvailable() bool { return true }
func (r *evalRepo) SearchMilvus([]float32, int) ([]ragchunk.MilvusHit, error) {
	r.vectorCalls++
	return []ragchunk.MilvusHit{{ID: 2, Distance: 1}, {ID: 1, Distance: .5}}, nil
}
func (r *evalRepo) SearchES(q string, k int) ([]ragchunk.ESHit, error) {
	r.esCalls++
	if r.esErr != nil {
		return nil, r.esErr
	}
	return r.queryRepo.SearchES(q, k)
}

func evaluationEngine(repo *evalRepo) *Engine {
	e := NewEngine(&config.APIConfig{RAGConfig: config.RAGConfig{ChunkSize: 200, TopK: 5, RRFConstantK: 60}}, repo, nil)
	e.SetEmbedFn(func(context.Context, string) ([]float64, error) { return []float64{1}, nil })
	e.SetReranker(NewLLMReranker(func(context.Context, string, string) (string, error) {
		return `{"scores":[{"idx":0,"score":1},{"idx":1,"score":10}]}`, nil
	}, 200))
	return e
}

func TestEvaluationModesReuseProductionRetrieval(t *testing.T) {
	for _, mode := range []string{"vector", "hybrid", "hybrid_rerank"} {
		t.Run(mode, func(t *testing.T) {
			repo := &evalRepo{}
			e := evaluationEngine(repo)
			got, err := e.RetrieveForEvaluation(context.Background(), "synthetic question", mode, 20)
			if err != nil || len(got) != 2 {
				t.Fatalf("%+v, %v", got, err)
			}
			want := int64(1)
			if mode != "hybrid" {
				want = 2
			}
			if got[0].PGID != want {
				t.Fatalf("got PGID=%d want=%d", got[0].PGID, want)
			}
			if mode == "vector" && repo.esCalls != 0 {
				t.Fatal("vector arm called keyword search")
			}
			if e.store.mode != "hybrid" || e.store.reranker == nil {
				t.Fatal("evaluation mutated live engine")
			}
		})
	}
}

func TestEvaluationRejectsSilentBackendFallback(t *testing.T) {
	want := errors.New("ES unavailable")
	e := evaluationEngine(&evalRepo{esErr: want})
	got, err := e.RetrieveForEvaluation(context.Background(), "q", "hybrid", 20)
	if got != nil || !errors.Is(err, want) {
		t.Fatalf("%+v, %v", got, err)
	}
}

func TestInvalidRerankIsAnError(t *testing.T) {
	for _, raw := range []string{`invalid`, `{"scores":[]}`, `{"scores":[{"idx":0,"score":5},{"idx":0,"score":5}]}`, `{"scores":[{"idx":0,"score":1.5},{"idx":1,"score":5}]}`} {
		r := NewLLMReranker(func(context.Context, string, string) (string, error) { return raw, nil }, 200)
		got, err := r.Rerank(context.Background(), "q", []HybridResult{{}, {}}, 5)
		if got != nil || err == nil {
			t.Fatalf("%+v, %v", got, err)
		}
	}
}

func TestEvaluationRejectsMissingRows(t *testing.T) {
	r := &evaluationRepo{Repo: queryRepo{}}
	_, _ = r.LoadByIDs([]int64{1, 2, 3})
	if r.failure() == nil {
		t.Fatal("missing indexed row was silently dropped")
	}
}
