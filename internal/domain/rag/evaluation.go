package rag

import (
	"context"
	"fmt"
	"sync"

	"agi-assistant/internal/infrastructure/persistence/ragchunk"
)

// RetrieveForEvaluation reuses the production retrieval implementation without
// generation, history rewriting or KG. It performs no startup, writes or indexing.
// candidateK is fixed across all three arms; topK is 5. Real embeddings/reranking
// may incur API costs, so callers must opt in after obtaining a budget.
// Errors are fatal here: a degraded hybrid run cannot masquerade as an ablation.
// PGID must be mapped to the corpus's canonical child chunk_id before scoring.
func (e *Engine) RetrieveForEvaluation(ctx context.Context, question, mode string, candidateK int) ([]HybridResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if candidateK < 5 {
		return nil, fmt.Errorf("candidateK must be >=5")
	}
	if mode != "vector" && mode != "hybrid" && mode != "hybrid_rerank" {
		return nil, fmt.Errorf("unknown evaluation mode %q", mode)
	}
	if !e.chunks.PGAvailable() || !e.chunks.MilvusAvailable() || (mode != "vector" && !e.chunks.ESAvailable()) {
		return nil, fmt.Errorf("required retrieval backends unavailable for %s", mode)
	}
	if e.store.embedFn == nil {
		return nil, fmt.Errorf("embedding callback missing")
	}
	if mode == "hybrid_rerank" && e.reranker == nil {
		return nil, fmt.Errorf("reranker missing")
	}
	checked := &evaluationRepo{Repo: e.chunks}
	hs := *e.store // Per-call copy: do not mutate live routing/callback state.
	hs.chunks, hs.kg, hs.reranker = checked, nil, nil
	hs.embedFn = func(ctx context.Context, text string) ([]float64, error) {
		v, err := e.store.embedFn(ctx, text)
		checked.record(err)
		return v, err
	}
	hs.mode = "hybrid"
	if mode == "vector" {
		hs.mode = "semantic"
	}
	results, err := hs.searchOne(ctx, question, candidateK)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := checked.failure(); err != nil {
		return nil, err
	}
	if mode == "hybrid_rerank" {
		return e.reranker.Rerank(ctx, question, results, 5)
	}
	return truncate(results, 5), nil
}

type evaluationRepo struct {
	ragchunk.Repo
	mu  sync.Mutex
	err error
}

func (r *evaluationRepo) record(err error) {
	if err != nil {
		r.mu.Lock()
		if r.err == nil {
			r.err = err
		}
		r.mu.Unlock()
	}
}
func (r *evaluationRepo) failure() error { r.mu.Lock(); defer r.mu.Unlock(); return r.err }
func (r *evaluationRepo) SearchMilvus(ctx context.Context, v []float32, k int) ([]ragchunk.MilvusHit, error) {
	hits, err := r.Repo.SearchMilvus(ctx, v, k)
	r.record(err)
	return hits, err
}
func (r *evaluationRepo) SearchES(ctx context.Context, q string, k int) ([]ragchunk.ESHit, error) {
	hits, err := r.Repo.SearchES(ctx, q, k)
	r.record(err)
	return hits, err
}
func (r *evaluationRepo) LoadByIDs(ctx context.Context, ids []int64) ([]ragchunk.Row, error) {
	rows, err := r.Repo.LoadByIDs(ctx, ids)
	r.record(err)
	if err == nil {
		requested := make(map[int64]bool, len(ids))
		for _, id := range ids {
			requested[id] = true
		}
		seen := make(map[int64]bool, len(rows))
		for _, row := range rows {
			if !requested[row.ID] || seen[row.ID] {
				r.record(fmt.Errorf("invalid/duplicate stored chunk ID %d", row.ID))
			}
			seen[row.ID] = true
		}
		if len(seen) != len(requested) {
			r.record(fmt.Errorf("retrieval index references missing stored chunks"))
		}
	}
	return rows, err
}
