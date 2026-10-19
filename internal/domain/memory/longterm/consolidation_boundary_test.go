package longterm

import (
	"agi-assistant/internal/domain/memory/consistency"
	"testing"
	"time"
)

func TestConsolidationExpiresSingleton(t *testing.T) {
	m := New()
	m.SetConsolidationConfig(&ConsolidationConfig{TTLDays: 30, MinImportance: 0.3, DecayRate: 1, DedupThreshold: 0.99, SimilarityThreshold: 0.9})
	now := time.Date(2026, 10, 19, 0, 0, 0, 0, time.UTC)
	m.StoreItem(Item{ID: 1, UserID: "u1", Content: "old", Importance: 0.1, CreatedAt: now.Add(-31 * 24 * time.Hour), Version: 2})
	p := m.PlanConsolidation(now)
	if len(p.Deletes) != 1 || p.Deletes[0].ID != 1 || p.Deletes[0].Reason != "expired" || m.Count() != 1 {
		t.Fatalf("plan=%+v count=%d", p, m.Count())
	}
}

func TestCommittedDecayDoesNotReapplyWholeAge(t *testing.T) {
	m := New()
	m.SetConsolidationConfig(&ConsolidationConfig{DecayRate: 0.9, DedupThreshold: 0.99, SimilarityThreshold: 0.9})
	now := time.Date(2026, 10, 19, 0, 0, 0, 0, time.UTC)
	m.StoreItem(Item{ID: 1, UserID: "u1", Content: "old", Importance: 1, CreatedAt: now.Add(-10 * 24 * time.Hour), UpdatedAt: now.Add(-24 * time.Hour), Version: 2})
	p := m.PlanConsolidation(now)
	if len(p.Updates) != 1 || p.Updates[0].Record.Importance != 0.9 {
		t.Fatalf("plan=%+v", p)
	}
	r := p.Updates[0].Record
	r.Version = 3
	r.UpdatedAt = now
	r.ContentHash, _ = consistency.ComputeContentHash(r)
	if err := m.ApplyCommitted(consistency.CommittedChangeSet{Upserts: []consistency.MemoryRecord{r}}); err != nil {
		t.Fatal(err)
	}
	again := m.PlanConsolidation(now)
	if len(again.Updates) != 0 || len(again.Deletes) != 0 {
		t.Fatalf("same-time run decayed again: %+v", again)
	}
}

func TestProjectionEmbeddingProvenanceSurvivesCacheRoundTrip(t *testing.T) {
	r := consistency.MemoryRecord{ID: 1, UserID: "u1", Version: 1, EmbeddingModel: "model", EmbeddingRevision: "rev"}
	got := itemToRecord(recordToItem(r))
	if got.EmbeddingModel != r.EmbeddingModel || got.EmbeddingRevision != r.EmbeddingRevision {
		t.Fatalf("provenance lost: %+v", got)
	}
}

func TestConsolidationDoesNotMergeUnsafeOrIncompatibleMemory(t *testing.T) {
	for _, kind := range []string{"quarantined", "superseded", "different model"} {
		t.Run(kind, func(t *testing.T) {
			m := New()
			m.SetConsolidationConfig(&ConsolidationConfig{DecayRate: 1, DedupThreshold: 0.99, SimilarityThreshold: 0.8})
			now := time.Now()
			m.StoreItem(Item{ID: 1, UserID: "u1", Content: "active", Importance: 0.8, Embedding: []float64{1, 0}, EmbeddingModel: "model", CreatedAt: now, Version: 1})
			other := Item{ID: 2, UserID: "u1", Content: "other", Importance: 0.9, Embedding: []float64{1, 0}, EmbeddingModel: "model", CreatedAt: now, Version: 1}
			switch kind {
			case "quarantined":
				other.Quarantined = true
			case "superseded":
				other.Superseded = true
			case "different model":
				other.EmbeddingModel = "different"
			}
			m.StoreItem(other)
			p := m.PlanConsolidation(now)
			if len(p.Updates)+len(p.Deletes) != 0 {
				t.Fatalf("unexpected merge: %+v", p)
			}
		})
	}
}
