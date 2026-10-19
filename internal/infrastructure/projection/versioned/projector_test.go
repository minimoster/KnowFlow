package versioned

import (
	"agi-assistant/internal/domain/memory/consistency"
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

type authorityStub struct {
	row *consistency.MemoryRecord
	err error
}

func (s authorityStub) LoadPage(_ context.Context, after int64, limit int) ([]consistency.MemoryRecord, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.row == nil {
		return nil, nil
	}
	if after != s.row.ID-1 || limit != 1 {
		return nil, errors.New("incorrect authoritative lookup")
	}
	return []consistency.MemoryRecord{*s.row}, nil
}

type storeStub struct {
	state  consistency.ProjectionState
	writes int
	row    consistency.MemoryRecord
}

func (s *storeStub) Get(context.Context, int64) (consistency.ProjectionState, bool, error) {
	return s.state, s.state.ID != 0, nil
}
func (s *storeStub) Upsert(_ context.Context, r consistency.MemoryRecord) error {
	s.writes++
	s.row = r
	s.state = consistency.ProjectionState{ID: r.ID, Version: r.Version, ContentHash: r.ContentHash}
	return nil
}
func (s *storeStub) Delete(context.Context, int64, int64) error {
	s.writes++
	s.state = consistency.ProjectionState{}
	return nil
}
func event(r consistency.MemoryRecord, delete bool) consistency.OutboxEvent {
	b, _ := json.Marshal(r)
	typ := consistency.EventUpsertMemoryVector
	if delete {
		typ = consistency.EventDeleteMemoryVector
	}
	return consistency.OutboxEvent{AggregateID: r.ID, AggregateVersion: r.Version, UserID: r.UserID, Type: typ, Target: consistency.TargetMilvus, Payload: b}
}
func projector(s *storeStub, a Authority) *Projector {
	return &Projector{Store: s, Source: a, TargetValue: consistency.TargetMilvus, Upserts: map[consistency.EventType]bool{consistency.EventUpsertMemoryVector: true}, Deletes: map[consistency.EventType]bool{consistency.EventDeleteMemoryVector: true}}
}

func TestTombstonePreventsResurrectionAfterPhysicalDelete(t *testing.T) {
	now := time.Now()
	live := consistency.MemoryRecord{ID: 1, UserID: "u1", Version: 1, ContentHash: "live"}
	dead := live
	dead.Version = 2
	dead.ContentHash = "dead"
	dead.DeletedAt = &now
	s := &storeStub{state: consistency.ProjectionState{ID: 1, Version: 1, ContentHash: "live"}}
	p := projector(s, authorityStub{row: &dead})
	if err := p.Apply(context.Background(), event(dead, true)); err != nil {
		t.Fatal(err)
	}
	if err := p.Apply(context.Background(), event(live, false)); err != nil {
		t.Fatal(err)
	}
	if s.state.ID != 0 || s.writes != 1 {
		t.Fatalf("old upsert resurrected tombstone: %+v", s)
	}
}

func TestReplayAndValidationBoundaries(t *testing.T) {
	r := consistency.MemoryRecord{ID: 1, UserID: "u1", Version: 2, ContentHash: "v2", Content: "authoritative"}
	for _, tc := range []struct {
		name    string
		source  Authority
		state   consistency.ProjectionState
		mutate  func(*consistency.OutboxEvent)
		wantErr bool
		writes  int
	}{
		{"duplicate", authorityStub{row: &r}, consistency.ProjectionState{ID: 1, Version: 2, ContentHash: "v2"}, nil, false, 0},
		{"newer target", authorityStub{row: &r}, consistency.ProjectionState{ID: 1, Version: 3}, nil, false, 0},
		{"hash conflict", authorityStub{row: &r}, consistency.ProjectionState{ID: 1, Version: 2, ContentHash: "other"}, nil, true, 0},
		{"source failure", authorityStub{err: errors.New("postgres down")}, consistency.ProjectionState{}, nil, true, 0},
		{"missing source", nil, consistency.ProjectionState{}, nil, true, 0},
		{"missing memory", authorityStub{}, consistency.ProjectionState{}, nil, true, 0},
		{"mismatched envelope", authorityStub{row: &r}, consistency.ProjectionState{}, func(e *consistency.OutboxEvent) { e.UserID = "other-user" }, true, 0},
		{"unsupported type", authorityStub{row: &r}, consistency.ProjectionState{}, func(e *consistency.OutboxEvent) { e.Type = "unknown" }, true, 0},
		{"wrong target", authorityStub{row: &r}, consistency.ProjectionState{}, func(e *consistency.OutboxEvent) { e.Target = consistency.TargetNeo4j }, true, 0},
		{"orphan repair cannot delete active row", authorityStub{row: &r}, consistency.ProjectionState{ID: 1, Version: 2}, func(e *consistency.OutboxEvent) { *e = event(consistency.MemoryRecord{ID: 1, Version: 2}, true) }, false, 0},
		{"confirmed orphan", authorityStub{}, consistency.ProjectionState{ID: 1, Version: 2}, func(e *consistency.OutboxEvent) { *e = event(consistency.MemoryRecord{ID: 1, Version: 2}, true) }, false, 1},
		{"authoritative upsert", authorityStub{row: &r}, consistency.ProjectionState{}, nil, false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &storeStub{state: tc.state}
			p := projector(s, tc.source)
			e := event(r, false)
			if tc.mutate != nil {
				tc.mutate(&e)
			}
			err := p.Apply(context.Background(), e)
			if (err != nil) != tc.wantErr || s.writes != tc.writes {
				t.Fatalf("err=%v writes=%d", err, s.writes)
			}
		})
	}
}

func TestPayloadCannotReplaceAuthoritativeContent(t *testing.T) {
	r := consistency.MemoryRecord{ID: 1, UserID: "u1", Version: 2, ContentHash: "v2", Content: "authoritative"}
	forged := r
	forged.Content = "event-supplied replacement"
	s := &storeStub{}
	if err := projector(s, authorityStub{row: &r}).Apply(context.Background(), event(forged, false)); err != nil {
		t.Fatal(err)
	}
	if s.row.Content != r.Content {
		t.Fatalf("persisted event content: %+v", s.row)
	}
}
