package versioned

import (
	"agi-assistant/internal/domain/memory/consistency"
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

var ErrProjectionConflict = errors.New("projection: equal version hash conflict")

type Store interface {
	Get(context.Context, int64) (consistency.ProjectionState, bool, error)
	Upsert(context.Context, consistency.MemoryRecord) error
	Delete(context.Context, int64, int64) error
}

// Authority includes tombstones. Checking only the projection cannot prevent
// an old upsert from resurrecting a row after a physical projection delete.
type Authority interface {
	LoadPage(context.Context, int64, int) ([]consistency.MemoryRecord, error)
}

type Projector struct {
	Store       Store
	Source      Authority
	TargetValue consistency.Target
	Upserts     map[consistency.EventType]bool
	Deletes     map[consistency.EventType]bool
}

func (p *Projector) Target() consistency.Target { return p.TargetValue }
func (p *Projector) Apply(ctx context.Context, e consistency.OutboxEvent) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if e.Target != p.TargetValue || (!p.Upserts[e.Type] && !p.Deletes[e.Type]) {
		return fmt.Errorf("projection: unsupported event type/target")
	}
	var r consistency.MemoryRecord
	if err := json.Unmarshal(e.Payload, &r); err != nil {
		return err
	}
	if r.ID <= 0 || r.Version <= 0 || e.AggregateID != r.ID || e.AggregateVersion != r.Version || e.UserID != r.UserID {
		return fmt.Errorf("projection: payload/envelope identity mismatch")
	}
	if p.Source == nil {
		return fmt.Errorf("projection: authoritative source unavailable")
	}
	records, err := p.Source.LoadPage(ctx, r.ID-1, 1)
	if err != nil {
		return err
	}
	found := len(records) > 0 && records[0].ID == r.ID
	if found {
		latest := records[0]
		if latest.Version > r.Version {
			return nil // obsolete, including upserts preceding a tombstone
		}
		if latest.Version < r.Version {
			return fmt.Errorf("projection: event is ahead of authoritative version")
		}
		if p.Deletes[e.Type] && latest.DeletedAt == nil {
			return nil // an old/orphan repair must not remove an active memory
		}
		if latest.UserID != r.UserID || latest.ContentHash != r.ContentHash {
			return ErrProjectionConflict
		}
		if p.Upserts[e.Type] && (latest.DeletedAt != nil || latest.ContentHash == "") {
			return ErrProjectionConflict
		}
		// Never write event-supplied content when the authoritative row exists.
		r = latest
	} else if p.Upserts[e.Type] {
		return fmt.Errorf("projection: authoritative memory missing")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	current, ok, err := p.Store.Get(ctx, r.ID)
	if err != nil {
		return err
	}
	if ok && r.Version < current.Version {
		return nil
	}
	if p.Upserts[e.Type] {
		if ok && r.Version == current.Version {
			if r.ContentHash != current.ContentHash {
				return ErrProjectionConflict
			}
			return nil
		}
		return p.Store.Upsert(ctx, r)
	}
	if p.Deletes[e.Type] {
		if !ok {
			return nil
		}
		if r.Version < current.Version {
			return nil
		}
		return p.Store.Delete(ctx, r.ID, r.Version)
	}
	return nil
}
