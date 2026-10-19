package memoryoutbox

import (
	"agi-assistant/internal/domain/memory/consistency"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/lib/pq"
	"time"
)

type Repository struct{ db *sql.DB }

var ErrLeaseLost = errors.New("outbox: claim ownership lost")

func New(db *sql.DB) *Repository { return &Repository{db: db} }

func (r *Repository) Claim(ctx context.Context, target consistency.Target, worker string, limit int, lease time.Duration) (events []consistency.OutboxEvent, err error) {
	if r.db == nil {
		return nil, fmt.Errorf("outbox unavailable")
	}
	if worker == "" || limit <= 0 || lease < time.Microsecond {
		return nil, fmt.Errorf("outbox: worker, positive limit and lease are required")
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, err
	}
	token := worker + ":" + hex.EncodeToString(nonce[:])
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()
	// Use the database clock for both lease acquisition and recovery. Workers
	// consuming a target must share the same lease configuration.
	rows, err := tx.QueryContext(ctx, `SELECT id,event_id,aggregate_id,user_id,aggregate_version,event_type,target,payload,attempts,created_at FROM memory_outbox WHERE target=$1 AND ((status='pending' AND available_at<=NOW()) OR (status='processing' AND (locked_at IS NULL OR locked_at<=NOW()-($3 * INTERVAL '1 microsecond')))) ORDER BY id LIMIT $2 FOR UPDATE SKIP LOCKED`, string(target), limit, lease.Microseconds())
	if err != nil {
		return nil, err
	}
	var ids []int64
	for rows.Next() {
		var e consistency.OutboxEvent
		var typ, targetText string
		var payload []byte
		if err = rows.Scan(&e.ID, &e.EventID, &e.AggregateID, &e.UserID, &e.AggregateVersion, &typ, &targetText, &payload, &e.Attempts, &e.CreatedAt); err != nil {
			rows.Close()
			return nil, err
		}
		e.Type = consistency.EventType(typ)
		e.Target = consistency.Target(targetText)
		e.Payload = json.RawMessage(payload)
		e.ClaimToken = token
		events = append(events, e)
		ids = append(ids, e.ID)
	}
	rows.Close()
	if err = rows.Err(); err != nil {
		return nil, err
	}
	if len(ids) > 0 {
		_, err = tx.ExecContext(ctx, `UPDATE memory_outbox SET status = 'processing', locked_by=$1, locked_at=NOW() WHERE id = ANY($2)`, token, pq.Array(ids))
		if err != nil {
			return nil, err
		}
	}
	if err = tx.Commit(); err != nil {
		return nil, err
	}
	return events, nil
}
func (r *Repository) MarkProcessed(ctx context.Context, e consistency.OutboxEvent) error {
	return r.finish(ctx, e, `UPDATE memory_outbox SET status='processed',processed_at=NOW(),locked_at=NULL,locked_by=NULL WHERE event_id=$1 AND locked_by=$2 AND status='processing'`)
}
func (r *Repository) MarkRetry(ctx context.Context, e consistency.OutboxEvent, msg string, at time.Time) error {
	return r.finish(ctx, e, `UPDATE memory_outbox SET status='pending',attempts=attempts+1,available_at=$3,last_error=$4,locked_at=NULL,locked_by=NULL WHERE event_id=$1 AND locked_by=$2 AND status='processing'`, at, msg)
}
func (r *Repository) MarkDead(ctx context.Context, e consistency.OutboxEvent, msg string) error {
	return r.finish(ctx, e, `UPDATE memory_outbox SET status='dead',attempts=attempts+1,last_error=$3,locked_at=NULL,locked_by=NULL WHERE event_id=$1 AND locked_by=$2 AND status='processing'`, msg)
}
func (r *Repository) finish(ctx context.Context, e consistency.OutboxEvent, q string, args ...any) error {
	if r.db == nil {
		return fmt.Errorf("outbox unavailable")
	}
	if e.ClaimToken == "" {
		return ErrLeaseLost
	}
	res, err := r.db.ExecContext(ctx, q, append([]any{e.EventID, e.ClaimToken}, args...)...)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n != 1 {
		return ErrLeaseLost
	}
	return nil
}

func (r *Repository) EnqueueRepair(ctx context.Context, e consistency.OutboxEvent, key string) (bool, error) {
	if r.db == nil {
		return false, fmt.Errorf("outbox unavailable")
	}
	if e.EventID == "" {
		var id [16]byte
		if _, err := rand.Read(id[:]); err != nil {
			return false, err
		}
		id[6] = (id[6] & 0x0f) | 0x40
		id[8] = (id[8] & 0x3f) | 0x80
		e.EventID = fmt.Sprintf("%x-%x-%x-%x-%x", id[0:4], id[4:6], id[6:8], id[8:10], id[10:16])
	}
	res, err := r.db.ExecContext(ctx, `INSERT INTO memory_outbox
		(event_id,aggregate_id,user_id,aggregate_version,event_type,target,payload,repair_dedupe_key)
		VALUES($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (repair_dedupe_key) DO NOTHING`,
		e.EventID, e.AggregateID, e.UserID, e.AggregateVersion, string(e.Type), string(e.Target), e.Payload, key)
	if err != nil {
		return false, err
	}
	n, err := res.RowsAffected()
	return n == 1, err
}
