package memoryprojection

import (
	"agi-assistant/internal/domain/memory/consistency"
	"context"
	"errors"
	"testing"
	"time"
)

type projectorFunc func(context.Context, consistency.OutboxEvent) error

func (projectorFunc) Target() consistency.Target                                   { return consistency.TargetMilvus }
func (f projectorFunc) Apply(ctx context.Context, e consistency.OutboxEvent) error { return f(ctx, e) }

func TestInterruptedBatchDoesNotAcknowledgeOrConsumeAttempts(t *testing.T) {
	for _, mode := range []string{"cancel", "lease"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			repo := &fakeOutbox{events: []consistency.OutboxEvent{{EventID: "e1", Attempts: 9}, {EventID: "e2"}}}
			calls := 0
			p := projectorFunc(func(ctx context.Context, _ consistency.OutboxEvent) error {
				calls++
				if mode == "cancel" {
					cancel()
				}
				<-ctx.Done()
				return nil // even a late success must not acknowledge the lease
			})
			lease := time.Minute
			if mode == "lease" {
				lease = 10 * time.Millisecond
			}
			err := NewWorker(repo, p, Config{WorkerID: "w", Lease: lease}).RunOnce(ctx)
			if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("err=%v", err)
			}
			if calls != 1 || len(repo.processed)+len(repo.retried)+len(repo.dead) != 0 {
				t.Fatalf("calls=%d repo=%+v", calls, repo)
			}
		})
	}
}

func TestProjectorPanicBecomesRetry(t *testing.T) {
	repo := &fakeOutbox{events: []consistency.OutboxEvent{{EventID: "e1"}}}
	err := NewWorker(repo, projectorFunc(func(context.Context, consistency.OutboxEvent) error { panic("broken") }), Config{WorkerID: "w"}).RunOnce(context.Background())
	if err != nil || len(repo.retried) != 1 || len(repo.processed) != 0 {
		t.Fatalf("repo=%+v err=%v", repo, err)
	}
}
