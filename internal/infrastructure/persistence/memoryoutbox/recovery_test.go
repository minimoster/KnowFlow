package memoryoutbox_test

import (
	"context"
	"database/sql/driver"
	"errors"
	"regexp"
	"testing"
	"time"

	"agi-assistant/internal/domain/memory/consistency"
	"agi-assistant/internal/infrastructure/persistence/memoryoutbox"
	"github.com/DATA-DOG/go-sqlmock"
)

func TestClaimRecoversExpiredProcessingWithFreshOwnership(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	repo := memoryoutbox.New(db)
	var last string
	for i := 0; i < 2; i++ {
		mock.ExpectBegin()
		mock.ExpectQuery(`status='processing'.*locked_at<=NOW\(\)-\(\$3 \* INTERVAL '1 microsecond'\).*FOR UPDATE SKIP LOCKED`).
			WithArgs("milvus", 1, time.Minute.Microseconds()).
			WillReturnRows(sqlmock.NewRows([]string{"id", "event_id", "aggregate_id", "user_id", "aggregate_version", "event_type", "target", "payload", "attempts", "created_at"}).
				AddRow(1, "e1", 41, "u1", 2, "upsert_memory_vector", "milvus", []byte(`{}`), 2, time.Now()))
		mock.ExpectExec(`locked_by=\$1, locked_at=NOW\(\)`).WithArgs(sqlmock.AnyArg(), "{1}").WillReturnResult(sqlmock.NewResult(0, 1))
		mock.ExpectCommit()
		events, err := repo.Claim(context.Background(), consistency.TargetMilvus, "same-worker", 1, time.Minute)
		if err != nil || len(events) != 1 || events[0].ClaimToken == "" || events[0].ClaimToken == last || events[0].Attempts != 2 {
			t.Fatalf("events=%+v err=%v previous=%q", events, err, last)
		}
		last = events[0].ClaimToken
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestCompletionRequiresCurrentClaimForEveryOutcome(t *testing.T) {
	for _, outcome := range []string{"processed", "retry", "dead"} {
		for _, affected := range []int64{0, 1} {
			t.Run(outcome+string(rune('0'+affected)), func(t *testing.T) {
				db, mock, _ := sqlmock.New()
				defer db.Close()
				repo := memoryoutbox.New(db)
				e := consistency.OutboxEvent{EventID: "e1", ClaimToken: "claim-1"}
				at := time.Now()
				args := []driver.Value{"e1", "claim-1"}
				if outcome == "retry" {
					args = append(args, at, "failed")
				}
				if outcome == "dead" {
					args = append(args, "failed")
				}
				mock.ExpectExec(`WHERE event_id=\$1 AND locked_by=\$2 AND status='processing'`).WithArgs(args...).WillReturnResult(sqlmock.NewResult(0, affected))
				var err error
				switch outcome {
				case "processed":
					err = repo.MarkProcessed(context.Background(), e)
				case "retry":
					err = repo.MarkRetry(context.Background(), e, "failed", at)
				case "dead":
					err = repo.MarkDead(context.Background(), e, "failed")
				}
				if affected == 0 && !errors.Is(err, memoryoutbox.ErrLeaseLost) {
					t.Fatalf("err=%v", err)
				}
				if affected == 1 && err != nil {
					t.Fatal(err)
				}
				if err := mock.ExpectationsWereMet(); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

func TestClaimCommitFailureReturnsNoEvents(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	mock.ExpectBegin()
	mock.ExpectQuery(`FOR UPDATE SKIP LOCKED`).WillReturnRows(sqlmock.NewRows([]string{"id", "event_id", "aggregate_id", "user_id", "aggregate_version", "event_type", "target", "payload", "attempts", "created_at"}).AddRow(1, "e1", 1, "u1", 1, "upsert_memory_vector", "milvus", []byte(`{}`), 0, time.Now()))
	mock.ExpectExec(`UPDATE memory_outbox`).WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit().WillReturnError(errors.New("commit disconnected"))
	events, err := memoryoutbox.New(db).Claim(context.Background(), consistency.TargetMilvus, "w", 1, time.Minute)
	if err == nil || len(events) != 0 {
		t.Fatalf("events=%+v err=%v", events, err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

type uuidArgument struct{}

func (uuidArgument) Match(v driver.Value) bool {
	s, ok := v.(string)
	return ok && regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`).MatchString(s)
}

func TestRepairIDFitsPostgresUUIDColumn(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	mock.ExpectExec(`INSERT INTO memory_outbox`).WithArgs(uuidArgument{}, int64(1), "u1", int64(2), "upsert_memory_vector", "milvus", []byte(`{}`), "repair-key").WillReturnResult(sqlmock.NewResult(0, 1))
	ok, err := memoryoutbox.New(db).EnqueueRepair(context.Background(), consistency.OutboxEvent{AggregateID: 1, UserID: "u1", AggregateVersion: 2, Type: consistency.EventUpsertMemoryVector, Target: consistency.TargetMilvus, Payload: []byte(`{}`)}, "repair-key")
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
