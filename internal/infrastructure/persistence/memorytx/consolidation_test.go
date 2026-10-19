package memorytx_test

import (
	"agi-assistant/internal/domain/memory/consistency"
	"agi-assistant/internal/infrastructure/persistence/memorytx"
	"context"
	"errors"
	"github.com/DATA-DOG/go-sqlmock"
	"testing"
	"time"
)

func consolidationRows(id, version int64) *sqlmock.Rows {
	now := time.Date(2026, 10, 19, 0, 0, 0, 0, time.UTC)
	return memoryRows().AddRow(id, "u1", "original", 0.5, []byte(`[1,0]`), "model", "rev", now, now, now, "preference", "{old}", "profile", false, "", false, nil, "{}", version, "hash", nil)
}

func TestConsolidationPersistsMergedMetadataBeforePublishing(t *testing.T) {
	for _, failCommit := range []bool{false, true} {
		t.Run(map[bool]string{false: "commit", true: "rollback"}[failCommit], func(t *testing.T) {
			db, mock, _ := sqlmock.New()
			defer db.Close()
			now := time.Date(2026, 10, 19, 1, 0, 0, 0, time.UTC)
			r := consistency.MemoryRecord{ID: 1, UserID: "u1", Content: "merged", Importance: 0.8, Embedding: []float64{1, 0}, Tags: []string{"old", "new"}, Supersedes: []int64{2}, LastAccessed: now}
			mock.ExpectBegin()
			mock.ExpectQuery(`FROM long_term_memory WHERE id = ANY.*FOR UPDATE`).WithArgs("{1}").WillReturnRows(consolidationRows(1, 2))
			mock.ExpectExec(`tags=\$7,supersedes=\$8,last_accessed=\$9 WHERE id=\$10 AND version=\$11`).WithArgs("merged", 0.8, []byte(`[1,0]`), int64(3), sqlmock.AnyArg(), sqlmock.AnyArg(), `{"old","new"}`, "{2}", now, int64(1), int64(2)).WillReturnResult(sqlmock.NewResult(0, 1))
			expectOutboxInsertVersion(mock, 1, 3, consistency.EventUpsertMemoryVector, consistency.TargetMilvus)
			expectOutboxInsertVersion(mock, 1, 3, consistency.EventUpsertMemoryGraphNode, consistency.TargetNeo4j)
			expectOutboxInsertVersion(mock, 1, 3, consistency.EventUpsertMemoryGraphEdges, consistency.TargetNeo4j)
			if failCommit {
				mock.ExpectCommit().WillReturnError(errors.New("commit failed"))
			} else {
				mock.ExpectCommit()
			}
			changes, err := memorytx.New(db).ApplyConsolidation(context.Background(), consistency.ConsolidationPlan{Updates: []consistency.MemoryUpdate{{Record: r, ExpectedVersion: 2}}})
			if failCommit {
				if err == nil || len(changes.Upserts)+len(changes.Deletes)+len(changes.EventIDs) != 0 {
					t.Fatalf("published rolled back data: %+v %v", changes, err)
				}
			} else {
				if err != nil || len(changes.Upserts) != 1 {
					t.Fatalf("changes=%+v err=%v", changes, err)
				}
				got := changes.Upserts[0]
				hash, _ := consistency.ComputeContentHash(got)
				if got.EmbeddingModel != "model" || got.EmbeddingRevision != "rev" || got.Category != "preference" || len(got.Tags) != 2 || len(got.Supersedes) != 1 || got.LastAccessed != now || got.ContentHash != hash {
					t.Fatalf("metadata lost: %+v", got)
				}
			}
			if err := mock.ExpectationsWereMet(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestConsolidationRejectsTenantMismatch(t *testing.T) {
	db, mock, _ := sqlmock.New()
	defer db.Close()
	mock.ExpectBegin()
	mock.ExpectQuery(`FROM long_term_memory WHERE id = ANY`).WillReturnRows(consolidationRows(1, 2))
	mock.ExpectRollback()
	changes, err := memorytx.New(db).ApplyConsolidation(context.Background(), consistency.ConsolidationPlan{Deletes: []consistency.MemoryDelete{{ID: 1, UserID: "other", ExpectedVersion: 2}}})
	if !errors.Is(err, memorytx.ErrVersionConflict) || len(changes.Deletes) != 0 {
		t.Fatalf("changes=%+v err=%v", changes, err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
