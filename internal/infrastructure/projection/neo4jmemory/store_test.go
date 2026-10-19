package neo4jmemory

import (
	"agi-assistant/internal/domain/memory/consistency"
	"context"
	"errors"
	"testing"
)

func TestUnavailableStoreNeverReportsSuccessfulProjection(t *testing.T) {
	s := NewNeo4jStore(nil)
	ctx := context.Background()
	_, _, getErr := s.Get(ctx, 1)
	_, listErr := s.ListPage(ctx, 0, 1)
	for name, err := range map[string]error{"get": getErr, "list": listErr, "upsert": s.Upsert(ctx, consistency.MemoryRecord{ID: 1}), "delete": s.Delete(ctx, 1, 1)} {
		if !errors.Is(err, ErrUnavailable) {
			t.Fatalf("%s: %v", name, err)
		}
	}
}
