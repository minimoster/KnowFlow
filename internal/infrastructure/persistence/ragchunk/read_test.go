package ragchunk

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	es "github.com/elastic/go-elasticsearch/v8"
	milvus "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type readTransport func(*http.Request) (*http.Response, error)

func (f readTransport) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func TestESSearchRejectsFailedAndIncompleteResponses(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		fail   bool
	}{
		{"429", 429, `{"error":{"type":"too_many_requests"}}`, true},
		{"503", 503, `{"error":"unavailable"}`, true},
		{"timeout", 200, `{"timed_out":true,"hits":{"hits":[]}}`, true},
		{"partial", 200, `{"_shards":{"failed":1},"hits":{"hits":[]}}`, true},
		{"invalid", 200, `not json`, true},
		{"empty", 200, `{"timed_out":false,"hits":{"hits":[]}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			client, err := es.NewClient(es.Config{DisableRetry: true, Transport: readTransport(func(req *http.Request) (*http.Response, error) {
				calls++
				return &http.Response{StatusCode: tc.status, Header: http.Header{"X-Elastic-Product": []string{"Elasticsearch"}}, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
			})})
			if err != nil {
				t.Fatal(err)
			}
			hits, err := NewStore(nil, nil, client).SearchES(context.Background(), "question", 5)
			if calls != 1 || (err != nil) != tc.fail || len(hits) != 0 {
				t.Fatalf("calls=%d hits=%v err=%v", calls, hits, err)
			}
		})
	}
}

func TestESSearchPropagatesRequestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	client, err := es.NewClient(es.Config{DisableRetry: true, Transport: readTransport(func(req *http.Request) (*http.Response, error) {
		close(started)
		<-req.Context().Done()
		return nil, req.Context().Err()
	})})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := NewStore(nil, nil, client).SearchES(ctx, "question", 5); done <- err }()
	select {
	case <-started:
		cancel()
	case <-time.After(time.Second):
		t.Fatal("transport not called")
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("transport ignored cancellation")
	}
}

func TestLoadEvidenceRejectsScanAndIterationFailures(t *testing.T) {
	for _, scan := range []bool{false, true} {
		db, mock, err := sqlmock.New()
		if err != nil {
			t.Fatal(err)
		}
		rows := sqlmock.NewRows([]string{"id", "content", "parent", "document", "version", "section"})
		if scan {
			rows.AddRow("not an integer", "child", "parent", "doc", "v1", "s")
		} else {
			rows.AddRow(42, "child", "parent", "doc", "v1", "s").RowError(0, errors.New("connection lost while reading"))
		}
		mock.ExpectQuery("SELECT id, content").WithArgs(int64(42)).WillReturnRows(rows)
		got, err := NewStore(db, nil, nil).LoadByIDs(context.Background(), []int64{42})
		if err == nil || got != nil {
			t.Fatalf("got=%v err=%v", got, err)
		}
		if err := mock.ExpectationsWereMet(); err != nil {
			t.Fatal(err)
		}
		db.Close()
	}
}

func TestLoadEvidenceHonorsDeadline(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	mock.ExpectQuery("SELECT id, content").WithArgs(int64(42)).WillDelayFor(time.Second).WillReturnRows(sqlmock.NewRows([]string{"id", "content", "parent", "document", "version", "section"}))
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err = NewStore(db, nil, nil).LoadByIDs(ctx, []int64{42})
	if err == nil || ctx.Err() != context.DeadlineExceeded || time.Since(start) > 500*time.Millisecond {
		t.Fatalf("err=%v ctx=%v elapsed=%s", err, ctx.Err(), time.Since(start))
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

type readMilvus struct {
	milvus.Client
	search func(context.Context) ([]milvus.SearchResult, error)
}

func (m readMilvus) Search(ctx context.Context, _ string, _ []string, _ string, _ []string, _ []entity.Vector, _ string, _ entity.MetricType, _ int, _ entity.SearchParam, _ ...milvus.SearchQueryOptionFunc) ([]milvus.SearchResult, error) {
	return m.search(ctx)
}

func TestMilvusSearchPropagatesContextAndResultError(t *testing.T) {
	want := errors.New("vector query failed")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewStore(nil, readMilvus{search: func(got context.Context) ([]milvus.SearchResult, error) {
		if got != ctx {
			t.Error("lost query context")
		}
		return []milvus.SearchResult{{Err: want}}, nil
	}}, nil)
	hits, err := store.SearchMilvus(ctx, []float32{1}, 5)
	if hits != nil || !errors.Is(err, want) {
		t.Fatalf("hits=%v err=%v", hits, err)
	}
	cancel()
	_, err = store.SearchMilvus(ctx, []float32{1}, 5)
	if !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestEmptyEvidenceIDsDoNotQueryDatabase(t *testing.T) {
	rows, err := NewStore(nil, nil, nil).LoadByIDs(context.Background(), nil)
	if rows != nil || err != nil {
		t.Fatalf("%v %v", rows, err)
	}
}
