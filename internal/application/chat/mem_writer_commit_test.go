package chat

import (
	"agi-assistant/config"
	"agi-assistant/internal/domain/memory/consistency"
	"agi-assistant/internal/domain/memory/longterm"
	"agi-assistant/internal/domain/memory/preference"
	"agi-assistant/internal/infrastructure/llm"
	"agi-assistant/internal/infrastructure/persistence/memorytx"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

type extractionTransport func(*http.Request) (*http.Response, error)

func (f extractionTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type observingStore struct {
	fakeMemoryStore
	before func()
	calls  int
}

type observingPreferences struct{ writes int }

func (p *observingPreferences) Save(string, string, string) { p.writes++ }
func (*observingPreferences) Load(string) map[string]string { return nil }

func (s *observingStore) Create(ctx context.Context, c memorytx.CreateCommand) (consistency.CommittedChangeSet, error) {
	s.calls++
	if s.before != nil {
		s.before()
	}
	return s.fakeMemoryStore.Create(ctx, c)
}

func TestExtractedMemoryCommitBoundary(t *testing.T) {
	for _, tc := range []struct {
		name, key, value, source string
		embedStatus              int
		dbErr                    bool
		canceled                 bool
		commits, prefs           int
	}{
		{"unsafe", "密码", "secret", "user", 200, false, false, 0, 0},
		{"embedding failure", "喜欢", "咖啡", "user", 429, false, false, 0, 0},
		{"database failure", "喜欢", "咖啡", "user", 200, true, false, 1, 0},
		{"canceled", "喜欢", "咖啡", "user", 200, false, true, 0, 0},
		{"user committed", "喜欢", "咖啡", "user", 200, false, false, 1, 1},
		{"exchange never preference", "喜欢", "咖啡", "exchange", 200, false, false, 1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.APIConfig{EmbeddingConfig: config.EmbeddingConfig{EmbeddingAPIUrl: "http://offline.invalid/embeddings", EmbeddingAPIKey: "offline-test", EmbeddingModel: "fixture"}}
			embedCalls := 0
			client := llm.NewWithHTTPClient(cfg, &http.Client{Transport: extractionTransport(func(*http.Request) (*http.Response, error) {
				embedCalls++
				return &http.Response{StatusCode: tc.embedStatus, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"data":[{"embedding":[1,0]}]}`))}, nil
			})})
			store := &observingStore{fakeMemoryStore: fakeMemoryStore{createChanges: consistency.CommittedChangeSet{Upserts: []consistency.MemoryRecord{{ID: 7, UserID: "u1", Content: "committed", Version: 1, ContentHash: "hash"}}}}}
			if tc.dbErr {
				store.createErr = errors.New("postgres unavailable")
			}
			prefs := &observingPreferences{}
			agent := &UnifiedAgent{cfg: cfg, llm: client, mem: &memoryStack{ltm: longterm.New(), prefByUser: make(map[string]*preference.Preference)}, repos: &repoBundle{memoryTx: store, pref: prefs}}
			store.before = func() {
				if agent.mem.ltm.Count() != 0 || len(agent.mem.prefSnapshot("u1")) != 0 || prefs.writes != 0 {
					t.Fatal("cache mutated before durable commit")
				}
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.canceled {
				cancel()
			}
			_, err := agent.commitExtractedKV(ctx, "u1", tc.source, tc.key, tc.value, 0.8, true)
			wantErr := tc.commits == 0 || tc.dbErr
			if (err != nil) != wantErr || store.calls != tc.commits || len(agent.mem.prefSnapshot("u1")) != tc.prefs || prefs.writes != tc.prefs {
				t.Fatalf("err=%v calls=%d prefs=%v", err, store.calls, agent.mem.prefSnapshot("u1"))
			}
			if wantErr && agent.mem.ltm.Count() != 0 {
				t.Fatal("failed write visible in LTM")
			}
			if (tc.name == "unsafe" || tc.canceled) && embedCalls != 0 {
				t.Fatal("rejected extraction called embedding")
			}
		})
	}
}
