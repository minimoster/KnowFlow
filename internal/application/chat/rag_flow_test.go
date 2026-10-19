package chat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"agi-assistant/config"
	"agi-assistant/internal/domain/rag"
	"agi-assistant/internal/infrastructure/llm"
	"agi-assistant/internal/infrastructure/persistence/chathistory"
	"agi-assistant/internal/infrastructure/persistence/ragchunk"
	"agi-assistant/internal/usercontext"
)

type ragFlowKey struct{}

// Unused Repo methods deliberately panic through the embedded nil interface.
// These tests must not initialize services, index documents, or write externally.
type ragFlowRepo struct {
	ragchunk.Repo
	t                     *testing.T
	mode                  string
	stage                 string
	failure               error
	empty, blank, missing bool
	onRead                func(context.Context, string) error
	calls                 []string
}

func (r *ragFlowRepo) PGAvailable() bool     { return r.mode != "unavailable" }
func (r *ragFlowRepo) ESAvailable() bool     { return r.mode == "keyword" || r.mode == "hybrid" }
func (r *ragFlowRepo) MilvusAvailable() bool { return r.mode == "semantic" || r.mode == "hybrid" }
func (r *ragFlowRepo) read(ctx context.Context, stage string) error {
	r.t.Helper()
	if ctx.Value(ragFlowKey{}) != "request-42" {
		r.t.Error("retrieval lost request context")
	}
	if _, ok := ctx.Deadline(); !ok {
		r.t.Error("retrieval lost request deadline")
	}
	r.calls = append(r.calls, stage)
	if r.onRead != nil {
		if err := r.onRead(ctx, stage); err != nil {
			return err
		}
	}
	if stage == r.stage {
		return r.failure
	}
	return ctx.Err()
}
func (r *ragFlowRepo) SearchES(ctx context.Context, q string, k int) ([]ragchunk.ESHit, error) {
	if err := r.read(ctx, "keyword"); err != nil {
		return nil, err
	}
	if r.empty {
		return nil, nil
	}
	return []ragchunk.ESHit{{PGID: 42, Score: 1}}, nil
}
func (r *ragFlowRepo) SearchMilvus(ctx context.Context, v []float32, k int) ([]ragchunk.MilvusHit, error) {
	if err := r.read(ctx, "vector"); err != nil {
		return nil, err
	}
	if r.empty {
		return nil, nil
	}
	return []ragchunk.MilvusHit{{ID: 42, Distance: 1}}, nil
}
func (r *ragFlowRepo) LoadByIDs(ctx context.Context, ids []int64) ([]ragchunk.Row, error) {
	if err := r.read(ctx, "evidence"); err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(ids, []int64{42}) {
		r.t.Errorf("unexpected evidence IDs: %v", ids)
	}
	if r.missing {
		return nil, nil
	}
	if r.blank {
		return []ragchunk.Row{{ID: 42, Content: " \n\t"}}, nil
	}
	return []ragchunk.Row{{ID: 42, Content: "child evidence", ParentContent: "parent evidence: backups retained for seven days"}}, nil
}

type ragFlowTransport func(*http.Request) (*http.Response, error)

func (f ragFlowTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type ragFlowEvents struct{ calls int }

func (p *ragFlowEvents) Publish(string, string) { p.calls++ }

func ragFlowAgent(t *testing.T, repo *ragFlowRepo, httpCall ragFlowTransport) (*UnifiedAgent, *ragFlowEvents) {
	t.Helper()
	cfg := &config.APIConfig{
		LLMConfig: config.LLMConfig{LLMAPIKey: "offline-test", LLMAPIUrl: "http://offline.invalid", LLMModel: "test"},
		RAGConfig: config.RAGConfig{ChunkSize: 200, TopK: 5},
	}
	pub := &ragFlowEvents{}
	a := &UnifiedAgent{cfg: cfg, mem: newTestStack(), runtime: newTaskRuntime(), tools: newToolRegistry(nil), repos: &repoBundle{events: pub}}
	a.llm = llm.NewWithHTTPClient(cfg, &http.Client{Transport: httpCall})
	a.rag = rag.NewEngine(cfg, repo, nil)
	a.rag.Loaded = true
	a.wireRAGCallbacks() // actual production synthesis callback and real-mode HTTP client
	// Keep vector input deterministic; no embedding service or credentials involved.
	a.rag.SetEmbedFn(func(ctx context.Context, _ string) ([]float64, error) {
		if err := repo.read(ctx, "embedding"); err != nil {
			return nil, err
		}
		return []float64{1, 0}, nil
	})
	return a, pub
}

func ragFlowContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithValue(context.Background(), ragFlowKey{}, "request-42"), 3*time.Second)
}

func TestKnowledgeQuestionFlow(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, mode := range []string{"keyword", "semantic", "hybrid"} {
			for _, outcome := range []string{"answer", "unloaded", "empty", "blank", "missing", "dependency", "generation429", "emptyAnswer"} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", mode, outcome, stream), func(t *testing.T) {
					repo := &ragFlowRepo{t: t, mode: mode, empty: outcome == "empty", blank: outcome == "blank", missing: outcome == "missing"}
					if outcome == "dependency" {
						repo.stage, repo.failure = "evidence", errors.New("postgres offline")
					}
					modelCalls := 0
					a, pub := ragFlowAgent(t, repo, func(req *http.Request) (*http.Response, error) {
						modelCalls++
						if req.Context().Value(ragFlowKey{}) != "request-42" {
							t.Error("model lost request context")
						}
						body, _ := io.ReadAll(req.Body)
						if !strings.Contains(string(body), "parent evidence") || !strings.Contains(string(body), "How long are backups retained?") {
							t.Errorf("model did not receive question and retrieved parent evidence: %s", body)
						}
						status, content := 200, `{"choices":[{"message":{"content":"Seven days (offline fixture)."}}]}`
						if outcome == "generation429" {
							status, content = 429, `{"error":"rate limited"}`
						}
						if outcome == "emptyAnswer" {
							content = `{"choices":[{"message":{"content":"   "}}]}`
						}
						return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(content))}, nil
					})
					a.rag.Loaded = outcome != "unloaded"
					ctx, cancel := ragFlowContext()
					defer cancel()
					var events []StreamEvent
					var resp *Response
					if stream {
						resp = a.ProcessStream(ctx, "How long are backups retained?", ChatOptions{UseRAG: true}, func(e StreamEvent) { events = append(events, e) })
					} else {
						resp = a.ProcessContext(ctx, "How long are backups retained?", ChatOptions{UseRAG: true})
					}
					failed := outcome == "missing" || outcome == "dependency" || outcome == "generation429" || outcome == "emptyAnswer"
					if resp.Mode != "rag" || (resp.Error != "") != failed || resp.Interrupted {
						t.Fatalf("response=%+v", resp)
					}
					wantCalls := 0
					if outcome == "answer" || outcome == "generation429" || outcome == "emptyAnswer" {
						wantCalls = 1
					}
					if modelCalls != wantCalls {
						t.Fatalf("modelCalls=%d want=%d", modelCalls, wantCalls)
					}
					if failed {
						if pub.calls != 0 || len(resp.SearchResults) != 0 {
							t.Fatalf("failed request finalized: %+v", resp)
						}
					} else if pub.calls != 1 {
						t.Fatal("successful request did not finalize exactly once")
					}
					if outcome == "answer" && (resp.Answer != "Seven days (offline fixture)." || len(resp.SearchResults) != 1 || !strings.Contains(resp.SearchResults[0].Chunk.Content, "parent evidence")) {
						t.Fatalf("%+v", resp)
					}
					if outcome == "unloaded" && (len(repo.calls) != 0 || !strings.Contains(resp.Answer, "知识库为空")) {
						t.Fatalf("%+v calls=%v", resp, repo.calls)
					}
					if (outcome == "empty" || outcome == "blank") && resp.Answer != "知识库中未找到相关内容。" {
						t.Fatalf("%+v", resp)
					}
					if stream {
						var types []string
						for _, e := range events {
							types = append(types, e.Type)
						}
						want := []string{"route", "rag_result", "token", "done"}
						if failed {
							want = []string{"route", "error", "done"}
						}
						if !reflect.DeepEqual(types, want) || events[len(events)-1].Data != resp {
							t.Fatalf("events=%v", types)
						}
					}
				})
			}
		}
	}
}

func TestKnowledgeQuestionDependencyFailuresDoNotGenerate(t *testing.T) {
	for _, stage := range []string{"embedding", "vector", "keyword", "evidence", "unavailable"} {
		t.Run(stage, func(t *testing.T) {
			repo := &ragFlowRepo{t: t, mode: "hybrid", stage: stage, failure: errors.New(stage + " failed")}
			if stage == "unavailable" {
				repo.mode = stage
			}
			a, pub := ragFlowAgent(t, repo, func(*http.Request) (*http.Response, error) {
				t.Error("model called after retrieval failure")
				return nil, errors.New("unexpected model call")
			})
			ctx, cancel := ragFlowContext()
			defer cancel()
			resp := a.ProcessContext(ctx, "question", ChatOptions{UseRAG: true})
			if resp.Error == "" || pub.calls != 0 {
				t.Fatalf("%+v published=%d", resp, pub.calls)
			}
		})
	}
}

func TestKnowledgeQuestionCancellationStopsPipeline(t *testing.T) {
	for _, stage := range []string{"embedding", "vector", "keyword", "evidence", "generation", "lateGeneration"} {
		t.Run(stage, func(t *testing.T) {
			ctx, cancel := ragFlowContext()
			defer cancel()
			started := make(chan struct{})
			block := func(ctx context.Context) { close(started); <-ctx.Done() }
			repo := &ragFlowRepo{t: t, mode: "hybrid"}
			repo.onRead = func(ctx context.Context, at string) error {
				if at == stage {
					block(ctx)
				}
				return ctx.Err()
			}
			a, pub := ragFlowAgent(t, repo, func(req *http.Request) (*http.Response, error) {
				block(req.Context())
				return nil, req.Context().Err()
			})
			if stage == "lateGeneration" {
				a.rag.SetGenerateFn(func(ctx context.Context, _, _ string) (string, error) { block(ctx); return "late answer", nil })
			}
			var events []string
			done := make(chan *Response, 1)
			go func() {
				done <- a.ProcessStream(ctx, "question", ChatOptions{UseRAG: true}, func(e StreamEvent) { events = append(events, e.Type) })
			}()
			select {
			case <-started:
				a.Cancel()
			case <-time.After(time.Second):
				t.Fatal("stage not reached")
			}
			select {
			case resp := <-done:
				if !resp.Interrupted || !strings.Contains(resp.Error, context.Canceled.Error()) || pub.calls != 0 || !reflect.DeepEqual(events, []string{"route", "error", "done"}) {
					t.Fatalf("resp=%+v events=%v", resp, events)
				}
			case <-time.After(time.Second):
				t.Fatal("canceled request did not return")
			}
		})
	}
}

type ragFlowHistory struct {
	chathistory.Repo
	roles []string
}

func (h *ragFlowHistory) Save(_, role, _ string) { h.roles = append(h.roles, role) }

func TestKnowledgeQuestionFailureDoesNotBecomeAssistantMemory(t *testing.T) {
	repo := &ragFlowRepo{t: t, mode: "keyword", stage: "keyword", failure: errors.New("ES offline")}
	prefRead := make(chan struct{}, 1)
	a, pub := ragFlowAgent(t, repo, func(*http.Request) (*http.Response, error) {
		// The authenticated input-side preference extraction is independent of RAG.
		// Return no preferences; no model/network is contacted by this transport.
		prefRead <- struct{}{}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"{}"}}]}`))}, nil
	})
	history := &ragFlowHistory{}
	a.repos.chat = history
	ctx, cancel := ragFlowContext()
	defer cancel()
	ctx = usercontext.With(ctx, "alice", "Alice")
	resp := a.ProcessContext(ctx, "How long are backups retained?", ChatOptions{UseRAG: true})
	if resp.Error == "" || pub.calls != 0 || !reflect.DeepEqual(history.roles, []string{"user"}) || a.mem.STM("alice").Count() != 1 || a.mem.ltm.Count() != 0 {
		t.Fatalf("response=%+v roles=%v", resp, history.roles)
	}
	select {
	case <-prefRead:
	case <-time.After(time.Second):
		t.Fatal("input preference extraction not reached")
	}
}
