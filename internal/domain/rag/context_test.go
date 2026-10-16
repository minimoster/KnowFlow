package rag

import (
	"context"
	"errors"
	"testing"
	"time"

	"agi-assistant/config"
	"agi-assistant/internal/infrastructure/persistence/ragchunk"
)

// Embedding the unused methods keeps the fake small. Only read methods are used;
// any accidental indexing/service initialization panics rather than doing I/O.
type queryRepo struct{ ragchunk.Repo }

func (queryRepo) PGAvailable() bool     { return true }
func (queryRepo) ESAvailable() bool     { return true }
func (queryRepo) MilvusAvailable() bool { return false }
func (queryRepo) SearchES(string, int) ([]ragchunk.ESHit, error) {
	return []ragchunk.ESHit{{PGID: 1, Score: 1}, {PGID: 2, Score: .5}}, nil
}
func (queryRepo) LoadByIDs([]int64) ([]ragchunk.Row, error) {
	return []ragchunk.Row{{ID: 1, Content: "evidence one"}, {ID: 2, Content: "evidence two"}}, nil
}

func TestRAGCallbacksReceiveTaskContextAndPropagateFailure(t *testing.T) {
	for _, stage := range []string{"rewrite", "rerank", "generate"} {
		t.Run(stage, func(t *testing.T) {
			e := NewEngine(&config.APIConfig{RAGConfig: config.RAGConfig{ChunkSize: 200, TopK: 5}}, queryRepo{}, nil)
			e.Loaded = true
			started := make(chan struct{})
			type taskKey struct{}
			ctx, cancel := context.WithCancel(context.WithValue(context.Background(), taskKey{}, "task-42"))
			defer cancel()
			callback := func(got context.Context, _, _ string) (string, error) {
				if got.Value(taskKey{}) != "task-42" {
					t.Error("lost task context")
				}
				close(started)
				<-got.Done()
				return "", got.Err()
			}
			switch stage {
			case "rewrite":
				e.SetRewriter(NewLLMRewriter(callback, 3))
			case "rerank":
				e.SetReranker(NewLLMReranker(callback, 200))
			case "generate":
				e.SetGenerateFn(callback)
			}
			done := make(chan error, 1)
			go func() {
				answer, _, err := e.QueryWithHistory(ctx, "question", nil)
				if answer != "" {
					t.Error("failure returned answer")
				}
				done <- err
			}()
			select {
			case <-started:
				cancel()
			case <-time.After(time.Second):
				t.Fatal("callback not reached")
			}
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("got %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("callback did not cancel")
			}
		})
	}
}

func TestRAGAPIErrorIsNotSuccessfulAnswer(t *testing.T) {
	e := NewEngine(&config.APIConfig{RAGConfig: config.RAGConfig{ChunkSize: 200, TopK: 5}}, queryRepo{}, nil)
	e.Loaded = true
	want := errors.New("LLM API HTTP 429")
	e.SetGenerateFn(func(context.Context, string, string) (string, error) { return "", want })
	answer, results, err := e.QueryWithHistory(context.Background(), "question", nil)
	if !errors.Is(err, want) || answer != "" || len(results) != 2 {
		t.Fatalf("answer=%q results=%d err=%v", answer, len(results), err)
	}
}
