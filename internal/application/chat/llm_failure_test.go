package chat

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"agi-assistant/config"
	"agi-assistant/internal/domain/graph"
	"agi-assistant/internal/infrastructure/llm"
)

type failingTransport struct{}

func (failingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: 429, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("rate limited"))}, nil
}
func failureAgent() *UnifiedAgent {
	cfg := &config.APIConfig{LLMConfig: config.LLMConfig{LLMAPIKey: "offline-test", LLMAPIUrl: "http://offline.invalid", LLMModel: "test"}}
	return &UnifiedAgent{cfg: cfg, llm: llm.NewWithHTTPClient(cfg, &http.Client{Transport: failingTransport{}})}
}

func TestSubAgentLLMFailureReturnsError(t *testing.T) {
	a := failureAgent()
	for _, worker := range []SubAgent{&writerAgent{a}, &reviewAgent{a}, &researchAgent{a}} {
		t.Run(worker.Name(), func(t *testing.T) {
			out, err := worker.Run(context.Background(), SubAgentTask{Goal: "report", Query: "question"})
			if err == nil || !strings.Contains(err.Error(), "429") || out != "" {
				t.Fatalf("%q, %v", out, err)
			}
		})
	}
}

func TestSubAgentCancellationReturnsErrorEvenInMockMode(t *testing.T) {
	a := failureAgent()
	a.cfg.LLMMode = "mock"
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, worker := range []SubAgent{&writerAgent{a}, &reviewAgent{a}, &researchAgent{a}, &docAgent{a}} {
		out, err := worker.Run(ctx, SubAgentTask{Goal: "report"})
		if out != "" || !errors.Is(err, context.Canceled) {
			t.Fatalf("%s: %q, %v", worker.Name(), out, err)
		}
	}
}

type countedDocAgent struct{ calls *int }

func (*countedDocAgent) Name() string        { return "doc_agent" }
func (*countedDocAgent) Description() string { return "offline write counter" }
func (a *countedDocAgent) Run(context.Context, SubAgentTask) (string, error) {
	(*a.calls)++
	return "saved", nil
}

func TestFailedWriterDoesNotRunDependentDocumentAgent(t *testing.T) {
	a := failureAgent()
	a.runtime = newTaskRuntime()
	a.repos = &repoBundle{snap: noopSnapshotRepo{}}
	a.subagents = newSubAgentRegistry()
	a.subagents.register(&writerAgent{a})
	writes := 0
	a.subagents.register(&countedDocAgent{calls: &writes})
	tg := graph.NewTaskGraph([]*graph.Node{
		{ID: "writer", Type: graph.NodeTypeSubAgent, AgentName: "writer_agent"},
		{ID: "doc", Type: graph.NodeTypeSubAgent, AgentName: "doc_agent", DependsOn: []graph.NodeID{"writer"}},
	})
	a.cfg.MaxRetries = 1
	rt := NewGraphRuntime(tg, a, GraphConfig{MaxParallel: 1}, nil, &TaskState{}, nil)
	rt.allowSubAgents = true
	rt.Execute(context.Background())
	if writes != 0 || tg.Nodes["writer"].Status != graph.StatusFailed || tg.Nodes["doc"].Status != graph.StatusFailed {
		t.Fatalf("writes=%d writer=%+v doc=%+v", writes, tg.Nodes["writer"], tg.Nodes["doc"])
	}
}

func TestDispatchPropagatesGenerationFailure(t *testing.T) {
	a := failureAgent()
	for _, stream := range []bool{false, true} {
		var onEvent func(StreamEvent)
		if stream {
			onEvent = func(StreamEvent) {}
		}
		resp := &Response{}
		err := a.dispatch(context.Background(), preparedRequest{mode: "chat"}, resp, onEvent)
		if err == nil || resp.Answer != "" {
			t.Fatalf("%q, %v", resp.Answer, err)
		}
	}
}

func TestRunOnceFailureSkipsAnswerPersistence(t *testing.T) {
	a := failureAgent()
	a.mem = newTestStack()
	a.tools = newToolRegistry(nil)
	// A nil repos bundle would panic if finalize tried to persist a failed answer.
	var events []string
	resp := a.runOnce(context.Background(), "hello", ChatOptions{}, func(e StreamEvent) { events = append(events, e.Type) })
	if !strings.Contains(resp.Error, "429") || !strings.HasPrefix(resp.Answer, "请求失败") {
		t.Fatalf("%+v", resp)
	}
	if len(events) < 2 || events[len(events)-2] != "error" || events[len(events)-1] != "done" {
		t.Fatalf("events=%v", events)
	}
}
