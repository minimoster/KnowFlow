package chat

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"agi-assistant/internal/domain/graph"
	"agi-assistant/internal/domain/rag"
	"agi-assistant/internal/domain/tool"
)

func TestCancellationBeforeDispatchReportsError(t *testing.T) {
	a := failureAgent()
	a.mem = newTestStack()
	a.tools = newToolRegistry(nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var events []string
	r := a.runOnce(ctx, "hello", ChatOptions{}, func(e StreamEvent) {
		events = append(events, e.Type)
		if e.Type == "route" {
			cancel()
		}
	})
	if !r.Interrupted || r.Error != context.Canceled.Error() || len(events) != 3 || events[1] != "error" || events[2] != "done" {
		t.Fatalf("response=%+v events=%v", r, events)
	}
}

func TestArtifactCancellationDoesNotCompleteTask(t *testing.T) {
	a := failureAgent()
	a.cfg.LLMMode = "mock"
	a.cfg.MaxRetries = 1
	a.cfg.GraphMaxParallel = 1
	a.cfg.SandboxArtifactHostDir = t.TempDir()
	a.runtime = newTaskRuntime()
	a.repos = &repoBundle{snap: noopSnapshotRepo{}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	artifactStarted, artifactCanceled := false, false
	ts := map[string]tool.Tool{"search_web": {ExecuteCtx: func(context.Context, map[string]interface{}) (string, error) { return "offline evidence", nil }}}
	answer, steps, task, err := a.runReAct(ctx, "搜索并写一份报告", ts, "", nil, func(e StreamEvent) {
		data, ok := e.Data.(map[string]string)
		if ok && data["id"] == "artifact" {
			if e.Type == "node_start" {
				artifactStarted = true
				cancel()
			}
			if e.Type == "node_done" {
				artifactCanceled = data["status"] == "cancelled"
			}
		}
	}, reactOpts{})
	if !artifactStarted || !artifactCanceled || !errors.Is(err, context.Canceled) || answer != "" || task == nil || task.Status != "interrupted" || task.Phase != "interrupted" || task.Result != "" {
		t.Fatalf("started=%v canceled=%v answer=%q task=%+v err=%v", artifactStarted, artifactCanceled, answer, task, err)
	}
	for _, step := range steps {
		if step.Type == StepFinalAnswer {
			t.Fatal("emitted final answer after cancellation")
		}
	}
	if err := filepath.WalkDir(a.cfg.SandboxArtifactHostDir, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() {
			t.Errorf("unexpected file: %s", path)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestCanceledRequestSkipsPreparation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	// No initialized dependencies: prepare would panic, so this also checks no I/O.
	a := &UnifiedAgent{}
	r := a.runOnce(ctx, "hello", ChatOptions{}, nil)
	if !r.Interrupted || r.Error == "" {
		t.Fatalf("%+v", r)
	}
}

func TestToolContextPreferenceAndLateCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	legacyCalled := false
	_, err := executeToolContext(ctx, tool.Tool{
		Execute: func(map[string]interface{}) (string, error) { legacyCalled = true; return "wrong", nil },
		ExecuteCtx: func(got context.Context, _ map[string]interface{}) (string, error) {
			if got != ctx {
				t.Error("context was replaced")
			}
			cancel()
			return "late", nil
		},
	}, nil)
	if legacyCalled || !errors.Is(err, context.Canceled) {
		t.Fatalf("legacy=%v err=%v", legacyCalled, err)
	}
}

func TestResearchSearchFailurePropagates(t *testing.T) {
	a := failureAgent()
	a.cfg.LLMMode = "mock"
	a.rag = &rag.Engine{}
	want := errors.New("search backend unavailable")
	a.tools = newToolRegistry(map[string]tool.Tool{"search_web": {
		ExecuteCtx: func(context.Context, map[string]interface{}) (string, error) { return "", want },
	}})
	out, err := (&researchAgent{a}).Run(context.Background(), SubAgentTask{Goal: "q"})
	if out != "" || !errors.Is(err, want) {
		t.Fatalf("%q %v", out, err)
	}
}

func TestGraphZeroRetriesStillExecutesOnce(t *testing.T) {
	a := failureAgent()
	calls := 0
	tg := graph.NewTaskGraph([]*graph.Node{{ID: "n1", Type: graph.NodeTypeTool, ToolName: "test"}})
	rt := NewGraphRuntime(tg, a, GraphConfig{MaxParallel: 1}, map[string]tool.Tool{
		"test": {ExecuteCtx: func(context.Context, map[string]interface{}) (string, error) { calls++; return "ok", nil }},
	}, &TaskState{}, nil)
	out, err := rt.executeSingleNode(context.Background(), "n1")
	if err != nil || out != "ok" || calls != 1 {
		t.Fatalf("%q %v calls=%d", out, err, calls)
	}
}

func TestRetryBackoffIsCancelable(t *testing.T) {
	a := failureAgent()
	a.cfg.MaxRetries = 3
	a.cfg.RetryDelayMs = 60000
	started := make(chan struct{})
	tg := graph.NewTaskGraph([]*graph.Node{{ID: "n1", Type: graph.NodeTypeTool, ToolName: "test"}})
	rt := NewGraphRuntime(tg, a, GraphConfig{MaxParallel: 1}, map[string]tool.Tool{
		"test": {ExecuteCtx: func(context.Context, map[string]interface{}) (string, error) {
			close(started)
			return "", errors.New("transient")
		}},
	}, &TaskState{}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := rt.executeSingleNode(ctx, "n1"); done <- err }()
	<-started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("backoff ignored cancellation")
	}
}

func TestCanceledArtifactDoesNotWrite(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	dir := t.TempDir()
	a := failureAgent()
	if a.produceArtifact(ctx, "生成报告", "body", dir, nil) != nil {
		t.Fatal("produced canceled artifact")
	}
	if ok, _ := a.materialize(ctx, dir, "report.md", "body"); ok {
		t.Fatal("materialized canceled artifact")
	}
	items, err := os.ReadDir(dir)
	if err != nil || len(items) != 0 {
		t.Fatalf("files=%v err=%v", items, err)
	}
}
