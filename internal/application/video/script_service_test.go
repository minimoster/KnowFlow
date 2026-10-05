package video

import (
	"context"
	"strings"
	"testing"

	"agi-assistant/internal/infrastructure/llm"
)

type fakeLLM struct {
	reply string
}

func (f fakeLLM) ChatContext(ctx context.Context, systemPrompt string, messages []llm.Message) string {
	return f.reply
}

func TestGenerateScriptSuccessWithCodeBlock(t *testing.T) {
	service := NewService(fakeLLM{reply: "```json\n{\"hook\":\"反常识开头\",\"body\":[\"第一段\",\"第二段\"],\"cta\":\"关注我\",\"duration_estimate\":60}\n```"}, "test-model")

	resp, err := service.GenerateScript(context.Background(), ScriptRequest{
		RequestID: "vf-task-1",
		UserID:    "u1",
		Topic:     "RAG",
		Duration:  60,
		Style:     "口播",
	})

	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp.Status != "succeeded" {
		t.Fatalf("expected succeeded, got %s", resp.Status)
	}
	if resp.Result.Hook != "反常识开头" {
		t.Fatalf("unexpected hook: %s", resp.Result.Hook)
	}
	if resp.Usage.Model != "test-model" {
		t.Fatalf("unexpected model: %s", resp.Usage.Model)
	}
}

func TestGenerateScriptInvalidJSON(t *testing.T) {
	service := NewService(fakeLLM{reply: "不是 JSON"}, "test-model")

	resp, err := service.GenerateScript(context.Background(), ScriptRequest{
		RequestID: "vf-task-1",
		Topic:     "RAG",
		Duration:  60,
		Style:     "口播",
	})

	if err == nil {
		t.Fatal("expected error")
	}
	if resp.Status != "failed" || resp.Error == nil || resp.Error.Code != "SCRIPT_JSON_INVALID" {
		t.Fatalf("unexpected error response: %+v", resp)
	}
}

func TestGenerateScriptInvalidRequest(t *testing.T) {
	service := NewService(fakeLLM{reply: "{}"}, "test-model")

	resp, err := service.GenerateScript(context.Background(), ScriptRequest{
		RequestID: "",
		Topic:     "RAG",
		Duration:  60,
		Style:     "口播",
	})

	if err == nil {
		t.Fatal("expected error")
	}
	if resp.Error == nil || resp.Error.Code != "SCRIPT_REQUEST_INVALID" {
		t.Fatalf("unexpected error response: %+v", resp)
	}
	if !strings.Contains(resp.Error.Message, "request_id") {
		t.Fatalf("expected request_id error, got %s", resp.Error.Message)
	}
}
