package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"agi-assistant/internal/application/video"
)

type fakeVideoGenerator struct {
	resp video.ScriptResponse
	err  error
}

func (f fakeVideoGenerator) GenerateScript(ctx context.Context, req video.ScriptRequest) (video.ScriptResponse, error) {
	return f.resp, f.err
}

func TestVideoScriptGenerateReturnsSaberEnvelope(t *testing.T) {
	s := &Server{video: fakeVideoGenerator{resp: video.ScriptResponse{
		RequestID: "vf-task-1",
		Status:    "succeeded",
		Result: video.ScriptResult{
			Hook:             "反常识开头",
			Body:             []string{"第一段", "第二段"},
			CTA:              "关注我",
			DurationEstimate: 60,
		},
	}}}
	req := httptest.NewRequest(http.MethodPost, "/api/video/script/generate", strings.NewReader(`{
		"request_id":"vf-task-1",
		"user_id":"u1",
		"topic":"RAG",
		"duration":60,
		"style":"口播"
	}`))
	rec := httptest.NewRecorder()

	s.videoScriptGenerate(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var got video.ScriptResponse
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Status != "succeeded" || got.Result.Hook != "反常识开头" {
		t.Fatalf("unexpected response: %+v", got)
	}
}

func TestVideoScriptGenerateRejectsInvalidMethod(t *testing.T) {
	s := &Server{}
	req := httptest.NewRequest(http.MethodGet, "/api/video/script/generate", nil)
	rec := httptest.NewRecorder()

	s.videoScriptGenerate(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rec.Code)
	}
}
