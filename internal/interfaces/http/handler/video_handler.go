package handler

import (
	"context"
	"encoding/json"
	"net/http"

	"agi-assistant/internal/application/video"
)

type videoGenerator interface {
	GenerateScript(ctx context.Context, req video.ScriptRequest) (video.ScriptResponse, error)
}

func (s *Server) videoScriptGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req video.ScriptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, video.ScriptResponse{
			RequestID: req.RequestID,
			Status:    "failed",
			Error: &video.ErrorBody{
				Code:    "INVALID_JSON_BODY",
				Message: "请求体不是合法 JSON",
			},
		})
		return
	}

	resp, err := s.video.GenerateScript(r.Context(), req)
	if err != nil {
		w.WriteHeader(http.StatusBadGateway)
		writeJSON(w, resp)
		return
	}

	writeJSON(w, resp)
}
