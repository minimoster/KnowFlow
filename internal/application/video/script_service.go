package video

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"agi-assistant/internal/infrastructure/llm"
)

type ScriptRequest struct {
	RequestID string `json:"request_id"`
	UserID    string `json:"user_id"`
	Topic     string `json:"topic"`
	Duration  int    `json:"duration"`
	Style     string `json:"style"`
}

type ScriptResult struct {
	Hook             string   `json:"hook"`
	Body             []string `json:"body"`
	CTA              string   `json:"cta"`
	DurationEstimate int      `json:"duration_estimate"`
}

type Usage struct {
	Model     string `json:"model"`
	LatencyMS int64  `json:"latency_ms"`
}

type ScriptResponse struct {
	RequestID string       `json:"request_id"`
	Status    string       `json:"status"`
	Result    ScriptResult `json:"result,omitempty"`
	Usage     Usage        `json:"usage,omitempty"`
	Error     *ErrorBody   `json:"error,omitempty"`
}

type ErrorBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type ChatClient interface {
	ChatContext(ctx context.Context, systemPrompt string, messages []llm.Message) string
}

type Service struct {
	llm   ChatClient
	model string
}

func NewService(client ChatClient, model string) *Service {
	return &Service{llm: client, model: model}
}

const scriptSystemPrompt = `你是资深短视频脚本写手，擅长为抖音、小红书、视频号生成结构化口播脚本。
你必须只输出 JSON，不要输出 markdown 代码块，不要输出解释。`

func (s *Service) GenerateScript(ctx context.Context, req ScriptRequest) (ScriptResponse, error) {
	if err := validateScriptRequest(req); err != nil {
		return failed(req.RequestID, "SCRIPT_REQUEST_INVALID", err.Error()), err
	}

	start := time.Now()
	userPrompt := fmt.Sprintf(`请根据以下信息生成一条短视频口播脚本：

主题：%s
目标时长：%d 秒
风格：%s

输出严格 JSON，结构如下：
{
  "hook": "开头 5 秒钩子，制造冲突、反常识或强利益点",
  "body": ["主体段落 1", "主体段落 2", "主体段落 3"],
  "cta": "结尾引导关注、评论或行动",
  "duration_estimate": 60
}`, req.Topic, req.Duration, req.Style)

	raw := s.llm.ChatContext(ctx, scriptSystemPrompt, []llm.Message{{Role: "user", Content: userPrompt}})
	result, err := parseScriptResult(raw)
	if err != nil {
		return failed(req.RequestID, "SCRIPT_JSON_INVALID", err.Error()), err
	}

	return ScriptResponse{
		RequestID: req.RequestID,
		Status:    "succeeded",
		Result:    result,
		Usage: Usage{
			Model:     s.model,
			LatencyMS: time.Since(start).Milliseconds(),
		},
	}, nil
}

func validateScriptRequest(req ScriptRequest) error {
	if strings.TrimSpace(req.RequestID) == "" {
		return fmt.Errorf("request_id 不能为空")
	}
	if strings.TrimSpace(req.Topic) == "" {
		return fmt.Errorf("topic 不能为空")
	}
	if req.Duration <= 0 {
		return fmt.Errorf("duration 必须大于 0")
	}
	if strings.TrimSpace(req.Style) == "" {
		return fmt.Errorf("style 不能为空")
	}
	return nil
}

func parseScriptResult(raw string) (ScriptResult, error) {
	cleaned := stripCodeBlock(raw)
	var result ScriptResult
	if err := json.Unmarshal([]byte(cleaned), &result); err != nil {
		return ScriptResult{}, fmt.Errorf("解析脚本 JSON 失败: %w", err)
	}
	if strings.TrimSpace(result.Hook) == "" {
		return ScriptResult{}, fmt.Errorf("hook 不能为空")
	}
	if len(result.Body) == 0 {
		return ScriptResult{}, fmt.Errorf("body 不能为空")
	}
	if strings.TrimSpace(result.CTA) == "" {
		return ScriptResult{}, fmt.Errorf("cta 不能为空")
	}
	if result.DurationEstimate <= 0 {
		return ScriptResult{}, fmt.Errorf("duration_estimate 必须大于 0")
	}
	return result, nil
}

func stripCodeBlock(s string) string {
	out := strings.TrimSpace(s)
	if strings.HasPrefix(out, "```") {
		re := regexp.MustCompile("(?s)^```[a-zA-Z]*\\n?(.*?)\\n?```$")
		if m := re.FindStringSubmatch(out); len(m) == 2 {
			return strings.TrimSpace(m[1])
		}
	}
	return out
}

func failed(requestID, code, message string) ScriptResponse {
	return ScriptResponse{
		RequestID: requestID,
		Status:    "failed",
		Error:     &ErrorBody{Code: code, Message: message},
	}
}
