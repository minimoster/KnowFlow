// mode_tool.go — 单工具调用模式（"tool"）。
//
//   - runTool: tool.Decide 选工具 → 执行 → 注入记忆前缀做 LLM 总结。
//     onEvent 非 nil 时通过 SSE 推送 tool_call / token 事件；为 nil 时静默执行。
package chat

import (
	"context"
	"fmt"

	promptctx "agi-assistant/internal/domain/promptctx"
	"agi-assistant/internal/domain/tool"
	"agi-assistant/internal/infrastructure/llm"
	"agi-assistant/internal/usercontext"
)

// runTool 执行单工具模式。onEvent 为 nil 时即"非流式"路径。
func (a *UnifiedAgent) runTool(
	ctx context.Context,
	query string,
	ts map[string]tool.Tool,
	memPrefix string,
	histMsgs []llm.Message,
	onEvent func(StreamEvent),
) (string, *tool.CallResult, error) {
	tc := tool.Decide(query, ts)
	if tc == nil {
		return "我无法处理这个请求。", nil, nil
	}
	t, ok := ts[tc.ToolName]
	if !ok {
		return "", tc, fmt.Errorf("工具 %s 不存在", tc.ToolName)
	}

	// 偏好感知参数自动填充——按当前用户的偏好桶
	a.fillParamsFromPreference(usercontext.UserIDFromContext(ctx), tc)

	result, err := executeToolContext(ctx, t, tc.Params)
	if err != nil {
		if ctx.Err() != nil {
			return "", tc, ctx.Err()
		}
		a.pctx.recordToolCall(promptctx.ToolCallTrace{
			ToolName: tc.ToolName, Success: false, Summary: err.Error(),
		})
		return "", tc, err
	}
	tc.ToolResult = result
	a.pctx.recordToolCall(promptctx.ToolCallTrace{
		ToolName: tc.ToolName, Success: true, Summary: result,
	})

	// 流式模式下立即推送一条 tool_call 事件（含工具名、参数、结果），让前端可以
	// 提前渲染调用记录而不必等 LLM 总结完成。
	emit(onEvent, "tool_call", map[string]interface{}{
		"tool_name":   tc.ToolName,
		"params":      tc.Params,
		"tool_result": result,
	})

	systemPrompt := a.buildSystemPrompt(memPrefix, "你是一个善于综合信息的AI助手。结合你掌握的用户信息，使回答更个性化。")
	userMsg := fmt.Sprintf("用户问：%s\n工具 %s 返回结果：%s\n请根据结果自然地回答用户。", query, tc.ToolName, result)
	answer, err := a.chatLLM(ctx, systemPrompt, []llm.Message{{Role: "user", Content: userMsg}}, onEvent)
	return answer, tc, err
}

// Prefer context-aware implementations and never treat a late canceled result as success.
func executeToolContext(ctx context.Context, t tool.Tool, params map[string]interface{}) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	var out string
	var err error
	switch {
	case t.ExecuteStructured != nil:
		result := t.ExecuteStructured(ctx, params)
		if !result.Success {
			if result.Error != nil {
				err = result.Error
			} else {
				err = fmt.Errorf("tool %s failed without error details", t.Name)
			}
		} else {
			out = result.Payload
		}
	case t.ExecuteCtx != nil:
		out, err = t.ExecuteCtx(ctx, params)
	case t.Execute != nil:
		out, err = t.Execute(params)
	default:
		err = fmt.Errorf("tool %s has no executor", t.Name)
	}
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	return out, err
}
