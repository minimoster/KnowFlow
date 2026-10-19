package llm

import (
	"context"
	"errors"
	"net/http"
	"testing"
)

func TestPreferenceExtractionFailureDoesNotBecomeRuleSuccess(t *testing.T) {
	c := testClient(func(*http.Request) (*http.Response, error) { return response(429, `{}`), nil })
	got, err := c.ExtractPreferencesContext(context.Background(), "我喜欢咖啡")
	if err == nil || len(got) != 0 {
		t.Fatalf("got=%v err=%v", got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err = c.ExtractPreferencesContext(ctx, "我喜欢咖啡")
	if !errors.Is(err, context.Canceled) || len(got) != 0 {
		t.Fatalf("got=%v err=%v", got, err)
	}
}
