package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"agi-assistant/config"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func testClient(transport roundTripFunc) *Client {
	cfg := &config.APIConfig{LLMConfig: config.LLMConfig{
		LLMAPIUrl: "http://offline.invalid/chat", LLMAPIKey: "offline-test", LLMModel: "test-model",
	}}
	return NewWithHTTPClient(cfg, &http.Client{Transport: transport})
}
func response(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

func TestExplicitMockAndMissingConfiguration(t *testing.T) {
	for _, mode := range []string{"", "real", "typo"} {
		c := New(&config.APIConfig{LLMConfig: config.LLMConfig{LLMMode: mode}})
		out, err := c.Chat("", []Message{{Role: "user", Content: "hello"}})
		if err == nil || out != "" {
			t.Fatalf("mode=%q: %q, %v", mode, out, err)
		}
	}
	c := New(&config.APIConfig{LLMConfig: config.LLMConfig{LLMMode: "mock"}})
	out, err := c.Chat("", []Message{{Role: "user", Content: "hello"}})
	if err != nil || !strings.Contains(out, "模拟") {
		t.Fatalf("%q, %v", out, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out, err = c.ChatContext(ctx, "", nil)
	if out != "" || !errors.Is(err, context.Canceled) {
		t.Fatalf("%q, %v", out, err)
	}
}

func TestHTTPFailureNeverFallsBackToMock(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, status := range []int{429, 503} {
			t.Run(fmt.Sprintf("stream=%t/status=%d", stream, status), func(t *testing.T) {
				var calls atomic.Int32
				c := testClient(func(*http.Request) (*http.Response, error) {
					calls.Add(1)
					return response(status, `{"choices":[{"message":{"content":"fake success"}}]}`), nil
				})
				var out string
				var err error
				tokens := ""
				if stream {
					out, err = c.ChatStreamContext(context.Background(), "", nil, func(s string) { tokens += s })
				} else {
					out, err = c.ChatContextFast(context.Background(), "", nil)
				}
				if err == nil || !strings.Contains(err.Error(), fmt.Sprint(status)) || out != "" || tokens != "" || calls.Load() != 1 {
					t.Fatalf("out=%q tokens=%q err=%v calls=%d", out, tokens, err, calls.Load())
				}
			})
		}
	}
}

func TestTimeoutAndInFlightCancellation(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, timeout := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%t/timeout=%t", stream, timeout), func(t *testing.T) {
				started := make(chan struct{})
				c := testClient(func(r *http.Request) (*http.Response, error) {
					close(started)
					<-r.Context().Done()
					return nil, r.Context().Err()
				})
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				if timeout {
					c.httpClient.Timeout = 50 * time.Millisecond
				}
				finished := make(chan error, 1)
				go func() {
					var out string
					var err error
					if stream {
						out, err = c.ChatStreamContext(ctx, "", nil, nil)
					} else {
						out, err = c.ChatContext(ctx, "", nil)
					}
					if out != "" {
						finished <- fmt.Errorf("unexpected answer: %s", out)
						return
					}
					finished <- err
				}()
				select {
				case <-started:
				case <-time.After(time.Second):
					t.Fatal("transport not reached")
				}
				if !timeout {
					cancel()
				}
				select {
				case err := <-finished:
					want := context.Canceled
					if timeout {
						want = context.DeadlineExceeded
					}
					if !errors.Is(err, want) {
						t.Fatalf("want %v, got %v", want, err)
					}
				case <-time.After(time.Second):
					t.Fatal("request did not stop")
				}
			})
		}
	}
}

func TestBadResponsesFail(t *testing.T) {
	for _, body := range []string{`not json`, `{"choices":[]}`, `{"choices":[{"message":{"content":""}}]}`} {
		c := testClient(func(*http.Request) (*http.Response, error) { return response(200, body), nil })
		out, err := c.Chat("", nil)
		if err == nil || out != "" {
			t.Fatalf("%q: out=%q err=%v", body, out, err)
		}
	}
}

func TestTruncatedStreamReturnsErrorWithoutReplay(t *testing.T) {
	var calls atomic.Int32
	c := testClient(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return response(200, "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"), nil
	})
	tokens := ""
	out, err := c.ChatStreamContext(context.Background(), "", nil, func(s string) { tokens += s })
	if out != "" || !errors.Is(err, io.ErrUnexpectedEOF) || tokens != "partial" || calls.Load() != 1 {
		t.Fatalf("out=%q tokens=%q err=%v calls=%d", out, tokens, err, calls.Load())
	}
}

type brokenBody struct{}

func (brokenBody) Read([]byte) (int, error) { return 0, io.ErrClosedPipe }
func (brokenBody) Close() error             { return nil }
func TestStreamReadAndMalformedFrameFail(t *testing.T) {
	for _, body := range []io.ReadCloser{brokenBody{}, io.NopCloser(strings.NewReader("data: invalid\n\n"))} {
		c := testClient(func(*http.Request) (*http.Response, error) {
			resp := response(200, "")
			resp.Body = body
			return resp, nil
		})
		out, err := c.ChatStreamContext(context.Background(), "", nil, nil)
		if err == nil || out != "" {
			t.Fatalf("%q, %v", out, err)
		}
	}
}

func TestSuccessfulSyncAndStream(t *testing.T) {
	for _, stream := range []bool{false, true} {
		c := testClient(func(*http.Request) (*http.Response, error) {
			if stream {
				return response(200, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"), nil
			}
			return response(200, `{"choices":[{"message":{"content":"ok"}}]}`), nil
		})
		var out string
		var err error
		if stream {
			out, err = c.ChatStreamContext(context.Background(), "", nil, nil)
		} else {
			out, err = c.Chat("", nil)
		}
		if err != nil || out != "ok" {
			t.Fatalf("%q, %v", out, err)
		}
	}
}

func TestLateCanceledSuccessIsRejected(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := testClient(func(*http.Request) (*http.Response, error) {
		cancel()
		return response(200, `{"choices":[{"message":{"content":"late success"}}]}`), nil
	})
	out, err := c.ChatContext(ctx, "", nil)
	if out != "" || !errors.Is(err, context.Canceled) {
		t.Fatalf("%q %v", out, err)
	}
}
