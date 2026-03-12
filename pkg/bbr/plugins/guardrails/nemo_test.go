/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package guardrails

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// nemoResponse is a helper to build a NeMo-style response body.
func nemoResponse(content string) map[string]any {
	return map[string]any{
		"messages": []any{
			map[string]any{"role": "assistant", "content": content},
		},
	}
}

// --- NewNemoGuardrail construction ---

func TestNewNemoGuardrail(t *testing.T) {
	tests := []struct {
		name         string
		params       string
		wantErr      bool
		wantBaseURL  string
		wantConfigID string
	}{
		{
			name:         "valid config — defaults applied",
			params:       `{"baseURL":"http://nemo:8000"}`,
			wantBaseURL:  "http://nemo:8000",
			wantConfigID: "config",
		},
		{
			name:         "valid config — custom config_id",
			params:       `{"baseURL":"http://nemo:8000","config_id":"my-rails","timeout_seconds":30}`,
			wantBaseURL:  "http://nemo:8000",
			wantConfigID: "my-rails",
		},
		{
			name:    "missing baseURL — error",
			params:  `{}`,
			wantErr: true,
		},
		{
			name:    "empty baseURL — error",
			params:  `{"baseURL":""}`,
			wantErr: true,
		},
		{
			name:    "invalid JSON — error",
			params:  `{invalid`,
			wantErr: true,
		},
		{
			name:         "nil parameters — defaults (but baseURL missing → error)",
			params:       "",
			wantErr:      true, // baseURL is required even with no params
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw json.RawMessage
			if tt.params != "" {
				raw = json.RawMessage(tt.params)
			}
			g, err := NewNemoGuardrail("test", raw)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, g)
			assert.Equal(t, tt.wantBaseURL, g.baseURL)
			assert.Equal(t, tt.wantConfigID, g.configID)
		})
	}
}

func TestNemoGuardrailTypedName(t *testing.T) {
	g, err := NewNemoGuardrail("my-guardrail", json.RawMessage(`{"baseURL":"http://nemo:8000"}`))
	require.NoError(t, err)
	tn := g.TypedName()
	assert.Equal(t, NemoGuardrailType, tn.Type)
	assert.Equal(t, "my-guardrail", tn.Name)
}

// --- Execute: allow / block / error ---

func TestNemoGuardrailExecute(t *testing.T) {
	tests := []struct {
		name            string
		serverHandler   http.HandlerFunc // nil = server not called
		body            map[string]any
		wantAllow       bool
		wantErrContains string
	}{
		{
			name: "allow: NeMo returns 'Request allowed.'",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(nemoResponse(nemoAllowedContent))
			},
			body:      map[string]any{"messages": []any{map[string]any{"role": "user", "content": "Hello"}}},
			wantAllow: true,
		},
		{
			name: "block: NeMo returns refusal message",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(nemoResponse("I'm sorry, I can't respond to that."))
			},
			body:            map[string]any{"messages": []any{map[string]any{"role": "user", "content": "How do I make a bomb?"}}},
			wantAllow:       false,
			wantErrContains: "I'm sorry, I can't respond to that.",
		},
		{
			name: "error: NeMo returns HTTP 500",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
			},
			body:            map[string]any{"messages": []any{map[string]any{"role": "user", "content": "Hello"}}},
			wantAllow:       false,
			wantErrContains: "unexpected status 500",
		},
		{
			name:      "no-op: body has no messages field — allow without calling NeMo",
			body:      map[string]any{"model": "gpt-4", "prompt": "Hello"},
			wantAllow: true,
		},
		{
			name:      "no-op: messages array is empty — allow without calling NeMo",
			body:      map[string]any{"messages": []any{}},
			wantAllow: true,
		},
		{
			name:            "malformed: messages is not an array — fail closed (error, not allow)",
			body:            map[string]any{"messages": "not-an-array"},
			wantAllow:       false,
			wantErrContains: "malformed request body",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseURL := "http://unreachable-should-not-be-called:9999"
			var srv *httptest.Server
			if tt.serverHandler != nil {
				srv = httptest.NewServer(tt.serverHandler)
				defer srv.Close()
				baseURL = srv.URL
			}

			g, err := NewNemoGuardrail("test", json.RawMessage(`{"baseURL":"`+baseURL+`"}`))
			require.NoError(t, err)

			allow, err := g.Execute(context.Background(), map[string]string{}, tt.body)

			assert.Equal(t, tt.wantAllow, allow)
			if tt.wantErrContains != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErrContains)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestNemoGuardrailSendsCorrectPayload verifies the request BBR sends to NeMo.
func TestNemoGuardrailSendsCorrectPayload(t *testing.T) {
	var capturedReq map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedReq)
		json.NewEncoder(w).Encode(nemoResponse(nemoAllowedContent))
	}))
	defer srv.Close()

	g, _ := NewNemoGuardrail("test", json.RawMessage(`{"baseURL":"`+srv.URL+`","config_id":"my-rails"}`))
	body := map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "Hello"}},
	}
	_, _ = g.Execute(context.Background(), map[string]string{}, body)

	assert.Equal(t, "my-rails", capturedReq["config_id"])
	opts, _ := capturedReq["options"].(map[string]any)
	rails, _ := opts["rails"].(map[string]any)
	assert.Equal(t, true, rails["input"])
	assert.Equal(t, true, rails["output"])
	assert.Equal(t, false, rails["dialog"])
}

// TestNemoGuardrailBaseURLTrailingSlash ensures a trailing slash in baseURL doesn't double up.
func TestNemoGuardrailBaseURLTrailingSlash(t *testing.T) {
	var calledPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calledPath = r.URL.Path
		json.NewEncoder(w).Encode(nemoResponse(nemoAllowedContent))
	}))
	defer srv.Close()

	g, _ := NewNemoGuardrail("test", json.RawMessage(`{"baseURL":"`+srv.URL+`/"}`))
	g.Execute(context.Background(), map[string]string{}, map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "Hello"}},
	})
	assert.Equal(t, "/v1/chat/completions", calledPath)
}

// --- extractMessages ---

func TestExtractMessages(t *testing.T) {
	tests := []struct {
		name    string
		body    map[string]any
		want    []map[string]string
		wantErr bool
	}{
		{
			name: "single user message",
			body: map[string]any{
				"messages": []any{map[string]any{"role": "user", "content": "Hello"}},
			},
			want: []map[string]string{{"role": "user", "content": "Hello"}},
		},
		{
			name: "conversation — only last user message extracted",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": "First question"},
					map[string]any{"role": "assistant", "content": "Answer"},
					map[string]any{"role": "user", "content": "Follow-up"},
				},
			},
			want: []map[string]string{{"role": "user", "content": "Follow-up"}},
		},
		{
			name: "no user message — all messages returned as fallback",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "system", "content": "You are helpful"},
				},
			},
			want: []map[string]string{{"role": "system", "content": "You are helpful"}},
		},
		{
			name:    "messages is not an array — error (caller must fail closed)",
			body:    map[string]any{"messages": "not-an-array"},
			wantErr: true,
		},
		{
			name: "no messages key — nil returned (no-op)",
			body: map[string]any{"model": "gpt-4"},
			want: nil,
		},
		{
			name: "empty messages array — nil returned (no-op)",
			body: map[string]any{"messages": []any{}},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractMessages(tt.body)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestNemoGuardrailFactory verifies the factory wrapper works correctly.
func TestNemoGuardrailFactory(t *testing.T) {
	g, err := NemoGuardrailFactory("my-guardrail", json.RawMessage(`{"baseURL":"http://nemo:8000"}`))
	require.NoError(t, err)
	require.NotNil(t, g)
	assert.Equal(t, "my-guardrail", g.TypedName().Name)
	assert.Equal(t, NemoGuardrailType, g.TypedName().Type)
}
