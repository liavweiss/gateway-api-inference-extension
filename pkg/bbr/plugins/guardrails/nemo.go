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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	// NemoGuardrailType is the plugin type for the NeMo Guardrails HTTP callout.
	NemoGuardrailType = "nemo-guardrails"
	// nemoAllowedContent is the response content when NeMo input rails allow the request.
	nemoAllowedContent = "Request allowed."
	defaultTimeoutSec  = 10
)

var _ framework.Guardrail = (*NemoGuardrail)(nil)

// NemoGuardrail calls a NeMo Guardrails service over HTTP to check request content (input rails).
type NemoGuardrail struct {
	typedName  plugin.TypedName
	baseURL    string
	configID   string
	httpClient *http.Client
}

// NemoGuardrailConfig is the JSON configuration for the plugin.
// baseURL is required (e.g. http://nemo-guardrails.namespace.svc:8000).
type NemoGuardrailConfig struct {
	BaseURL        string `json:"baseURL"`
	ConfigID       string `json:"config_id"`
	TimeoutSeconds int    `json:"timeout_seconds"`
}

// NewNemoGuardrail builds a NeMo guardrail from name and optional JSON parameters.
func NewNemoGuardrail(name string, parameters json.RawMessage) (*NemoGuardrail, error) {
	cfg := NemoGuardrailConfig{
		ConfigID:       "config",
		TimeoutSeconds: defaultTimeoutSec,
	}
	if len(parameters) > 0 {
		if err := json.Unmarshal(parameters, &cfg); err != nil {
			return nil, fmt.Errorf("nemo-guardrails config: %w", err)
		}
	}
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("nemo-guardrails config: baseURL is required")
	}
	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = defaultTimeoutSec * time.Second
	}
	return &NemoGuardrail{
		typedName: plugin.TypedName{Type: NemoGuardrailType, Name: name},
		baseURL:  cfg.BaseURL,
		configID: cfg.ConfigID,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// TypedName returns the type and name of this plugin.
func (n *NemoGuardrail) TypedName() plugin.TypedName {
	return n.typedName
}

// Execute calls the NeMo Guardrails service. It extracts the last user message from body (OpenAI-style messages),
// POSTs to NeMo /v1/chat/completions for input-rail check, and returns (false, err with message) if NeMo blocks.
func (n *NemoGuardrail) Execute(ctx context.Context, _ map[string]string, body map[string]any) (bool, error) {
	messages, err := extractMessages(body)
	if err != nil {
		// Malformed messages field — fail closed: return an error so the caller can return 500.
		return false, fmt.Errorf("nemo-guardrails: malformed request body: %w", err)
	}
	if len(messages) == 0 {
		return true, nil // no messages to check (e.g. non-chat request) -> allow
	}

	reqBody := map[string]any{
		"config_id": n.configID,
		"messages":  messages,
		"options": map[string]any{
			"rails": map[string]any{
				"input":  true,
				"output": true,
				"dialog": false,
			},
		},
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return false, fmt.Errorf("nemo-guardrails: marshal request: %w", err)
	}

	baseURL := n.baseURL
	if len(baseURL) > 0 && baseURL[len(baseURL)-1] == '/' {
		baseURL = baseURL[:len(baseURL)-1]
	}
	url := baseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return false, fmt.Errorf("nemo-guardrails: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := n.httpClient.Do(req)
	if err != nil {
		return false, fmt.Errorf("nemo-guardrails: call failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("nemo-guardrails: unexpected status %d", resp.StatusCode)
	}

	var out struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return false, fmt.Errorf("nemo-guardrails: decode response: %w", err)
	}

	if len(out.Messages) == 0 {
		return false, fmt.Errorf("nemo-guardrails: empty response from NeMo (no messages in body)")
	}
	content := out.Messages[0].Content
	if content == nemoAllowedContent {
		return true, nil
	}
	// NeMo returned a block message (e.g. "I'm sorry, I can't respond to that...")
	return false, fmt.Errorf("%s", content)
}

// extractMessages pulls OpenAI-style "messages" from body and returns the last user message as a single-element slice
// for input-rail check, or all messages if present.
func extractMessages(body map[string]any) ([]map[string]string, error) {
	raw, ok := body["messages"]
	if !ok {
		return nil, nil
	}
	slice, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("messages is not an array")
	}
	var messages []map[string]string
	for _, m := range slice {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		content, _ := msg["content"].(string)
		messages = append(messages, map[string]string{"role": role, "content": content})
	}
	// For input rails we only need the user input: take the last user message, or all messages.
	// NeMo expects at least one user message. Use last user message as the one being checked.
	var lastUser []map[string]string
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i]["role"] == "user" {
			lastUser = messages[i : i+1]
			break
		}
	}
	if len(lastUser) > 0 {
		return lastUser, nil
	}
	if len(messages) > 0 {
		return messages, nil
	}
	return nil, nil
}

// NemoGuardrailFactory is the factory for the nemo-guardrails guardrail plugin.
func NemoGuardrailFactory(name string, parameters json.RawMessage) (framework.Guardrail, error) {
	return NewNemoGuardrail(name, parameters)
}
