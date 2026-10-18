# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] — 2026-10-18

### Added

- `providerMetadata` on `ToolCall`, assistant messages, `ChatResult.message` and stream deltas carries opaque per-provider turn state across a tool loop.

### Changed

- **Breaking:** Anthropic `reasoning` sends adaptive thinking with `output_config.effort` instead of a `budget_tokens` budget; models before Claude 4.6 now return the vendor's 400 (use `_passthrough` for budget mode).
- Anthropic's default `max_tokens` is 16000 instead of 4096, so always-thinking models no longer stop at `length` with no text.
- `usage` counts billed tokens: Anthropic/Bedrock `inputTokens` include cache reads and writes, Gemini `outputTokens` include thoughts.
- **Breaking:** Bedrock `toolChoice: 'none'` omits `toolConfig`, so the model is offered no tools; after a tool loop Bedrock may answer 400.
- In-body errors — mid-stream error frames and 2xx bodies carrying an `error` (OpenRouter, embeddings) — keep the HTTP status they arrived with on `statusCode` and take their code from the embedded error (`unknown` when it names none).
- Gemini stream usage arrives on the finish delta instead of a separate trailing delta.
- **Breaking:** OpenAI and Azure OpenAI send `maxOutputTokens` as `max_completion_tokens`, so reasoning models accept it; Azure `api-version`s before `2024-09-01-preview` reject it (the v1 API takes it).
- Gemini sends tool schemas as `parametersJsonSchema` and `json_schema` response formats as `responseJsonSchema`.

### Fixed

- Gemini 3 tool loops keep their thought signatures (or send Google's validator-skip value), so the follow-up turn no longer 400s.
- Anthropic tool loops replay the turn's thinking blocks verbatim when the message (text, tool ids and arguments) still matches them; once a turn is rebuilt, later turns are rebuilt too.
- Gemini's own function-call ids are echoed back, and synthetic ids are unique per call.
- Ordinary 400s are no longer misread as `context_length_exceeded` (Anthropic, Gemini, Bedrock).
- Gemini `API_KEY_INVALID` maps to `auth_failed`.
- 402 (credit exhaustion) maps to `rate_limited`, 413 to `invalid_request`, 408 and Bedrock 424 to `provider_unavailable`, and mid-stream error events honour their type or code.
- Anthropic `model_context_window_exceeded`, Bedrock `model_context_window_exceeded` and Gemini `CONTINUATION` map to `length`; Gemini image-safety reasons to `content_filter`.
- A stream that ends before its terminal event, or a 2xx stream that is not SSE, throws instead of ending silently.
- A single CRLF no longer splits an SSE event, so CRLF multi-line `data:` events are kept whole.
- Bedrock merges differently-cased duplicate headers before signing, so they no longer break SigV4.
- Bedrock rejects a `region` that is not a region name, so the signed request cannot be steered to another host.
- Mistral embeddings send `dimensions` as `output_dimension`.
- `Retry-After` is read only in its RFC 9110 forms (whole seconds or an HTTP-date, past dates as 0); blank, negative or decimal values are ignored.
- Gemini tool results take their function name from the nearest preceding call, so repeated synthetic ids no longer mislabel earlier results.
- Bedrock signs the full path of a `baseUrl` with a path prefix and drops empty text parts.
- Embeddings classify a 2xx error's string code (e.g. `insufficient_quota`) the way chat does.
- Anthropic drops empty text parts, which the Messages API rejects.
- A 2xx carrying an `error` (OpenRouter's mid-generation failures) or missing its payload throws instead of returning an empty success.
- Embeddings throw on a vector count that doesn't match the inputs or a non-numeric cell instead of misaligning or coercing.
- Mistral reasoning replies (chunk-list `content`) yield their answer text instead of the raw chunk array.

## [1.0.1] — 2026-08-08

### Fixed

- A non-2xx is classified even when the host's `fetch` rejects instead of resolving, and a
  transport rejection never carries the key-bearing request URL into the error.

## [1.0.0] — 2026-07-24

### Added

- Initial release of `@thinwrap/llm`: `Chat` (`complete` + `stream`) and `Embeddings` facades
  over 18 providers behind one normalized surface, with a typed `ConnectorError`.
- Fifteen OpenAI-compatible providers share one connector driven by a spec registry; Anthropic,
  Bedrock and Gemini have bespoke wire adapters emitting identical shapes.
- Zero runtime dependencies: bring-your-own `fetch`; Bedrock SigV4 hand-rolled on `node:crypto`.
