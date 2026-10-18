# Anthropic (Claude) — `@thinwrap/llm`

Native adapter over the Anthropic **Messages API** (`POST /v1/messages`). Anthropic does not speak OpenAI's Chat Completions shape, so this connector is a full translation layer — it produces the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `anthropic` is still just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('anthropic', { apiKey: process.env.ANTHROPIC_API_KEY! });
const res = await chat.complete({
  model: 'claude-opus-5-5',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `x-api-key` (not `Authorization: Bearer`). |
| `baseUrl` | no | `https://api.anthropic.com/v1` | Override for proxies / gateways. |
| `anthropicVersion` | no | `2023-06-01` | `anthropic-version` header. |
| `defaultMaxTokens` | no | `16000` | Anthropic **requires** `max_tokens`; used when `maxOutputTokens` is omitted. |
| `fetch` | no | `globalThis.fetch` | BYO HTTP client. |

## Auth setup

Create an API key in the Anthropic Console. It is sent as the `x-api-key` header along with `anthropic-version`. No environment inference — pass `apiKey` explicitly (or read it from your own env as above).

## Chat

### Endpoint

`POST <baseUrl>/messages` (default `https://api.anthropic.com/v1/messages`).

### Notes / quirks

How the normalized shape maps:

- **System message** → hoisted to the top-level `system` field (multiple system messages are concatenated).
- **`role: 'tool'` messages** → carried inside a `user` turn as `tool_result` blocks; consecutive tool messages coalesce into one turn (native Anthropic requirement).
- **Assistant `toolCalls`** → `tool_use` content blocks; **`tools`** → `{ name, description, input_schema }` (JSON Schema in `input_schema`, not nested under `function`).
- **`toolChoice`** → `auto`→`{type:'auto'}`, `required`→`{type:'any'}`, `none`→`{type:'none'}`, `{function:{name}}`→`{type:'tool', name}`.
- **Images** → `{ type:'image', source:{ type:'base64', media_type, data } }`.
- **`stop`** → `stop_sequences`.
- **`reasoning.effort`** → `thinking: { type: 'adaptive' }` + `output_config: { effort }`, for every model (no model-id switch, no `budget_tokens`, `max_tokens` unchanged). `temperature` / `top_p` are omitted while `reasoning` is set. Models before Claude 4.6 (verified on Haiku 4.5) reject adaptive thinking with a 400 that passes through as `invalid_request`; on those budget-mode models leave `reasoning` unset and send `_passthrough: { body: { thinking: { type: 'enabled', budget_tokens: N } } }` (with `maxOutputTokens` above `N`). `_passthrough.body` merges shallowly, so an `output_config` sent there (e.g. for `format`) replaces the connector's whole `output_config`, effort included — put `effort` in it too. An empty effort is sent as `medium`; any other value goes out as given for the API to validate.
- **Usage** → `{ inputTokens, outputTokens, totalTokens }`: `inputTokens` is `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens` (Anthropic's `input_tokens` counts only the uncached tail); the total is summed.
- **`stop_reason`** → `finishReason` (`end_turn`/`stop_sequence`→`stop`, `max_tokens`/`model_context_window_exceeded`→`length`, `tool_use`→`tool_calls`, `refusal`→`content_filter`, `pause_turn`→`unknown` — resend the turn to let the server-tool loop continue).
- **Thinking blocks across turns** → the response's content blocks, verbatim, ride on `message.providerMetadata.anthropic.content` (a stream rebuilds the same array from its events and puts it on the terminal delta). When an assistant message carries them and they still match it — same `tool_use` ids in order, each `tool_use.input` equal to that call's parsed `arguments`, same joined text — they are sent back as-is, thinking included. Otherwise the turn is rebuilt from `content` / `toolCalls` without its thinking blocks, and so is **every later assistant turn in the request**: Anthropic voids all thinking blocks after an edited or stripped turn, and accounts that enforce preserved thinking reject them with a 400. A rebuilt turn loses its earlier reasoning.

Vendor rules that pass through as `invalid_request` (never emulated): `temperature` / `top_p` return a 400 on Claude Opus 4.7 and later, on every 5.x model and on Fable (on Sonnet 5.5 and Haiku 5.5, only non-default values); a forced `toolChoice` (`required` or a named tool) returns a 400 on Opus 5.5, Sonnet 5.5 and Fable 5.1 — use `auto` and steer from the prompt.

`max_tokens` defaults to 16000 when neither `maxOutputTokens` nor `defaultMaxTokens` is set: it is a ceiling, not a charge, and thinking counts against it.

Not normalized (passthrough / raw — never emulated):

- **`responseFormat`** — Anthropic has no `response_format`; use tool-based structured output via `_passthrough`. This connector does **not** apply `responseFormat`.
- **Reasoning CoT output** — `thinking` blocks are not normalized text; they appear in `ChatResult.raw` and, opaquely, in `providerMetadata`.
- **Prompt caching** — pass `cache_control` markers via `_passthrough.body`; the read/write split stays in `raw` (both are folded into `usage.inputTokens`).

## Embeddings

Anthropic exposes no OpenAI-float embeddings surface (its embeddings partner is Voyage), so there is no `Embeddings` support — `new Embeddings('anthropic', …)` throws `ConnectorError` (`invalid_request`).

## Error mapping & passthrough

- `ConnectorError.providerCode`: `401/403`→`auth_failed`, `429` and `402 billing_error` (credit exhaustion)→`rate_limited`, `408` and any `5xx` (incl. `504 timeout_error`, `529 overloaded_error`)→`provider_unavailable`, other `4xx` (`400`, `404`, `413 request_too_large`, …)→`invalid_request`; a `400` whose message is a context overflow ("prompt is too long", "… exceed context limit") →`context_length_exceeded`. A mid-stream `error` event keeps the stream's HTTP status (200) on `statusCode` and takes its code from `error.type` the same way (`unknown` for an unrecognised type). The raw error body is on `cause.raw`; a non-empty `Retry-After` on `cause.retryAfter`, with `cause.retryAfterSeconds` when it parses to a non-negative delay.
- Provider-specific request fields go through `_passthrough.body` / `.headers` / `.query`; vendor response extras stay on `ChatResult.raw`.
- Rate-limit guidance: <https://docs.anthropic.com/en/api/rate-limits>.
