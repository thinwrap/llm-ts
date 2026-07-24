# Google Gemini (AI Studio) — `@thinwrap/llm`

Native adapter for the Gemini `generateContent` API (`generativelanguage.googleapis.com/v1beta`). Gemini uses `contents[]`/`parts[]` (roles `user`/`model`), a top-level `systemInstruction`, and `functionDeclarations`, so this is a full translation layer emitting the identical normalized `ChatResult`/`ChatStreamDelta`/`ConnectorError`. **Streaming is real SSE** (`:streamGenerateContent?alt=sse`) — genuinely incremental, unlike the Bedrock v1 fallback.

> This is the **AI Studio** Gemini API (`gemini` provider id). Vertex AI Gemini is a separate provider (`vertex`, a future connector) with different auth (GCP OAuth/ADC) and endpoints.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('gemini', { apiKey: process.env.GEMINI_API_KEY! });
const res = await chat.complete({
  model: 'gemini-2.5-flash',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `x-goog-api-key`. |
| `baseUrl` | no | `https://generativelanguage.googleapis.com/v1beta` | Override for proxies. |
| `defaultMaxTokens` | no | — | `generationConfig.maxOutputTokens` when `maxOutputTokens` is omitted. |
| `fetch` | no | `globalThis.fetch` | BYO HTTP client. |

## Auth setup

Create an API key in Google AI Studio. It is sent as the `x-goog-api-key` header (the key is not placed in the query string). No environment inference — pass `apiKey` explicitly.

## Chat

### Endpoint

`POST <baseUrl>/models/{model}:generateContent` (non-streaming) and `POST <baseUrl>/models/{model}:streamGenerateContent?alt=sse` (streaming). The model is path-escaped. Default base URL `https://generativelanguage.googleapis.com/v1beta`.

### Notes / quirks

How the normalized shape maps:

- **System message** → top-level `systemInstruction`.
- **Roles** → `assistant`→`model`, `user`→`user`.
- **`role: 'tool'` messages** → `functionResponse` parts inside a `user` turn, keyed by function **NAME** (Gemini has no tool-call ids). The name is resolved from the matching prior assistant tool call (`toolCallId`→name), or `message.name`, falling back to `toolCallId`. Consecutive tool messages coalesce into one turn.
- **Assistant `toolCalls`** → `functionCall` parts; **`tools`** → `[{ functionDeclarations: [...] }]`.
- **`toolChoice`** → `functionCallingConfig.mode`: `auto`→`AUTO`, `required`→`ANY`, `none`→`NONE`, `{function:{name}}`→`ANY` + `allowedFunctionNames`.
- **Images** → `{ inlineData: { mimeType, data } }`.
- **Sampling** → `generationConfig` `{ maxOutputTokens, temperature, topP, stopSequences }`.
- **`responseFormat`** → **mapped** (unlike Anthropic): `json_object`→`responseMimeType: application/json`; `json_schema`→ that plus `responseSchema` (Gemini honors a JSON-Schema subset).
- **`reasoning.effort`** → `generationConfig.thinkingConfig.thinkingBudget` (low/medium/high ≈ 1024/8192/24576).
- **Usage** → `usageMetadata` `{ promptTokenCount, candidatesTokenCount, totalTokenCount }` → `{ inputTokens, outputTokens, totalTokens }`.
- **`finishReason`** → `STOP`→`stop`, `MAX_TOKENS`→`length`, `SAFETY`/`RECITATION`/etc.→`content_filter`; a `functionCall` part maps to `tool_calls`.

Not normalized (passthrough / raw):

- **Gemini-3 `thought_signature`** round-trip and reasoning CoT output → `raw` / `_passthrough` (the wrapper does not persist or re-send signatures automatically).
- **Prompt caching** (`cachedContent`), safety settings, and video params → `_passthrough.body`.

## Embeddings

Gemini's native embeddings use a different (non-OpenAI-float) shape, so there is no `Embeddings` support here — `new Embeddings('gemini', …)` throws `ConnectorError` (`invalid_request`). A dedicated embeddings surface is a future connector.

## Error mapping & passthrough

- `ConnectorError.providerCode`: `401/403`→`auth_failed`, `429`→`rate_limited`, `5xx`→`provider_unavailable`, `400/404/422`→`invalid_request` (or `context_length_exceeded`). Gemini emits **no rate-limit headers** — retry timing is parsed from the 429 body's `RetryInfo.retryDelay` into `cause.retryAfterSeconds`. Raw body on `cause.raw`.
- Provider-specific request fields go through `_passthrough.body` / `.headers` / `.query`; vendor response extras stay on `ChatResult.raw`.
- Rate-limit guidance: <https://ai.google.dev/gemini-api/docs/rate-limits>.
