# Google Gemini (AI Studio) — `@thinwrap/llm`

Native adapter for the Gemini `generateContent` API (`generativelanguage.googleapis.com/v1beta`). Gemini uses `contents[]`/`parts[]` (roles `user`/`model`), a top-level `systemInstruction`, and `functionDeclarations`, so this is a full translation layer emitting the identical normalized `ChatResult`/`ChatStreamDelta`/`ConnectorError`. **Streaming is real SSE** (`:streamGenerateContent?alt=sse`) — genuinely incremental, unlike Bedrock's single-call stream.

> This is the **AI Studio** Gemini API (`gemini` provider id). Vertex AI Gemini is a separate provider (`vertex`, a future connector) with different auth (GCP OAuth/ADC) and endpoints.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('gemini', { apiKey: process.env.GEMINI_API_KEY! });
const res = await chat.complete({
  model: 'gemini-3.8-flash',
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
- **`role: 'tool'` messages** → `functionResponse` parts inside a `user` turn, keyed by function **NAME**. The name is `message.name`, else the name of the call with that `toolCallId` in the nearest preceding assistant turn (synthetic ids repeat across turns), falling back to `toolCallId`. Consecutive tool messages coalesce into one turn.
- **Tool-call ids** → Gemini's own `functionCall.id` becomes `ToolCall.id` and is echoed back on the replayed `functionCall` and on `functionResponse.id`. A call that came back without one gets `call_<index>` (unique per call, so parallel calls to one function stay distinct); only those made-up ids are kept off the wire — any other id, including one from another provider's history, is sent on both the call and its response.
- **Assistant `toolCalls`** → `functionCall` parts; **`tools`** → `[{ functionDeclarations: [...] }]` with the JSON Schema in `parametersJsonSchema` (the older `parameters` field takes only an OpenAPI subset and rejects e.g. `additionalProperties`).
- **Thought signatures** → a `functionCall` part's `thoughtSignature` is stored on `ToolCall.providerMetadata.gemini.thoughtSignature` (streamed on `toolCallDelta.providerMetadata`) and put back on that part when the turn is replayed; Gemini 3 rejects a tool-loop turn without it. A turn with no stored signature (hand-built, or from another provider) gets Google's documented `skip_thought_signature_validator` value on its first function call.
- **`toolChoice`** → `functionCallingConfig.mode`: `auto`→`AUTO`, `required`→`ANY`, `none`→`NONE`, `{function:{name}}`→`ANY` + `allowedFunctionNames`.
- **Images** → `{ inlineData: { mimeType, data } }`.
- **Sampling** → `generationConfig` `{ maxOutputTokens, temperature, topP, stopSequences }`.
- **`responseFormat`** → **mapped** (unlike Anthropic): `json_object`→`responseMimeType: application/json`; `json_schema`→ that plus `responseJsonSchema` (JSON Schema; the deprecated `responseSchema` takes only an OpenAPI subset).
- **`reasoning.effort`** → `generationConfig.thinkingConfig.thinkingBudget` (low/medium/high ≈ 1024/8192/24576).
- **Usage** → `inputTokens` = `promptTokenCount` + `toolUsePromptTokenCount`, `outputTokens` = `candidatesTokenCount` + `thoughtsTokenCount` (thoughts are billed as output), `totalTokens` = their sum. A stream reports the cumulative figures once, on the finish delta (the last one).
- **`finishReason`** (the google-genai SDK `FinishReason` enum) → `STOP`→`stop`, `MAX_TOKENS`/`CONTINUATION`→`length`, `SAFETY`/`RECITATION`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII`/`IMAGE_SAFETY`/`IMAGE_PROHIBITED_CONTENT`/`IMAGE_RECITATION`→`content_filter`, everything else (`LANGUAGE`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS`, `NO_IMAGE`, `IMAGE_OTHER`, `OTHER`, …)→`unknown`; a `functionCall` part maps to `tool_calls`.

Not normalized (passthrough / raw):

- **Reasoning CoT output** → `raw`.
- **Prompt caching** (`cachedContent`), safety settings, and video params → `_passthrough.body`.

## Embeddings

Gemini's native embeddings use a different (non-OpenAI-float) shape, so there is no `Embeddings` support here — `new Embeddings('gemini', …)` throws `ConnectorError` (`invalid_request`). A dedicated embeddings surface is a future connector.

## Error mapping & passthrough

- `ConnectorError.providerCode`: `401/403` and a `400` whose `ErrorInfo.reason` is `API_KEY_INVALID`→`auth_failed`, `429` and `402` (depleted prepaid credit)→`rate_limited`, `408`/`5xx`→`provider_unavailable`, other `4xx`→`invalid_request`; a `400` "input token count … exceeds the maximum number of tokens"→`context_length_exceeded`. A mid-stream `{"error": …}` frame keeps the stream's HTTP status (200) on `statusCode` and takes its code from the frame's `code`, or its `status` name when the code is missing (`unknown` with neither). Gemini emits **no rate-limit headers** — retry timing is parsed from the 429 body's `RetryInfo.retryDelay` into `cause.retryAfterSeconds`. Raw body on `cause.raw`.
- Provider-specific request fields go through `_passthrough.body` / `.headers` / `.query`; vendor response extras stay on `ChatResult.raw`.
- Rate-limit guidance: <https://ai.google.dev/gemini-api/docs/rate-limits>.
