# AWS Bedrock — `@thinwrap/llm`

Native adapter over Amazon Bedrock's unified **Converse API** (`POST /model/{modelId}/converse`). Requests are signed with **hand-rolled AWS SigV4** on `node:crypto` — no `@aws-sdk/*`, no third-party crypto (same zero-dep approach as the notifications SES/SNS connectors). Emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('bedrock', {
  region: 'us-east-1',
  accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  // sessionToken: process.env.AWS_SESSION_TOKEN,  // for temporary creds
});
const res = await chat.complete({
  model: 'amazon.nova-pro-v1:0',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `region` | yes | — | e.g. `us-east-1`. No environment inference. Must be a region name (lowercase letters, digits, hyphens); anything else throws `ConnectorError` (`invalid_request`) at construction, since it becomes part of the signed host. |
| `accessKeyId` / `secretAccessKey` | yes | — | SigV4 credentials. |
| `sessionToken` | no | — | STS temporary-credential token → `X-Amz-Security-Token`. |
| `baseUrl` | no | `https://bedrock-runtime.<region>.amazonaws.com` | Override origin (VPC endpoint / proxy). A path prefix (a gateway's `/bedrock`) is kept and signed with the rest of the path. |
| `defaultMaxTokens` | no | `4096` | Converse `inferenceConfig.maxTokens` when `maxOutputTokens` is omitted. |
| `fetch` | no | `globalThis.fetch` | BYO HTTP client. |

## Auth setup

Uses standard AWS SigV4 (service `bedrock`) hand-rolled on `node:crypto` — no AWS SDK. Supply `accessKeyId` / `secretAccessKey` (and `sessionToken` for temporary credentials). The body is serialized once and those exact bytes are signed together with the canonical path so URL, body, and signature all agree.

## Chat

### Endpoint

`POST <origin>/model/{modelId}/converse` where `origin` defaults to `https://bedrock-runtime.<region>.amazonaws.com`. The model id is percent-encoded for the path (e.g. `...-v2:0` → `...-v2%3A0`) so the SigV4 canonical URI matches the wire URL exactly.

### Notes / quirks

How the normalized shape maps:

- **System message** → top-level `system: [{ text }]`.
- **`role: 'tool'` messages** → `toolResult` blocks inside a `user` turn; consecutive tool messages coalesce into one turn.
- **Assistant `toolCalls`** → `toolUse` blocks; **`tools`** → `toolConfig.tools[].toolSpec` with the JSON Schema under `inputSchema.json`.
- **`toolChoice`** → `auto`→`{auto:{}}`, `required`→`{any:{}}`, `{function:{name}}`→`{tool:{name}}`. `none` has no Converse counterpart, so `toolConfig` (the tools included) is omitted. Converse requires `toolConfig` whenever the history holds `toolUse` / `toolResult` blocks, so `none` after a tool loop returns the vendor's 400 (`invalid_request`); use `auto` there.
- **Images** → `{ image: { format, source: { bytes } } }` (`format` derived from `mediaType`).
- **Empty text parts** are dropped: Converse rejects an empty text block with a 400.
- **Sampling** → `inferenceConfig` `{ maxTokens, temperature, topP, stopSequences }`.
- **Usage** → `inputTokens` = `inputTokens` + `cacheReadInputTokens` + `cacheWriteInputTokens` (cache reads and writes are billed input), `outputTokens` as reported, `totalTokens` = their sum.
- **`stopReason`** → `finishReason` (`end_turn`/`stop_sequence`→`stop`, `max_tokens`/`model_context_window_exceeded`→`length`, `tool_use`→`tool_calls`, `content_filtered`/`guardrail_intervened`→`content_filter`, `malformed_model_output`/`malformed_tool_use`→`unknown`).

Limitations (documented, not silent):

- **Streaming is non-incremental.** Bedrock's `converse-stream` uses AWS's binary event-stream framing (`application/vnd.amazon.eventstream`), not SSE. This connector does not parse that binary protocol — `stream()` issues one Converse call and yields the full result as deltas.
- **Reasoning is not auto-mapped.** Reasoning control on Bedrock is model-dependent; pass `additionalModelRequestFields` via `_passthrough.body`.
- **`responseFormat`** is not mapped (use tool-based structured output via `_passthrough`); **prompt caching** via `_passthrough` (`cachePoint` blocks).
- **SigV4 signature is not pinned to an external cross-verified vector** (structural tests only).

## Embeddings

Bedrock has no OpenAI-float embeddings surface (Amazon Titan embeddings are not a Converse/OpenAI-float shape), so there is no `Embeddings` support — `new Embeddings('bedrock', …)` throws `ConnectorError` (`invalid_request`).

## Error mapping & passthrough

- `ConnectorError.providerCode`: `401/403`→`auth_failed`, `429` and `402`→`rate_limited`, `5xx`, `408 ModelTimeoutException` and `424 ModelErrorException`→`provider_unavailable`, other `4xx`→`invalid_request` (or `rate_limited` for throttling; a `400` ValidationException worded as an overflow — Bedrock relays the model provider's text: "Input is too long for requested model", "prompt is too long", "too many input tokens", "maximum context length", "exceed context limit" — →`context_length_exceeded`). Raw body on `cause.raw`; a non-empty `Retry-After` on `cause.retryAfter`, with `cause.retryAfterSeconds` when it parses to a non-negative delay.
- Headers from `config.headers` and `_passthrough.headers` are signed; names are merged case-insensitively (later source wins), so a differently-cased duplicate never reaches the wire.
- Provider-specific request fields go through `_passthrough.body` / `.headers` / `.query`; vendor response extras stay on `ChatResult.raw`.
- Rate-limit / quota guidance: <https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html>.
