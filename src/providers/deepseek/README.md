# DeepSeek — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `deepseek` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('deepseek', { apiKey: process.env.DEEPSEEK_API_KEY! });
const res = await chat.complete({
  model: 'deepseek-chat',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://api.deepseek.com/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create a key at <https://platform.deepseek.com/api_keys>. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.deepseek.com/v1/chat/completions`

### Notes / quirks

- The OpenAI-compatible endpoint is first-class (the native API).
- **No strict `json_schema`** — `response_format` supports only `text`/`json_object`; a `json_schema` request degrades to best-effort JSON.
- `frequency_penalty` / `presence_penalty` are silently ignored; reasoning is exposed via `reasoning_content` (in `raw`).

## Embeddings

DeepSeek exposes no OpenAI-float embeddings surface, so there is no `Embeddings` support — `new Embeddings('deepseek', …)` is a type error and throws `ConnectorError` (`invalid_request`).

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://api-docs.deepseek.com/quick_start/rate_limit>.
