# xAI (Grok) — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `xai` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('xai', { apiKey: process.env.XAI_API_KEY! });
const res = await chat.complete({
  model: 'grok-4',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://api.x.ai/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create a key at <https://console.x.ai>. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.x.ai/v1/chat/completions`

### Notes / quirks

- Model ids are `grok-*`.
- xAI positions Chat Completions as a **legacy** endpoint (it steers new work to its own Responses API); this connector targets Chat Completions.
- In streaming, a tool/function call arrives **whole in a single chunk**, not as incremental argument deltas.

## Embeddings

xAI exposes no OpenAI-float embeddings surface, so there is no `Embeddings` support — `new Embeddings('xai', …)` is a type error and throws `ConnectorError` (`invalid_request`).

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://docs.x.ai/docs/rate-limits>.
