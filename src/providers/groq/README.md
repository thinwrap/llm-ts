# Groq — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `groq` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('groq', { apiKey: process.env.GROQ_API_KEY! });
const res = await chat.complete({
  model: 'llama-3.3-70b-versatile',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://api.groq.com/openai/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create a key at <https://console.groq.com/keys>. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.groq.com/openai/v1/chat/completions`

### Notes / quirks

- **Baseline-exception (handled):** in streaming, token usage is on `x_groq.usage` of the final chunk, not top-level `usage`. The connector relocates it so `ChatStreamDelta.usage` is populated the same as everywhere else (spec `streamUsagePath: 'x_groq'`).
- Strict JSON-schema structured output is limited to `openai/gpt-oss-*` models and can't combine with streaming or tools; other models are best-effort.
- Model ids are provider-namespaced (`openai/…`, `meta-llama/…`, `qwen/…`).

## Embeddings

Groq exposes no OpenAI-float embeddings surface, so there is no `Embeddings` support — `new Embeddings('groq', …)` is a type error and throws `ConnectorError` (`invalid_request`).

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://console.groq.com/docs/rate-limits>.
