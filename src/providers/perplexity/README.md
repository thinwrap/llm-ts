# Perplexity (Sonar) — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `perplexity` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('perplexity', { apiKey: process.env.PERPLEXITY_API_KEY! });
const res = await chat.complete({
  model: 'sonar',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://api.perplexity.ai` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create a key at <https://www.perplexity.ai/settings/api>. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.perplexity.ai/chat/completions`

### Notes / quirks

- **Baseline-exception (handled):** the Sonar chat surface has **no tool/function calling** (`tools` are not sent); tools live on Perplexity's separate Agent API.
- Sampling surface is stripped (no `top_k`/penalties/`seed`/`n` on Sonar).
- Model ids are `sonar` / `sonar-pro` / `sonar-reasoning-pro` / `sonar-deep-research`.

## Embeddings

Perplexity exposes no OpenAI-float embeddings surface, so there is no `Embeddings` support — `new Embeddings('perplexity', …)` is a type error and throws `ConnectorError` (`invalid_request`).

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://docs.perplexity.ai/guides/usage-tiers>.
