# Mistral — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `mistral` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('mistral', { apiKey: process.env.MISTRAL_API_KEY! });
const res = await chat.complete({
  model: 'mistral-large-latest',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://api.mistral.ai/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create a key at <https://console.mistral.ai/api-keys>. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.mistral.ai/v1/chat/completions`

### Notes / quirks

- **Baseline-exception (handled):** Mistral 422s on unknown params, so the connector omits OpenAI-only extras like `stream_options`. Keep `_passthrough.body` to known Mistral fields.
- Model ids are date-stamped (e.g. `-YYMM`).

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('mistral', { apiKey: process.env.MISTRAL_API_KEY! });
const out = await emb.create({
  model: 'mistral-embed',
  input: ['hello', 'world'],
});
```

`POST https://api.mistral.ai/v1/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://docs.mistral.ai/admin/user-management-finops/tier>.
