# OpenAI — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `openai` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('openai', { apiKey: process.env.OPENAI_API_KEY! });
const res = await chat.complete({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://api.openai.com/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create an API key at <https://platform.openai.com/api-keys>. It is sent verbatim as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.openai.com/v1/chat/completions`

### Notes / quirks

- The reference provider — its Chat Completions shape is the canonical facade shape.
- Reasoning models (o-series / GPT-5-class) reject `temperature`/`top_p`/penalties with a hard **HTTP 400** and require `max_completion_tokens` — guard by model family; don't blanket-pass sampling knobs.
- OpenAI also ships the newer **Responses API**; this connector targets Chat Completions (D1).

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('openai', { apiKey: process.env.OPENAI_API_KEY! });
const out = await emb.create({
  model: 'text-embedding-3-small',
  input: ['hello', 'world'],
});
```

`POST https://api.openai.com/v1/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` (Matryoshka) when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://platform.openai.com/docs/guides/rate-limits>.
