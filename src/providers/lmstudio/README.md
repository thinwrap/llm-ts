# LM Studio (self-host) — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `lmstudio` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('lmstudio', {});
const res = await chat.complete({
  model: '<loaded-model>',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | no | — | Optional `Authorization: Bearer <apiKey>` (API-token auth since v0.4.0; off by default). |
| `baseUrl` | no | `http://localhost:1234/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Local single-user server; auth is off by default (leave `apiKey` empty → no auth header). If you enabled API-token auth, pass the token as `apiKey`; it is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST http://localhost:1234/v1/chat/completions`

### Notes / quirks

- Self-host: local single-user server; override `baseUrl` if not on `:1234`.
- With a single model loaded, the server ignores the requested model id and serves whatever is loaded.
- Structured output is grammar-enforced (llama.cpp / MLX); small models can be unreliable.

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('lmstudio', {});
const out = await emb.create({
  model: '<embedding-model>',
  input: ['hello', 'world'],
});
```

`POST http://localhost:1234/v1/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://lmstudio.ai/docs/app/api>.
