# OpenRouter — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `openrouter` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('openrouter', { apiKey: process.env.OPENROUTER_API_KEY! });
const res = await chat.complete({
  model: 'openai/gpt-4o',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://openrouter.ai/api/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create a key at <https://openrouter.ai/keys>. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://openrouter.ai/api/v1/chat/completions`

### Notes / quirks

- Aggregator: model ids are `{provider}/{model}` (e.g. `anthropic/claude-sonnet-4.5`), with optional `:free`/`:nitro`/`:floor` suffixes.
- Pass `HTTP-Referer` / `X-Title` via `headers` for app attribution.
- Drops params an underlying provider lacks (unless `provider.require_parameters` is set via `_passthrough`); `n` is unsupported.

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('openrouter', { apiKey: process.env.OPENROUTER_API_KEY! });
const out = await emb.create({
  model: 'openai/text-embedding-3-small',
  input: ['hello', 'world'],
});
```

`POST https://openrouter.ai/api/v1/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://openrouter.ai/docs/api-reference/limits>.
