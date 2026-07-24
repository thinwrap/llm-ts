# Cloudflare Workers AI — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `cloudflare` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('cloudflare', { apiKey: process.env.CLOUDFLARE_API_TOKEN!, baseUrl: '<resource URL — see quirks>' });
const res = await chat.complete({
  model: '@cf/meta/llama-3.1-8b-instruct',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | yes | — | Required — no public default (embeds your account id; see Chat notes). |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create an API token (Workers AI scope) in the Cloudflare dashboard. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1/chat/completions`

### Notes / quirks

- **`baseUrl` is required**: `https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1`.
- Model ids are `@cf/{provider}/{model}` (some `@hf/…`), required verbatim.
- JSON mode is **not strict** (no schema-adherence guarantee) and can't stream.

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('cloudflare', { apiKey: process.env.CLOUDFLARE_API_TOKEN!, baseUrl: '<resource URL — see quirks>' });
const out = await emb.create({
  model: '@cf/baai/bge-large-en-v1.5',
  input: ['hello', 'world'],
});
```

`POST https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://developers.cloudflare.com/workers-ai/platform/limits/>.
