# DeepInfra — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `deepinfra` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('deepinfra', { apiKey: process.env.DEEPINFRA_API_KEY! });
const res = await chat.complete({
  model: 'meta-llama/Meta-Llama-3.1-70B-Instruct',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no | `https://api.deepinfra.com/v1/openai` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Create a token at <https://deepinfra.com/dash/api_keys>. It is sent as `Authorization: Bearer <apiKey>`.

## Chat

### Endpoint

`POST https://api.deepinfra.com/v1/openai/chat/completions`

### Notes / quirks

- Model ids are HuggingFace-style `org/model` slugs (case-sensitive).
- Runs on a vLLM backend (superset of OpenAI); `usage` adds a non-standard `estimated_cost`; vision `detail` is ignored.

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('deepinfra', { apiKey: process.env.DEEPINFRA_API_KEY! });
const out = await emb.create({
  model: 'BAAI/bge-large-en-v1.5',
  input: ['hello', 'world'],
});
```

`POST https://api.deepinfra.com/v1/openai/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://docs.deepinfra.com/account/rate-limits>.
