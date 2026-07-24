# Azure OpenAI — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `azure-openai` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('azure-openai', { apiKey: process.env.AZURE_OPENAI_API_KEY!, baseUrl: '<resource URL — see quirks>' });
const res = await chat.complete({
  model: '<your-deployment-name>',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | yes | — | Sent as the `api-key: <apiKey>` header (Microsoft Entra OAuth is also accepted on the v1 GA API). |
| `baseUrl` | yes | — | Required — no public default (see Chat notes). |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Get the key and resource endpoint from the Azure portal. The connector sends it as the `api-key` header; Microsoft Entra OAuth bearer tokens are also accepted on the v1 GA API — supply that token via `headers` instead.

## Chat

### Endpoint

`POST https://<resource>.openai.azure.com/openai/v1/chat/completions`

### Notes / quirks

- **`baseUrl` is required**: `https://<resource>.openai.azure.com/openai/v1` (v1 GA API, opt-in Aug 2025 — no `api-version` needed).
- The `model` field is the per-resource **deployment name**, not a portable global model id.
- Auth is dual-mode: `api-key` header or an Entra bearer token (supply the latter via `headers`).

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('azure-openai', { apiKey: process.env.AZURE_OPENAI_API_KEY!, baseUrl: '<resource URL — see quirks>' });
const out = await emb.create({
  model: '<embedding-model>',
  input: ['hello', 'world'],
});
```

`POST https://<resource>.openai.azure.com/openai/v1/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://learn.microsoft.com/azure/ai-services/openai/quotas-limits>.
