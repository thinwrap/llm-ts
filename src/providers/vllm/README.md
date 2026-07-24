# vLLM (self-host) — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `vllm` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('vllm', { apiKey: process.env.VLLM_API_KEY! });
const res = await chat.complete({
  model: '<served-model-name>',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | no | — | Optional `Authorization: Bearer <apiKey>` (open if the server has no `--api-key`). |
| `baseUrl` | no | `http://localhost:8000/v1` | Override for proxies / gateways. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

Open by default. If the server was launched with `--api-key`, pass that value as `apiKey`; it is sent as `Authorization: Bearer <apiKey>`. When `apiKey` is empty, no auth header is sent.

## Chat

### Endpoint

`POST http://localhost:8000/v1/chat/completions`

### Notes / quirks

- Self-host: override `baseUrl` for a remote server; the compat server is vLLM's first-class interface.
- Model ids are operator-defined (the HF repo id or `--served-model-name`).
- `tool_choice: 'auto'` requires the server launched with `--enable-auto-tool-choice` + a model-matched `--tool-call-parser`.

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('vllm', { apiKey: process.env.VLLM_API_KEY! });
const out = await emb.create({
  model: '<embedding-model>',
  input: ['hello', 'world'],
});
```

`POST http://localhost:8000/v1/embeddings` with `encoding_format: 'float'`. `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/>.
