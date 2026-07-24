# Ollama (self-host) — `@thinwrap/llm`

First-class OpenAI-compatible provider — served by the shared `OpenAICompatConnector` via a row in `src/providers/_shared/spec.ts` (no per-provider connector class). It emits the identical normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other connector, so switching to/from `ollama` is just the provider id + `model`.

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('ollama', {});
const res = await chat.complete({
  model: 'llama3.1:8b',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
```

## Configuration

| Field | Required | Default | Notes |
|---|---|---|---|
| `apiKey` | no | — | None by default (local server); if set, sent as `Authorization: Bearer <apiKey>`. |
| `baseUrl` | no* | `http://localhost:11434/v1` | *Required for a non-default host. |
| `fetch` | no | `globalThis.fetch` | Bring-your-own fetch. |
| `headers` | no | — | Extra headers merged onto every request. |

## Auth setup

No auth by default — leave `apiKey` empty and no auth header is sent. Override `baseUrl` for a remote host.

## Chat

### Endpoint

`POST http://localhost:11434/v1/chat/completions` (override `baseUrl` for a non-default host).

### Notes / quirks

- Self-host: no auth by default; override `baseUrl` for a non-default host.
- Model ids are host-local `name:tag` (e.g. `llama3.1:8b`) — a 404 means the model isn't pulled (`ollama pull …`).
- The `/v1` compat layer is documented-partial (drops `tool_choice`/`logit_bias`/`n`; base64-only images).

## Embeddings

Supported (served by the shared `OpenAICompatEmbeddingsConnector`).

```ts
import { Embeddings } from '@thinwrap/llm';

const emb = new Embeddings('ollama', {});
const out = await emb.create({
  model: 'nomic-embed-text',
  input: ['hello'],
});
```

`POST http://localhost:11434/v1/embeddings` with `encoding_format: 'float'`; `dimensions` maps to `dimensions` when set. Vectors are returned in input order on `out.embeddings`.

## Error mapping & passthrough

- Non-2xx responses throw `ConnectorError` with a 7-value `providerCode` (`auth_failed` / `rate_limited` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`). The raw vendor body is on `cause.raw`; retry hints (when present) on `cause.retryAfter` / `cause.retryAfterSeconds`.
- Sub-baseline features — `topK`, `seed`, `frequency`/`presence_penalty`, `logitBias`, `n`, reasoning control beyond `reasoning.effort`, prompt caching — ride through `_passthrough` (request) / `raw` (result). They are never emulated.
- Rate-limit guidance: <https://docs.ollama.com/api>.
