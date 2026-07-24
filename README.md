# @thinwrap/llm

> ⚠️ **Work in progress (v0).** Milestone 1: the shared OpenAI-compatible chat connector across the first-class provider tier. Native adapters (Anthropic, Bedrock), embeddings, and per-connector docs are in progress. See `_bmad-output/planning-artifacts/prd-llm.md`.

Unified, **zero-dependency** TypeScript facade over LLM chat-completion providers. One typed `Chat` facade; switch vendor by changing the provider id + model. Stateless, bring-your-own `fetch`, no vendor SDKs — the in-process, zero-egress complement to an LLM gateway.

## Install

```bash
npm install @thinwrap/llm
```

Node ≥ 18 (native `fetch`).

## Quick start

```ts
import { Chat } from '@thinwrap/llm';

const chat = new Chat('openai', { apiKey: process.env.OPENAI_API_KEY! });

const res = await chat.complete({
  model: 'gpt-5.6',
  messages: [{ role: 'user', content: 'Say hi in one word.' }],
});
console.log(res.message.content, res.usage);
```

Switching vendor is the provider id + model:

```ts
const chat = new Chat('anthropic', { apiKey: process.env.ANTHROPIC_API_KEY! });
// same .complete(input) shape, same ChatResult
```

Streaming:

```ts
for await (const delta of new Chat('groq', { apiKey }).stream(input)) {
  if (delta.contentDelta) process.stdout.write(delta.contentDelta);
}
```

## Design

Facade → connector dispatch → shared `OpenAICompatConnector` (or a native adapter). The normalized facade holds only the ≥90%-supported core; provider-specific features ride through `_passthrough` (request) / `raw` (result). See the contributor docs in `.ai/`.

Per-provider docs (auth, config, quirks, rate-limit links, error mapping) live in
[`src/providers/`](src/providers) — one README per provider.

## Timeouts & cancellation

The wrapper is stateless and holds no timeout of its own — it awaits whatever your `fetch` does. To bound a hung provider, bring your own timeout via the injected `fetch`:

```ts
const chat = new Chat('openai', {
  apiKey: process.env.OPENAI_API_KEY,
  fetch: (url, init) => fetch(url, { ...init, signal: AbortSignal.timeout(30_000) }),
});
```

An aborted request surfaces as a `ConnectorError` (`providerCode: 'invalid_request'`), same as every other failure.

MIT © Dmitry Polyanovsky
