# @thinwrap/llm

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

## Bring your own `fetch`

Every connector takes an optional `fetch` in config — useful for tracing, mocking, proxying,
or bounding a request (see *Timeouts & cancellation* above).

### The injected `fetch` does not isolate you from the host

**Contract: a non-2xx must be RETURNED as a `Response`, not thrown** — that is plain fetch
semantics, and it is what lets each connector map the status to a `providerCode`
(429 → `rate_limited`, 401 → `auth_failed`, …).

On Node, both `globalThis.fetch` and `undici.fetch` dispatch through undici's
*process-global* dispatcher, so whatever the application installed applies to your calls
too — including a `responseError()` interceptor, under which a non-2xx **rejects** instead
of resolving. The library detects that and rebuilds the `Response`, so classification
still works. It cannot repair one thing: if the provider gzipped the error body, that
interceptor buffers it below fetch's content-decoding and the bytes are destroyed before
any library sees them, so the vendor's message is lost.

To make your calls genuinely independent of the host's configuration, pass an explicit
dispatcher instead of relying on the global one:

```ts
import { Agent, fetch as undiciFetch } from 'undici';

const isolated = new Agent();                      // no inherited interceptors
const fetchImpl = ((url, init) =>
  undiciFetch(url as string, { ...init, dispatcher: isolated })) as typeof fetch;
```

## Security

Report vulnerabilities **privately** — please do not open a public issue. Preferred: a
[private security advisory](https://github.com/thinwrap/llm-ts/security/advisories/new)
on this repository. Alternatively, email **security@thinwrap.dev**. Include the affected
versions and a minimal reproduction if you have one.

A vulnerability in a *provider's* own API or service belongs to that vendor rather than
to this wrapper — please report those upstream.

MIT © Dmitry Polyanovsky
