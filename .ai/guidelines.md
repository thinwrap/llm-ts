# `@thinwrap/llm` — contributor guide

This folder (`.ai/`) is for developers — and the coding agents working alongside them — who are
**changing this library**: adding a provider connector or improving the package. It is not usage
documentation.

> **Using the package in your app?** See [`../README.md`](../README.md) and the per-connector
> READMEs under [`../src/providers/`](../src/providers). `.ai/` is not part of the npm tarball —
> its only audience is people working in the repo.

## Map of this folder

- **guidelines.md** (this file) — entry point + the two "add a connector" recipes.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — the facade → dispatch → connector model, the two connector tiers, and the invariants every change must hold.
- [`CONVENTIONS.md`](./CONVENTIONS.md) — file layout, naming, TypeScript/build config, the per-connector README convention.

## The shape in one sentence

A consumer constructs the `Chat` facade by provider id (`new Chat('openai', cfg)`); the facade
dispatches to a connector that satisfies `IChatConnector`, which turns the normalized `ChatInput`
into the vendor's wire request and the vendor response back into the normalized `ChatResult` /
`ChatStreamDelta` / `ConnectorError`. No global middleware — vendor specifics stay local to the
connector.

## Two connector tiers

Providers fall into exactly two buckets (see the PRD's bimodal-compat finding):

1. **First-class OpenAI-compatible (15 providers).** They speak OpenAI Chat Completions natively.
   They share **one** connector — `src/providers/_shared/openai-compat.connector.ts` — parameterized
   by a row in the `SPECS` registry (`src/providers/_shared/spec.ts`). Adding one is a config row,
   not a new class.
2. **Native adapters (Anthropic, Bedrock, Gemini; Vertex/Cohere pending).** Their wire shape is
   structurally different (Messages / Converse / generateContent). Each gets its own
   `src/providers/<id>/` directory with a real connector class. It must emit the **identical**
   normalized surface so it stays drop-in swappable.

## Setup & verify

```bash
npm install
npm run typecheck && npm test
```

Node ≥18 (native `fetch`). **Zero runtime dependencies — do not add any** (AWS SigV4 for Bedrock is
hand-rolled on `node:crypto`).

## Add a first-class OpenAI-compatible provider

Copy an existing `SPECS` row. Touch-points:

1. **Register the id** — add it to `LLM_PROVIDER_IDS` in [`src/types/provider-id.enum.ts`](../src/types/provider-id.enum.ts).
2. **Add the spec row** — in [`src/providers/_shared/spec.ts`](../src/providers/_shared/spec.ts): `id`, `defaultBaseUrl`, `buildAuthHeaders`, and any quirk flags (`supportsTools`, `maxTokensField`, `streamUsagePath`, `strictParams`). That's the whole connector.
3. **README** — create `src/providers/<id>/README.md` (plain Markdown that opens with its `# Title`, no metadata block — see CONVENTIONS).
4. **Test** — add a case to `src/providers/_shared/openai-compat.connector.spec.ts` (base URL + auth + any quirk).

No new facade case is needed — the facade already dispatches every `SPECS` id to the shared connector.

## Add a native-adapter provider

Copy [`src/providers/anthropic/`](../src/providers/anthropic) (JSON/SSE) or
[`src/providers/bedrock/`](../src/providers/bedrock) (signed request + binary-stream caveat).
Touch-points, in order:

1. **Register the id** — `LLM_PROVIDER_IDS` in `src/types/provider-id.enum.ts`.
2. **Config map** — add `'<id>': <Name>Config` to the intersection in [`src/types/config-map.type.ts`](../src/types/config-map.type.ts) (add the id to the `Exclude<...>` list so it doesn't also get `OpenAICompatConfig`).
3. **Create `src/providers/<id>/`**:
   - `<id>.config.ts` — the `<Name>Config` interface (auth fields first, optional `fetch?`).
   - `<id>.connector.ts` — a class `extends BaseConnector implements IChatConnector` with `complete()`, `stream()`, `readonly id`, and a private `mapVendorError()`. All wire translation is local.
   - `<id>.connector.spec.ts` — vitest; inject a `vi.fn()` fetch mock.
   - `index.ts` — barrel re-export of the connector + config.
   - `README.md` — plain Markdown that opens directly with its `# Title` (no metadata block).
4. **Dispatch** — add a `if (providerId === '<id>')` branch to [`src/facades/chat.facade.ts`](../src/facades/chat.facade.ts).
5. **Export** — re-export the connector + config from [`src/index.ts`](../src/index.ts), the only public surface.

### Definition of done (the CI gates)

```bash
npm run typecheck                     # strict; provider-id ↔ config-map ↔ facade must line up
npm test                              # vitest — single file: npx vitest src/providers/<id>/<id>.connector.spec.ts
npm run build && npm run check:dist   # dual CJS/ESM emit + offline dual-entrypoint import smoke
```

## Invariants you must not break

Full reasoning lives in [`ARCHITECTURE.md`](./ARCHITECTURE.md); the short list:

- **Zero runtime deps / no vendor SDKs.** SigV4 and any signing are hand-rolled on `node:crypto`.
- **Stateless wrapper.** No caching, retries, idempotency keys, conversation state, or telemetry inside the wrapper. Prompt caching is a consumer concern via `_passthrough`.
- **Canonical shape = OpenAI Chat Completions (D1).** One facade; no second Responses-shaped facade. Connectors may target the Responses wire under the hood.
- **≥90% baseline-coverage rule.** A field belongs on `ChatInput`/`ChatResult` only if ≥90% of providers support it normalizably; everything else is `_passthrough` (in) / `raw` (out) — never emulated. The one deliberate sub-90% carve-out is `reasoning.effort` (D2).
- **Per-connector locality.** Wire translation + `mapVendorError` live inside the connector — never in `BaseConnector`.
- **Every connector emits the identical `ChatInput`/`ChatResult`/`ChatStreamDelta`/`ConnectorError`.** That parity is the whole product.
- **Seven `ProviderCode` values**, surfaced via `ConnectorError`; retry hints ride in `cause.retryAfter` / `cause.retryAfterSeconds` — never a top-level field.
- **Pinned `vite ^6` / `vitest ^3` — do not bump** (Node 18 floor).
