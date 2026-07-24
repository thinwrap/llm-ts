# `@thinwrap/llm` — Conventions

Naming, file layout, and test patterns for adding or refactoring a connector.

## Where files live

```
src/
  index.ts                              # public-API barrel (the only public surface)
  base/
    base.connector.ts                   # BaseConnector (fetch plumbing) + parseSSEStream
  facades/
    chat.facade.ts                      # Chat facade + dispatch + *.spec.ts
  providers/
    _shared/
      openai-compat.connector.ts        # the ONE connector for all 15 first-class providers
      spec.ts                           # SPECS registry — a row per first-class provider
      openai-compat.connector.spec.ts
    <id>/                               # one directory per NATIVE adapter (anthropic, bedrock, gemini)
      <id>.connector.ts                 # class extends BaseConnector implements IChatConnector
      <id>.connector.spec.ts            # vitest, co-located
      <id>.config.ts                    # <Name>Config interface
      index.ts                          # barrel re-export
      README.md                         # per-connector consumer doc (plain Markdown)
    <id>/README.md                      # first-class providers get a README-only dir (no class)
  types/                                # ChatInput/ChatResult/etc., ConnectorError, provider-id enum, config map
```

## Provider ids

A single string-literal union, declared in `src/types/provider-id.enum.ts`:

```typescript
export const LLM_PROVIDER_IDS = ['openai', 'azure-openai', /* … */, 'anthropic', 'bedrock', 'gemini'] as const;
export type LlmProviderId = (typeof LLM_PROVIDER_IDS)[number];
```

`ProviderConfigMap` (in `src/types/config-map.type.ts`) maps the 15 first-class ids to
`OpenAICompatConfig` and each native id to its own config via an intersection. Adding a native id
means `Exclude`-ing it from the compat mapping and adding it to the intersection.

## File naming (native adapters)

| File | Required? | Purpose |
|---|---|---|
| `<id>.connector.ts` | yes | Connector class `extends BaseConnector implements IChatConnector` |
| `<id>.connector.spec.ts` | yes | vitest spec, co-located |
| `<id>.config.ts` | yes | Exported `<Name>Config` interface |
| `index.ts` | yes | Barrel re-export |
| `README.md` | yes | Per-connector consumer doc (plain Markdown) |

First-class providers need only a `SPECS` row + a `README.md` (no class, no config file — they share `OpenAICompatConfig`).

## Test pattern (vitest)

Inject a `vi.fn()` fetch mock via the `fetch` config field — never a global module-level mock.
Assert both the normalized result AND the outbound wire request. For streaming, feed an SSE (or
provider-native) string as the `Response` body.

```typescript
const fetchMock = vi.fn((_url: string, _init?: RequestInit): Promise<Response> =>
  Promise.resolve(new Response(JSON.stringify({ /* vendor shape */ }), { status: 200 })));
const res = await new SomeConnector({ apiKey: 'k', fetch: fetchMock as unknown as typeof fetch }).complete(input);
```

## Per-connector README

Each provider's per-connector `README.md` is plain Markdown — it opens directly with its
`# Title` (no YAML metadata block). It is the connector's consumer-facing doc; keep it
complete and at parity with the sibling-language libraries.

## TypeScript / build

- `tsconfig.json` is `strict: true` with `noUncheckedIndexedAccess`. Target ES2021, lib ES2022.
- `npm run typecheck` (`tsc --noEmit`) clean is the canary for any connector change.
- Dual build → `dist/cjs/` (CommonJS) + `dist/esm/` (ES2020). `scripts/fix-esm-imports.mjs` adds
  `.js` specifiers to the ESM emit (source imports are extensionless); `npm run check:dist`
  import-smokes both entrypoints the way Node loads them. Source/declaration maps are excluded from
  the build (tarball size).
- **`vite` is pinned to `^6` and `vitest` to `^3` — do not bump** (Node 18 floor; Vite 7 / Vitest 4
  dropped Node 18). Revisit at the umbrella-wide v2 cutover (ESM-only + Node ≥22).
