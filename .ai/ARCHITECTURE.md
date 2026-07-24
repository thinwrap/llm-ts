# `@thinwrap/llm` — Architecture

One-page summary of the facade → dispatch → connector pattern as it manifests in this package.
Full scope rationale + the binding decisions (D1–D7) live in
`_bmad-output/planning-artifacts/prd-llm.md` and the scope-research decision log.

## Why facade + dispatch + connector

Consumer constructs the `Chat` facade by provider id; the facade dispatches to a connector that
satisfies `IChatConnector`; the connector translates the normalized request to the vendor wire and
back. No global middleware.

```
Consumer code
    │  new Chat('openai', cfg)      new Chat('anthropic', cfg)
    ▼                                   ▼
Chat facade ── lookup ──► OpenAICompatConnector(spec)   AnthropicConnector   (native adapter)
    │  .complete(input)          │  (15 first-class          │  extends BaseConnector
    │  .stream(input)            │   share one connector      │  full wire translation
    ▼                            ▼   + a SPECS row)           ▼
IChatConnector           BaseConnector.sendPostJson / invokeFetch / parseSSEStream
                                   │
                                   ▼  fetch (BYO or globalThis.fetch)
                              Vendor API
```

## Two connector tiers (compat is bimodal, not a spectrum)

- **First-class OpenAI-compatible (15):** OpenAI, Azure, OpenRouter, Groq, Together, Fireworks,
  DeepSeek, xAI, Mistral, Perplexity, DeepInfra, Cloudflare Workers AI, vLLM, Ollama, LM Studio.
  One `OpenAICompatConnector` + a `SPECS` row each (base URL, auth, quirks).
- **Native adapters (structurally different wire):** Anthropic (Messages), AWS Bedrock (Converse),
  Google Gemini (generateContent). Vertex + Cohere pending. Each is its own connector class that
  emits the identical normalized surface.

## The normalized surface (`IChatConnector`)

- **`complete(input): Promise<ChatResult>`** and **`stream(input): AsyncIterable<ChatStreamDelta>`**.
- `ChatResult` = `{ message: { role, content, toolCalls? }, finishReason, usage: {inputTokens, outputTokens, totalTokens}, model, raw }`.
- `ChatStreamDelta` = incremental `{ contentDelta?, toolCallDelta?, finishReason?, usage?, raw? }`, one shape regardless of wire transport (SSE / typed events / AWS event-stream / NDJSON).
- On vendor non-2xx (and network/abort), connectors throw `ConnectorError` (`statusCode`, `providerCode` ∈ 6-value union, `providerMessage`, `cause: { raw, retryAfter?, retryAfterSeconds? }`).

## `id` introspection

Each connector exposes `connector.id` (the provider-id literal). The facade exposes `chat.id`.

## Baseline coverage discipline (≥90% rule)

`ChatInput`/`ChatResult` carry only what ≥90% of providers support normalizably: text chat, a
normalized delta stream, function tool-calling, `response_format` (json_object / json_schema —
**acceptance**, not a strict guarantee), a canonical base64 image part, four sampling knobs
(`temperature` / `topP` / `maxOutputTokens` / `stop`), a normalized usage block, and a normalized
error. Everything else — `topK`, `seed`, penalties, `logitBias`, `n`, prompt-cache control, reasoning
CoT output, cost, rate-limit hints, audio/video — rides through `_passthrough` (input) / `raw`
(output) and is **never emulated**.

The one deliberate sub-90% carve-out is **`reasoning: { effort }`** (D2): normalized on input and
mapped per-connector (OpenAI-style `reasoning_effort`; a thinking-token budget for
Anthropic/Gemini). Reasoning *output* stays raw.

## Baseline-exception register (single-provider wire fixes, kept local)

- **Groq** — streaming usage is read from `x_groq.usage` on the final chunk.
- **Mistral** — 422s on unknown params → the shared connector skips OpenAI-only extras (`stream_options`).
- **Perplexity** — no tools on its primary Sonar surface (`supportsTools: false`).

## Native-adapter divergences (documented, translated locally)

- **Anthropic** — top-level `system`, tool results coalesced into a user turn, `tool_use`↔`toolCalls`, `reasoning.effort`→`thinking.budget_tokens` (sampling dropped while thinking). No `response_format` (structured output via `_passthrough`).
- **Bedrock** — Converse (`system[]`, `inferenceConfig`, `toolConfig`), hand-rolled SigV4. **v1 streaming is a non-incremental fallback** — `converse-stream` is AWS binary event-stream framing, not SSE; a real binary parser is a planned follow-up.
- **Gemini** — `systemInstruction` + `contents[]/parts`, tool results keyed by function **name** (resolved from the prior tool-call id), `response_format`→`responseSchema` (Gemini honors it), real SSE streaming.

## Stateless wrapper

The wrapper holds no state: no caching, retries, idempotency, conversation state, or telemetry.
Prompt caching, when a vendor supports it, is expressed by the consumer via `_passthrough`
(cache markers) with cached-token counts surfaced in `raw`. Refreshable auth (Azure Entra, Bedrock
SigV4, future Vertex OAuth) is computed per request.

## Canonical shape = Chat Completions, not Responses (D1)

One stateless, Chat-Completions-shaped facade. No second Responses-shaped facade. A connector is
free to translate onto the Responses wire under the hood for a provider that requires it;
Responses-only capabilities (server state, built-in tools, semantic-event streaming) are
`_passthrough`/raw.

## Cross-reference

- Naming, file layout, test patterns, README conventions: [`./CONVENTIONS.md`](./CONVENTIONS.md)
- Adding a connector / contributor entry point: [`./guidelines.md`](./guidelines.md)
- Consumer usage: [`../README.md`](../README.md)
