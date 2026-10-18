import { appendPassthroughQuery, applyRetryAfter, BaseConnector, parseSSEStream } from '../../base/base.connector';
import { ConnectorError } from '../../types';
import type {
  ChatInput,
  ChatMessage,
  ChatResult,
  ChatStreamDelta,
  FinishReason,
  IChatConnector,
  ProviderCode,
  ToolCall,
  ToolChoice,
  Usage,
} from '../../types';
import type { AnthropicConfig } from './anthropic.config';

const DEFAULT_BASE_URL = 'https://api.anthropic.com/v1';
const DEFAULT_VERSION = '2023-06-01';
// A ceiling, not a charge. Thinking counts against it, and current models
// always think (Opus 5.5 at medium effort by default), so a small default
// often ends at `length` with no text; 16000 is Anthropic's recommended
// non-streaming default.
const DEFAULT_MAX_TOKENS = 16000;

/**
 * Native adapter for the Anthropic **Messages API** (`POST /v1/messages`).
 * Anthropic is structurally different from OpenAI Chat Completions — different
 * auth, a top-level `system`, typed content blocks, tool results carried inside
 * a user turn, `stop_sequences`, and Anthropic-shaped SSE events — so this is a
 * full translation layer, not an OpenAI passthrough. It emits the identical
 * normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other
 * connector, so it is drop-in swappable via the `Chat` facade.
 *
 * Not normalized here (surfaced via `raw` or `_passthrough`, never emulated):
 * `responseFormat` (Anthropic has no `response_format` — use tool-based
 * structured output via `_passthrough`); reasoning CoT output (`thinking`
 * blocks land in `raw`); prompt caching (`cache_control` via `_passthrough`,
 * cache read/write counts in `raw`, folded into `usage.inputTokens`).
 *
 * The response's content blocks travel verbatim on
 * `message.providerMetadata.anthropic.content` so the next request can replay
 * the turn with its thinking blocks intact.
 */
export class AnthropicConnector extends BaseConnector implements IChatConnector {
  public readonly id = 'anthropic';
  private readonly baseUrl: string;
  private readonly config: AnthropicConfig;

  constructor(config: AnthropicConfig) {
    super(config.fetch);
    this.config = config;
    this.baseUrl = (config.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
  }

  async complete(input: ChatInput): Promise<ChatResult> {
    const body = this.buildRequest(input, false);
    const response = await this.sendPostJson(
      appendPassthroughQuery(`${this.baseUrl}/messages`, input._passthrough?.query),
      body,
      this.authHeaders(input),
      input.signal,
    );
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw this.mapVendorError(response.status, errBody, response.headers);
    }
    const json = this.requireDecodedBody<AnthropicMessageResponse>((await response.json().catch(() => null)) as AnthropicMessageResponse | null, response.status);
    // Every Messages response carries a `content` array; without one there is
    // no message to parse.
    if (!Array.isArray(json.content)) throw this.missingPayloadError(response.status, json, 'no content');
    return this.parseResult(json, input.model);
  }

  async *stream(input: ChatInput): AsyncGenerator<ChatStreamDelta> {
    const body = this.buildRequest(input, true);
    const response = await this.invokeFetch(appendPassthroughQuery(`${this.baseUrl}/messages`, input._passthrough?.query), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...this.authHeaders(input),
      },
      body: JSON.stringify(body),
      signal: input.signal,
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw this.mapVendorError(response.status, errBody, response.headers);
    }
    this.requireEventStream(response);

    let startUsage: AnthropicUsage | undefined;
    let sawStop = false;
    // The turn's content blocks, rebuilt from the events in block order, so a
    // streamed turn replays exactly like a non-streamed one.
    const blocks = new StreamedBlocks();
    // Anthropic indexes content blocks (text + tool_use interleaved), so the
    // first tool call can arrive at block index 1. Normalize to a 0-based
    // tool-call-relative index to match OpenAI-compat's emitted surface.
    let nextToolIndex = 0;
    const toolIndexByBlock = new Map<number, number>();
    for await (const raw of parseSSEStream(response)) {
      const evt = raw as AnthropicStreamEvent;
      switch (evt.type) {
        case 'error':
          // A provider error emitted mid-stream (e.g. `overloaded_error`) — throw
          // instead of silently ending the stream with no finishReason.
          throw this.mapStreamErrorEvent(response.status, evt);
        case 'message_start':
          startUsage = evt.message?.usage;
          break;
        case 'content_block_start': {
          const cb = evt.content_block;
          blocks.start(evt.index ?? 0, cb);
          if (cb?.type === 'tool_use') {
            const toolIndex = nextToolIndex++;
            toolIndexByBlock.set(evt.index ?? 0, toolIndex);
            yield {
              toolCallDelta: {
                index: toolIndex,
                ...(cb.id ? { id: cb.id } : {}),
                ...(cb.name ? { functionName: cb.name } : {}),
              },
              raw: evt,
            };
          }
          break;
        }
        case 'content_block_delta': {
          const d = evt.delta;
          blocks.apply(evt.index ?? 0, d);
          if (d?.type === 'text_delta' && typeof d.text === 'string' && d.text.length > 0) {
            yield { contentDelta: d.text, raw: evt };
          } else if (d?.type === 'input_json_delta' && typeof d.partial_json === 'string') {
            const toolIndex = toolIndexByBlock.get(evt.index ?? 0) ?? 0;
            yield { toolCallDelta: { index: toolIndex, argumentsDelta: d.partial_json }, raw: evt };
          }
          // thinking_delta / signature_delta → reasoning is not normalized
          // text; it reaches the caller only inside providerMetadata and raw.
          break;
        }
        case 'content_block_stop':
          blocks.stop(evt.index ?? 0);
          break;
        case 'message_delta': {
          const out: ChatStreamDelta = { raw: evt };
          if (evt.delta?.stop_reason) {
            out.finishReason = mapStopReason(evt.delta.stop_reason);
            out.providerMetadata = { anthropic: { content: blocks.toArray() } };
            sawStop = true;
          }
          // message_delta usage is cumulative; each count it carries as a
          // number supersedes message_start's (as the official SDK's
          // MessageStream merges), while a null or absent one keeps it.
          out.usage = mapUsage(mergeUsage(startUsage, evt.usage));
          yield out;
          break;
        }
        // message_stop / ping / unknown event types → ignored
      }
    }
    if (!sawStop) throw this.streamEndedEarlyError();
  }

  private authHeaders(input: ChatInput): Record<string, string> {
    return {
      'x-api-key': this.config.apiKey,
      'anthropic-version': this.config.anthropicVersion ?? DEFAULT_VERSION,
      ...this.config.headers,
      ...input._passthrough?.headers,
    };
  }

  private buildRequest(input: ChatInput, stream: boolean): Record<string, unknown> {
    const { system, messages } = splitMessages(input.messages);

    const body: Record<string, unknown> = {
      model: input.model,
      max_tokens: input.maxOutputTokens ?? this.config.defaultMaxTokens ?? DEFAULT_MAX_TOKENS,
      messages,
    };
    if (system) body.system = system;

    if (input.tools && input.tools.length > 0) {
      body.tools = input.tools.map((t) => ({
        name: t.function.name,
        ...(t.function.description ? { description: t.function.description } : {}),
        input_schema: t.function.parameters ?? { type: 'object', properties: {} },
      }));
    }
    if (input.toolChoice !== undefined) {
      const tc = mapToolChoice(input.toolChoice);
      if (tc) body.tool_choice = tc;
    }

    if (input.reasoning) {
      // Adaptive thinking with an effort level is the one reasoning control the
      // current models accept; `budget_tokens` is rejected from Claude 4.7 on.
      // It is sent for every model rather than switched on model id: Haiku 4.5
      // and older reject it with a 400, and a budget-mode caller passes
      // `thinking` through `_passthrough.body` instead. Sampling params are
      // omitted because thinking models reject non-default values.
      body.thinking = { type: 'adaptive' };
      // An empty effort means "no preference": medium is the middle level. Any
      // other value goes out as given, so the vendor validates it (and a newer
      // level works without a release here).
      body.output_config = { effort: input.reasoning.effort || 'medium' };
    } else {
      if (input.temperature !== undefined) body.temperature = input.temperature;
      if (input.topP !== undefined) body.top_p = input.topP;
    }
    if (input.stop !== undefined) {
      body.stop_sequences = Array.isArray(input.stop) ? input.stop : [input.stop];
    }
    // NOTE: `responseFormat` is intentionally NOT mapped — Anthropic has no
    // `response_format`; structured output is via tools / `_passthrough`.

    if (stream) body.stream = true;
    if (input._passthrough?.body) Object.assign(body, input._passthrough.body);
    return body;
  }

  private parseResult(json: AnthropicMessageResponse | null, requestedModel: string): ChatResult {
    const blocks = json?.content ?? [];
    let text = '';
    const toolCalls: ToolCall[] = [];
    for (const b of blocks) {
      if (b.type === 'text' && typeof b.text === 'string') {
        text += b.text;
      } else if (b.type === 'tool_use') {
        toolCalls.push({
          id: b.id ?? '',
          type: 'function',
          function: { name: b.name ?? '', arguments: JSON.stringify(b.input ?? {}) },
        });
      }
      // thinking / redacted_thinking → not normalized text; replayed via providerMetadata
    }
    return {
      message: {
        role: 'assistant',
        content: text.length > 0 ? text : null,
        ...(toolCalls.length > 0 ? { toolCalls } : {}),
        providerMetadata: { anthropic: { content: blocks } },
      },
      finishReason: mapStopReason(json?.stop_reason),
      usage: mapUsage(json?.usage),
      model: json?.model ?? requestedModel,
      raw: json,
    };
  }

  /**
   * A mid-stream `error` event arrives inside a 200 stream, so the error
   * carries that status; the code comes from the event's `error.type`, and an
   * event without a recognised type is `unknown`.
   */
  private mapStreamErrorEvent(httpStatus: number, evt: AnthropicStreamEvent): ConnectorError {
    const effective = errorEventStatus(evt.error?.type);
    const message = extractAnthropicError(evt) ?? `HTTP ${httpStatus}`;
    return new ConnectorError({
      message,
      statusCode: httpStatus,
      providerCode: effective === null ? 'unknown' : mapStatusToProviderCode(effective, message),
      providerMessage: message,
      cause: { raw: evt },
    });
  }

  private mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
    const message = extractAnthropicError(body) ?? `HTTP ${status}`;
    const providerCode = mapStatusToProviderCode(status, message);
    const cause: Record<string, unknown> = { raw: body ?? null };
    applyRetryAfter(cause, headers);
    return new ConnectorError({
      message,
      statusCode: status,
      providerCode,
      providerMessage: message,
      cause,
    });
  }
}

// ---------------------------------------------------------------------------
// Anthropic wire shapes (minimal — only fields the connector reads)
// ---------------------------------------------------------------------------

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
}

type AnthropicBlock = { type?: string; text?: string; id?: string; name?: string; input?: unknown } & Record<
  string,
  unknown
>;

interface AnthropicMessageResponse {
  model?: string;
  stop_reason?: string | null;
  usage?: AnthropicUsage;
  content?: AnthropicBlock[];
}

interface AnthropicStreamDelta {
  type?: string;
  text?: string;
  partial_json?: string;
  thinking?: string;
  signature?: string;
  citation?: unknown;
  stop_reason?: string | null;
}

interface AnthropicStreamEvent {
  type?: string;
  index?: number;
  message?: { model?: string; usage?: AnthropicUsage };
  content_block?: AnthropicBlock;
  delta?: AnthropicStreamDelta;
  usage?: AnthropicUsage;
  error?: { type?: string; message?: string };
}

/**
 * Rebuilds a streamed turn's `content` array: each block starts as its
 * `content_block_start` payload and its deltas are folded in (`text`,
 * `thinking`, `signature`, `citations`, and `input` parsed from the
 * accumulated `partial_json`). Blocks with no deltas (`redacted_thinking`,
 * server-tool results) stay exactly as started.
 */
class StreamedBlocks {
  private readonly blocks: Array<Record<string, unknown> | undefined> = [];
  private readonly partialJson = new Map<number, string>();

  start(index: number, block: AnthropicBlock | undefined): void {
    this.blocks[index] = { ...(block ?? {}) };
  }

  apply(index: number, delta: AnthropicStreamDelta | undefined): void {
    const block = this.blocks[index];
    if (!block || !delta) return;
    switch (delta.type) {
      case 'text_delta':
        block.text = `${typeof block.text === 'string' ? block.text : ''}${delta.text ?? ''}`;
        break;
      case 'thinking_delta':
        block.thinking = `${typeof block.thinking === 'string' ? block.thinking : ''}${delta.thinking ?? ''}`;
        break;
      case 'signature_delta':
        block.signature = `${typeof block.signature === 'string' ? block.signature : ''}${delta.signature ?? ''}`;
        break;
      case 'citations_delta':
        block.citations = [...(Array.isArray(block.citations) ? block.citations : []), delta.citation];
        break;
      case 'input_json_delta':
        this.partialJson.set(index, `${this.partialJson.get(index) ?? ''}${delta.partial_json ?? ''}`);
        break;
    }
  }

  stop(index: number): void {
    const json = this.partialJson.get(index);
    const block = this.blocks[index];
    if (json === undefined || !block) return;
    this.partialJson.delete(index);
    block.input = json.trim() === '' ? {} : safeJsonParse(json);
  }

  toArray(): Array<Record<string, unknown>> {
    for (const index of [...this.partialJson.keys()]) this.stop(index);
    return this.blocks.filter((b): b is Record<string, unknown> => b !== undefined);
  }
}

type AnthropicMessage = { role: string; content: unknown };

// ---------------------------------------------------------------------------
// Module-private mappers
// ---------------------------------------------------------------------------

function splitMessages(messages: ChatMessage[]): {
  system?: string;
  messages: AnthropicMessage[];
} {
  let system: string | undefined;
  const out: AnthropicMessage[] = [];
  let rebuilding = false;
  for (const m of messages) {
    if (m.role === 'system') {
      const text = contentToText(m.content);
      system = system ? `${system}\n\n${text}` : text;
      continue;
    }
    if (m.role === 'tool') {
      const block = {
        type: 'tool_result',
        tool_use_id: m.toolCallId ?? '',
        content: contentToText(m.content),
      };
      const last = out[out.length - 1];
      if (last && last.role === 'user' && isToolResultContent(last.content)) {
        (last.content as unknown[]).push(block);
      } else {
        out.push({ role: 'user', content: [block] });
      }
      continue;
    }
    if (m.role === 'assistant') {
      // Anthropic voids every thinking block after an edited or stripped turn
      // (accounts that enforce preserved thinking get a 400 for sending them),
      // so once one turn is rebuilt, every later one is rebuilt too.
      const verbatim = rebuilding ? null : replayableContent(m);
      if (verbatim === null) rebuilding = true;
      out.push({ role: m.role, content: verbatim ?? mapContentBlocks(m) });
      continue;
    }
    out.push({ role: m.role, content: mapContentBlocks(m) });
  }
  return { system, messages: out };
}

/**
 * The turn's stored content blocks (thinking included) when they still describe
 * this message, else `null` so the turn is rebuilt from its normalized fields.
 * They match when their `tool_use` blocks, in order, carry the message's
 * tool-call ids and inputs equal to the parsed arguments, and their joined text
 * is the message's text — a caller who edited the turn gets the edit, not the
 * stale blocks. A rebuilt turn loses its thinking blocks, and with them the
 * earlier reasoning.
 */
function replayableContent(m: ChatMessage): unknown[] | null {
  const stored = (m.providerMetadata?.anthropic as { content?: unknown } | undefined)?.content;
  if (!Array.isArray(stored) || stored.length === 0) return null;
  if (Array.isArray(m.content) && m.content.some((p) => p.type !== 'text')) return null;
  const blocks = stored as Array<AnthropicBlock | null>;
  const storedCalls = blocks.filter((b) => b?.type === 'tool_use');
  const calls = m.toolCalls ?? [];
  if (storedCalls.length !== calls.length) return null;
  const callsMatch = storedCalls.every((b, i) => {
    const call = calls[i]!;
    if (b?.id !== call.id) return false;
    let args: unknown;
    try {
      args = JSON.parse(call.function.arguments);
    } catch {
      return false;
    }
    return jsonEqual(b.input, args);
  });
  if (!callsMatch) return null;
  const storedText = blocks
    .filter((b) => b?.type === 'text')
    .map((b) => (typeof b?.text === 'string' ? b.text : ''))
    .join('');
  if (storedText !== contentToText(m.content)) return null;
  return stored;
}

/** Structural equality of two JSON values; object key order does not matter. */
function jsonEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) {
    const other = b as unknown[];
    return a.length === other.length && a.every((v, i) => jsonEqual(v, other[i]));
  }
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  return (
    ka.length === kb.length &&
    ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && jsonEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))
  );
}

function contentToText(content: ChatMessage['content']): string {
  if (content === null) return '';
  if (typeof content === 'string') return content;
  return content
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map((p) => p.text)
    .join('');
}

function mapContentBlocks(m: ChatMessage): unknown[] {
  const blocks: unknown[] = [];
  if (typeof m.content === 'string') {
    if (m.content.length > 0) blocks.push({ type: 'text', text: m.content });
  } else if (Array.isArray(m.content)) {
    for (const p of m.content) {
      if (p.type === 'text') {
        // Anthropic rejects empty text blocks; an empty part carries nothing.
        if (p.text.length > 0) blocks.push({ type: 'text', text: p.text });
      } else {
        blocks.push({
          type: 'image',
          source: { type: 'base64', media_type: p.mediaType, data: p.base64 },
        });
      }
    }
  }
  if (m.role === 'assistant' && m.toolCalls) {
    for (const tc of m.toolCalls) {
      blocks.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input: safeJsonParse(tc.function.arguments),
      });
    }
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  return blocks;
}

function isToolResultContent(content: unknown): boolean {
  return (
    Array.isArray(content) &&
    content.length > 0 &&
    typeof content[0] === 'object' &&
    content[0] !== null &&
    (content[0] as { type?: string }).type === 'tool_result'
  );
}

function mapToolChoice(tc: ToolChoice): Record<string, unknown> | null {
  if (tc === 'auto') return { type: 'auto' };
  if (tc === 'required') return { type: 'any' };
  if (tc === 'none') return { type: 'none' };
  if (typeof tc === 'object' && tc.type === 'function') {
    return { type: 'tool', name: tc.function.name };
  }
  return null;
}

function mapStopReason(sr: string | null | undefined): FinishReason {
  switch (sr) {
    case 'end_turn':
    case 'stop_sequence':
      return 'stop';
    case 'max_tokens':
      return 'length';
    case 'tool_use':
      return 'tool_calls';
    // The context window filled up mid-generation: output was cut short, the
    // same outcome as hitting max_tokens.
    case 'model_context_window_exceeded':
      return 'length';
    case 'refusal':
      return 'content_filter';
    // pause_turn (a server-tool loop paused; the caller resends to continue) has
    // no normalized counterpart.
    default:
      return 'unknown';
  }
}

function mergeUsage(base: AnthropicUsage | undefined, update: AnthropicUsage | undefined): AnthropicUsage {
  const merged: AnthropicUsage = { ...base };
  for (const [key, value] of Object.entries(update ?? {})) {
    if (typeof value === 'number') (merged as Record<string, unknown>)[key] = value;
  }
  return merged;
}

/**
 * Anthropic's `input_tokens` counts only the tokens after the last cache
 * breakpoint; cache reads and writes are billed input too, so they are added.
 */
function mapUsage(u: AnthropicUsage | undefined): Usage {
  const inputTokens =
    (u?.input_tokens ?? 0) + (u?.cache_read_input_tokens ?? 0) + (u?.cache_creation_input_tokens ?? 0);
  const outputTokens = u?.output_tokens ?? 0;
  return { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens };
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return {};
  }
}

function extractAnthropicError(body: unknown): string | null {
  if (typeof body === 'string') return body;
  if (!body || typeof body !== 'object') return null;
  const b = body as Record<string, unknown>;
  if (b.error && typeof b.error === 'object') {
    const em = (b.error as Record<string, unknown>).message;
    if (typeof em === 'string') return em;
  }
  if (typeof b.message === 'string') return b.message;
  return null;
}

/**
 * The HTTP status an Anthropic SSE `error` event's `type` stands for on the
 * non-stream path, or `null` for a missing or unrecognised type.
 */
function errorEventStatus(type: string | undefined): number | null {
  switch (type) {
    case 'authentication_error':
      return 401;
    case 'billing_error':
      return 402;
    case 'permission_error':
      return 403;
    case 'not_found_error':
      return 404;
    case 'conflict_error':
      return 409;
    case 'request_too_large':
      return 413;
    case 'rate_limit_error':
      return 429;
    case 'invalid_request_error':
      return 400;
    case 'timeout_error':
      return 504;
    case 'api_error':
      return 500;
    case 'overloaded_error':
      return 529;
    default:
      return null;
  }
}

function mapStatusToProviderCode(status: number, message: string): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  // 402 is credit/quota exhaustion, which OpenAI (429 insufficient_quota) and
  // Gemini (RESOURCE_EXHAUSTED) already report as rate-limited, and
  // @thinwrap/notifications maps Pusher Beams' 402 the same way.
  if (status === 429 || status === 402) return 'rate_limited';
  // Any 5xx is a provider-side failure, gateway / Cloudflare-origin ones
  // (504/520/522/524) included, so a consumer's retry policy fires on them.
  // A 408 is a timeout too.
  if (status >= 500 || status === 408) {
    return 'provider_unavailable';
  }
  if (status >= 400) {
    // Only the overflow messages count: a looser match ("context", "max_tokens")
    // would also catch ordinary 400s, and a caller that trims history on this
    // code would then loop forever.
    if (status === 400 && isContextOverflow(message)) return 'context_length_exceeded';
    // 413 request_too_large (a byte limit, not a token one),
    // 404, 409, 422: the request cannot succeed as sent.
    return 'invalid_request';
  }
  return 'unknown';
}

/** "prompt is too long: N tokens > M maximum" and "input length and `max_tokens` exceed context limit: …". */
function isContextOverflow(message: string): boolean {
  const msg = message.toLowerCase();
  return msg.includes('prompt is too long') || msg.includes('exceed context limit');
}
