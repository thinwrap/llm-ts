import { appendPassthroughQuery, BaseConnector, parseSSEStream } from '../../base/base.connector';
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
} from '../../types';
import type { AnthropicConfig } from './anthropic.config';

const DEFAULT_BASE_URL = 'https://api.anthropic.com/v1';
const DEFAULT_VERSION = '2023-06-01';
const DEFAULT_MAX_TOKENS = 4096;

/**
 * Native adapter for the Anthropic **Messages API** (`POST /v1/messages`).
 * Anthropic is structurally different from OpenAI Chat Completions — different
 * auth, a top-level `system`, typed content blocks, tool results carried inside
 * a user turn, `stop_sequences`, and Anthropic-shaped SSE events — so this is a
 * full translation layer, not an OpenAI passthrough. It emits the identical
 * normalized `ChatResult` / `ChatStreamDelta` / `ConnectorError` as every other
 * connector, so it is drop-in swappable via the `Chat` facade.
 *
 * Not normalized here (per D2/D4, surfaced via `raw` or `_passthrough`, never
 * emulated): `responseFormat` (Anthropic has no `response_format` — use
 * tool-based structured output via `_passthrough`); reasoning CoT output
 * (`thinking` blocks land in `raw`); prompt caching (`cache_control` via
 * `_passthrough`, cached-token counts in `raw`).
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

    let inputTokens = 0;
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
          throw this.mapVendorError(errorEventStatus(evt.error?.type), evt, new Headers());
        case 'message_start':
          inputTokens = evt.message?.usage?.input_tokens ?? 0;
          break;
        case 'content_block_start': {
          const cb = evt.content_block;
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
          if (d?.type === 'text_delta' && typeof d.text === 'string' && d.text.length > 0) {
            yield { contentDelta: d.text, raw: evt };
          } else if (d?.type === 'input_json_delta' && typeof d.partial_json === 'string') {
            const toolIndex = toolIndexByBlock.get(evt.index ?? 0) ?? 0;
            yield { toolCallDelta: { index: toolIndex, argumentsDelta: d.partial_json }, raw: evt };
          }
          // thinking_delta → CoT output not normalized (D2); available only via raw event stream
          break;
        }
        case 'message_delta': {
          const out: ChatStreamDelta = { raw: evt };
          if (evt.delta?.stop_reason) out.finishReason = mapStopReason(evt.delta.stop_reason);
          const outputTokens = evt.usage?.output_tokens ?? 0;
          out.usage = { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens };
          yield out;
          break;
        }
        // message_stop / content_block_stop / ping → ignored
      }
    }
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

    const thinking = input.reasoning
      ? { type: 'enabled', budget_tokens: effortBudget(input.reasoning.effort) }
      : undefined;
    const budget = thinking ? (thinking.budget_tokens as number) : 0;
    const callerSetMax = input.maxOutputTokens !== undefined;
    let maxTokens = input.maxOutputTokens ?? this.config.defaultMaxTokens ?? DEFAULT_MAX_TOKENS;
    if (thinking && maxTokens <= budget) {
      // Anthropic requires max_tokens > thinking.budget_tokens (max_tokens is the
      // TOTAL, thinking included). If the CALLER set an explicit maxOutputTokens
      // below the budget, silently inflating it would spend past their cap — throw
      // so they lower reasoning.effort or raise the cap. Only a non-explicit default
      // is inflated to keep the request valid.
      if (callerSetMax) {
        throw new ConnectorError({
          message: `maxOutputTokens (${maxTokens}) must be greater than the reasoning budget (${budget}) for extended thinking — raise maxOutputTokens or lower reasoning.effort`,
          statusCode: null,
          providerCode: 'invalid_request',
        });
      }
      maxTokens = budget + DEFAULT_MAX_TOKENS;
    }

    const body: Record<string, unknown> = {
      model: input.model,
      max_tokens: maxTokens,
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

    if (thinking) {
      // Extended thinking requires temperature=1 and disallows top_p/top_k — omit sampling.
      body.thinking = thinking;
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
      // thinking blocks → raw only (D2)
    }
    const inputTokens = json?.usage?.input_tokens ?? 0;
    const outputTokens = json?.usage?.output_tokens ?? 0;
    return {
      message: {
        role: 'assistant',
        content: text.length > 0 ? text : null,
        ...(toolCalls.length > 0 ? { toolCalls } : {}),
      },
      finishReason: mapStopReason(json?.stop_reason),
      usage: { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens },
      model: json?.model ?? requestedModel,
      raw: json,
    };
  }

  private mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
    const message = extractAnthropicError(body) ?? `HTTP ${status}`;
    const providerCode = mapStatusToProviderCode(status, message);
    const retryAfter = headers.get('retry-after');
    const cause: Record<string, unknown> = { raw: body ?? null };
    if (retryAfter != null) {
      cause.retryAfter = retryAfter;
      const secs = parseRetryAfterSeconds(retryAfter);
      if (secs != null) cause.retryAfterSeconds = secs;
    }
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

interface AnthropicMessageResponse {
  model?: string;
  stop_reason?: string | null;
  usage?: { input_tokens?: number; output_tokens?: number };
  content?: Array<{ type?: string; text?: string; id?: string; name?: string; input?: unknown }>;
}

interface AnthropicStreamEvent {
  type?: string;
  index?: number;
  message?: { model?: string; usage?: { input_tokens?: number } };
  content_block?: { type?: string; id?: string; name?: string };
  delta?: { type?: string; text?: string; partial_json?: string; stop_reason?: string | null };
  usage?: { output_tokens?: number };
  error?: { type?: string; message?: string };
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
    out.push({ role: m.role, content: mapContentBlocks(m) });
  }
  return { system, messages: out };
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
        blocks.push({ type: 'text', text: p.text });
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
    case 'refusal':
      return 'content_filter';
    default:
      return 'unknown';
  }
}

function effortBudget(effort: 'low' | 'medium' | 'high'): number {
  switch (effort) {
    case 'low':
      return 1024;
    case 'medium':
      return 4096;
    case 'high':
      return 12288;
  }
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

/** Map an Anthropic SSE `error` event's `type` to the HTTP status its non-stream path would carry. */
function errorEventStatus(type: string | undefined): number {
  switch (type) {
    case 'authentication_error':
      return 401;
    case 'permission_error':
      return 403;
    case 'not_found_error':
      return 404;
    case 'rate_limit_error':
      return 429;
    case 'invalid_request_error':
      return 400;
    case 'overloaded_error':
      return 529;
    default:
      return 500;
  }
}

function mapStatusToProviderCode(status: number, message: string): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  if (status === 429) return 'rate_limited';
  // Any 5xx is a provider-side failure. A fixed {500,502,503,529} whitelist
  // dropped 504/520/522/524 (gateway / Cloudflare-origin) to unknown, so a
  // consumer's retry policy wouldn't fire on them.
  if (status >= 500) {
    return 'provider_unavailable';
  }
  if (status === 400 || status === 404 || status === 413 || status === 422) {
    const msg = message.toLowerCase();
    if (msg.includes('too long') || msg.includes('context') || msg.includes('max_tokens')) {
      return 'context_length_exceeded';
    }
    return 'invalid_request';
  }
  return 'unknown';
}

function parseRetryAfterSeconds(header: string): number | null {
  const asNum = Number(header);
  if (Number.isFinite(asNum)) return asNum;
  const asDate = Date.parse(header);
  if (!Number.isNaN(asDate)) {
    const secs = Math.round((asDate - Date.now()) / 1000);
    return secs >= 0 ? secs : 0;
  }
  return null;
}
