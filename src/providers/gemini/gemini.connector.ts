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
  ResponseFormat,
  ToolCall,
  ToolChoice,
  Usage,
} from '../../types';
import type { GeminiConfig } from './gemini.config';

const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';

/**
 * Native adapter for the Google Gemini (AI Studio) `generateContent` API.
 * Gemini uses `contents[]`/`parts[]` (roles `user`/`model`), a top-level
 * `systemInstruction`, `functionDeclarations`, and keys tool results by
 * function NAME (not id). This connector translates the normalized surface to
 * that shape and back, emitting the identical `ChatResult`/`ChatStreamDelta`/
 * `ConnectorError`. Unlike Bedrock, Gemini streaming IS SSE
 * (`:streamGenerateContent?alt=sse`), so streaming is truly incremental.
 *
 * Passthrough/raw (not emulated): Gemini-3 `thought_signature` round-trip and
 * reasoning CoT output (`raw`), prompt caching (`cachedContent` via
 * `_passthrough`), safety settings / video params (`_passthrough`).
 */
export class GeminiConnector extends BaseConnector implements IChatConnector {
  public readonly id = 'gemini';
  private readonly config: GeminiConfig;
  private readonly baseUrl: string;

  constructor(config: GeminiConfig) {
    super(config.fetch);
    this.config = config;
    this.baseUrl = (config.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
  }

  async complete(input: ChatInput): Promise<ChatResult> {
    const url = appendPassthroughQuery(
      `${this.baseUrl}/models/${encodeURIComponent(input.model)}:generateContent`,
      input._passthrough?.query,
    );
    const response = await this.sendPostJson(url, this.buildRequest(input), this.authHeaders(input), input.signal);
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw this.mapVendorError(response.status, errBody, response.headers);
    }
    const json = this.requireDecodedBody<GeminiResponse>((await response.json().catch(() => null)) as GeminiResponse | null, response.status);
    return this.parseResult(json, input.model);
  }

  async *stream(input: ChatInput): AsyncGenerator<ChatStreamDelta> {
    const url = appendPassthroughQuery(
      `${this.baseUrl}/models/${encodeURIComponent(input.model)}:streamGenerateContent?alt=sse`,
      input._passthrough?.query,
    );
    const response = await this.invokeFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', ...this.authHeaders(input) },
      body: JSON.stringify(this.buildRequest(input)),
      signal: input.signal,
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw this.mapVendorError(response.status, errBody, response.headers);
    }

    let sawFunctionCall = false;
    // Gemini yields each functionCall part in its own chunk, so assign a
    // monotonically-incrementing tool-call index (mirrors Anthropic's
    // `nextToolIndex++` and the non-stream `idx`) — a hardcoded 0 would make a
    // consumer keyed on `index` merge parallel/multiple calls into one.
    let nextToolIndex = 0;
    // Gemini's `usageMetadata` is cumulative and can appear on more than one
    // chunk. Buffer the latest and emit a SINGLE terminal `usage` delta at
    // stream end (matching the OpenAI-compat `include_usage` trailer and
    // Anthropic's single `message_delta`) so a consumer summing `delta.usage`
    // across the stream never double-counts.
    let latestUsage: GeminiResponse['usageMetadata'];
    let latestUsageChunk: GeminiResponse | undefined;
    for await (const raw of parseSSEStream(response)) {
      const chunk = raw as GeminiResponse;
      // A provider error emitted mid-stream arrives as an `{"error": …}` frame —
      // throw instead of silently ending the stream with no finishReason.
      if (chunk.error != null) {
        throw this.mapVendorError(errorFrameStatus(chunk.error), chunk, new Headers());
      }
      // A prompt-level safety block on a stream arrives as a chunk with no
      // candidate and only promptFeedback — emit a content_filter finish so the
      // consumer sees the block instead of a silent, empty end.
      if (chunk.candidates?.[0] == null && chunk.promptFeedback?.blockReason) {
        yield { finishReason: 'content_filter', raw: chunk };
        continue;
      }
      const cand = chunk.candidates?.[0];
      for (const p of cand?.content?.parts ?? []) {
        if (p.thought === true) {
          continue; // reasoning CoT, not answer content (available via raw)
        } else if (typeof p.text === 'string' && p.text.length > 0) {
          yield { contentDelta: p.text, raw: chunk };
        } else if (p.functionCall) {
          sawFunctionCall = true;
          yield {
            toolCallDelta: {
              index: nextToolIndex++,
              id: p.functionCall.id ?? `call_${p.functionCall.name ?? ''}`,
              functionName: p.functionCall.name,
              argumentsDelta: JSON.stringify(p.functionCall.args ?? {}),
            },
            raw: chunk,
          };
        }
      }
      if (chunk.usageMetadata) {
        latestUsage = chunk.usageMetadata;
        latestUsageChunk = chunk;
      }
      if (cand?.finishReason) {
        yield {
          finishReason: sawFunctionCall ? 'tool_calls' : mapFinishReason(cand.finishReason),
          raw: chunk,
        };
      }
    }
    if (latestUsage) yield { usage: mapUsage(latestUsage), raw: latestUsageChunk };
  }

  private authHeaders(input: ChatInput): Record<string, string> {
    return {
      'x-goog-api-key': this.config.apiKey,
      ...this.config.headers,
      ...input._passthrough?.headers,
    };
  }

  private buildRequest(input: ChatInput): Record<string, unknown> {
    const { systemInstruction, contents } = splitMessages(input.messages);

    const body: Record<string, unknown> = { contents };
    if (systemInstruction) body.systemInstruction = systemInstruction;

    if (input.tools && input.tools.length > 0) {
      body.tools = [
        {
          functionDeclarations: input.tools.map((t) => ({
            name: t.function.name,
            ...(t.function.description ? { description: t.function.description } : {}),
            ...(t.function.parameters ? { parameters: t.function.parameters } : {}),
          })),
        },
      ];
      const fc = mapFunctionCallingConfig(input.toolChoice);
      if (fc) body.toolConfig = { functionCallingConfig: fc };
    }

    const generationConfig: Record<string, unknown> = {};
    const maxTok = input.maxOutputTokens ?? this.config.defaultMaxTokens;
    if (maxTok !== undefined) generationConfig.maxOutputTokens = maxTok;
    if (input.temperature !== undefined) generationConfig.temperature = input.temperature;
    if (input.topP !== undefined) generationConfig.topP = input.topP;
    if (input.stop !== undefined) {
      generationConfig.stopSequences = Array.isArray(input.stop) ? input.stop : [input.stop];
    }
    applyResponseFormat(generationConfig, input.responseFormat);
    if (input.reasoning) {
      generationConfig.thinkingConfig = { thinkingBudget: effortBudget(input.reasoning.effort) };
    }
    if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;

    if (input._passthrough?.body) Object.assign(body, input._passthrough.body);
    return body;
  }

  private parseResult(json: GeminiResponse | null, requestedModel: string): ChatResult {
    const cand = json?.candidates?.[0];
    const parts = cand?.content?.parts ?? [];
    let text = '';
    const toolCalls: ToolCall[] = [];
    let idx = 0;
    for (const p of parts) {
      if (p.thought === true) {
        continue; // reasoning CoT, not answer content (available via raw)
      } else if (typeof p.text === 'string') {
        text += p.text;
      } else if (p.functionCall) {
        toolCalls.push({
          id: p.functionCall.id ?? `call_${idx}`,
          type: 'function',
          function: {
            name: p.functionCall.name ?? '',
            arguments: JSON.stringify(p.functionCall.args ?? {}),
          },
        });
        idx++;
      }
    }
    // No candidate came back: a prompt-level safety block reports only
    // `promptFeedback.blockReason`. Surface it as content_filter rather than a
    // silent empty completion with finishReason unknown.
    const finishReason: FinishReason =
      toolCalls.length > 0
        ? 'tool_calls'
        : cand?.finishReason == null && json?.promptFeedback?.blockReason
          ? 'content_filter'
          : mapFinishReason(cand?.finishReason);
    return {
      message: {
        role: 'assistant',
        content: text.length > 0 ? text : null,
        ...(toolCalls.length > 0 ? { toolCalls } : {}),
      },
      finishReason,
      usage: mapUsage(json?.usageMetadata),
      model: json?.modelVersion ?? requestedModel,
      raw: json,
    };
  }

  private mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
    const message = extractGeminiError(body) ?? `HTTP ${status}`;
    const providerCode = mapStatusToProviderCode(status, message);
    const cause: Record<string, unknown> = { raw: body ?? null };
    const retryAfter = headers.get('retry-after');
    if (retryAfter != null) {
      cause.retryAfter = retryAfter;
      const secs = Number(retryAfter);
      if (Number.isFinite(secs)) cause.retryAfterSeconds = secs;
    }
    // Gemini has no rate-limit headers — retry timing lives in the 429 body's RetryInfo.
    if (cause.retryAfterSeconds === undefined) {
      const bodySecs = extractRetryDelaySeconds(body);
      if (bodySecs != null) cause.retryAfterSeconds = bodySecs;
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
// Gemini wire shapes (minimal)
// ---------------------------------------------------------------------------

interface GeminiPart {
  text?: string;
  functionCall?: { id?: string; name?: string; args?: unknown };
  /**
   * Gemini-3 marks a reasoning/chain-of-thought part with `thought: true`. It is
   * NOT user-facing answer content — concatenating it into `content` would leak
   * the model's private reasoning into the normalized text. Skipped from
   * `content`; the full parts (including thoughts) remain available via `raw`.
   */
  thought?: boolean;
}

interface GeminiResponse {
  modelVersion?: string;
  usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number; totalTokenCount?: number };
  candidates?: Array<{ content?: { parts?: GeminiPart[] }; finishReason?: string | null }>;
  /**
   * Prompt-level safety block: when the *input* is rejected, Gemini returns no
   * candidates and only `promptFeedback.blockReason` (e.g. `SAFETY`, `BLOCKLIST`,
   * `PROHIBITED_CONTENT`, `OTHER`). Without reading it the result would look like
   * a clean empty completion with `finishReason: unknown`.
   */
  promptFeedback?: { blockReason?: string | null };
  /** Mid-stream error frame: `{"error": {"code": 429, "message": …, "status": …}}`. */
  error?: { code?: number; message?: string; status?: string };
}

type GeminiContent = { role: string; parts: unknown[] };

// ---------------------------------------------------------------------------
// Mappers
// ---------------------------------------------------------------------------

function splitMessages(messages: ChatMessage[]): {
  systemInstruction?: { parts: Array<{ text: string }> };
  contents: GeminiContent[];
} {
  const nameById = new Map<string, string>();
  for (const m of messages) {
    if (m.role === 'assistant' && m.toolCalls) {
      for (const tc of m.toolCalls) nameById.set(tc.id, tc.function.name);
    }
  }

  let systemText = '';
  const contents: GeminiContent[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const t = contentToText(m.content);
      systemText = systemText ? `${systemText}\n\n${t}` : t;
      continue;
    }
    if (m.role === 'tool') {
      const name =
        m.name ?? (m.toolCallId ? nameById.get(m.toolCallId) : undefined) ?? m.toolCallId ?? 'tool';
      const part = {
        functionResponse: { name, response: { result: parseJsonOrString(contentToText(m.content)) } },
      };
      const last = contents[contents.length - 1];
      if (last && last.role === 'user' && isFunctionResponseParts(last.parts)) {
        last.parts.push(part);
      } else {
        contents.push({ role: 'user', parts: [part] });
      }
      continue;
    }
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts: mapParts(m) });
  }

  return {
    systemInstruction: systemText ? { parts: [{ text: systemText }] } : undefined,
    contents,
  };
}

function contentToText(content: ChatMessage['content']): string {
  if (content === null) return '';
  if (typeof content === 'string') return content;
  return content
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map((p) => p.text)
    .join('');
}

function mapParts(m: ChatMessage): unknown[] {
  const parts: unknown[] = [];
  if (typeof m.content === 'string') {
    if (m.content.length > 0) parts.push({ text: m.content });
  } else if (Array.isArray(m.content)) {
    for (const p of m.content) {
      parts.push(
        p.type === 'text'
          ? { text: p.text }
          : { inlineData: { mimeType: p.mediaType, data: p.base64 } },
      );
    }
  }
  if (m.role === 'assistant' && m.toolCalls) {
    for (const tc of m.toolCalls) {
      parts.push({ functionCall: { name: tc.function.name, args: safeJsonParse(tc.function.arguments) } });
    }
  }
  if (parts.length === 0) parts.push({ text: '' });
  return parts;
}

function isFunctionResponseParts(parts: unknown[]): boolean {
  return (
    parts.length > 0 &&
    typeof parts[0] === 'object' &&
    parts[0] !== null &&
    'functionResponse' in (parts[0] as object)
  );
}

function mapFunctionCallingConfig(tc: ToolChoice | undefined): Record<string, unknown> | null {
  if (tc === undefined) return null;
  if (tc === 'auto') return { mode: 'AUTO' };
  if (tc === 'required') return { mode: 'ANY' };
  if (tc === 'none') return { mode: 'NONE' };
  if (typeof tc === 'object' && tc.type === 'function') {
    return { mode: 'ANY', allowedFunctionNames: [tc.function.name] };
  }
  return null;
}

function applyResponseFormat(
  generationConfig: Record<string, unknown>,
  rf: ResponseFormat | undefined,
): void {
  if (!rf) return;
  if (rf.type === 'json_object') {
    generationConfig.responseMimeType = 'application/json';
  } else if (rf.type === 'json_schema') {
    generationConfig.responseMimeType = 'application/json';
    generationConfig.responseSchema = rf.jsonSchema.schema;
  }
}

function mapFinishReason(fr: string | null | undefined): FinishReason {
  switch (fr) {
    case 'STOP':
      return 'stop';
    case 'MAX_TOKENS':
      return 'length';
    case 'SAFETY':
    case 'RECITATION':
    case 'BLOCKLIST':
    case 'PROHIBITED_CONTENT':
    case 'SPII':
      return 'content_filter';
    default:
      return 'unknown';
  }
}

function mapUsage(u: GeminiResponse['usageMetadata']): Usage {
  const inputTokens = u?.promptTokenCount ?? 0;
  const outputTokens = u?.candidatesTokenCount ?? 0;
  return { inputTokens, outputTokens, totalTokens: u?.totalTokenCount ?? inputTokens + outputTokens };
}

function effortBudget(effort: 'low' | 'medium' | 'high'): number {
  switch (effort) {
    case 'low':
      return 1024;
    case 'medium':
      return 8192;
    case 'high':
      return 24576;
  }
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return {};
  }
}

function parseJsonOrString(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
}

function extractGeminiError(body: unknown): string | null {
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

function extractRetryDelaySeconds(body: unknown): number | null {
  if (!body || typeof body !== 'object') return null;
  const err = (body as Record<string, unknown>).error;
  if (!err || typeof err !== 'object') return null;
  const details = (err as Record<string, unknown>).details;
  if (!Array.isArray(details)) return null;
  for (const d of details) {
    const rd = (d as Record<string, unknown>)?.retryDelay;
    if (typeof rd === 'string') {
      const match = rd.match(/^(\d+(?:\.\d+)?)s$/);
      if (match) return Math.round(Number(match[1]));
    }
  }
  return null;
}

/** HTTP status for a mid-stream Gemini `{"error": …}` frame — Gemini's `error.code` IS the HTTP status. */
function errorFrameStatus(error: GeminiResponse['error']): number {
  const code = error?.code;
  return typeof code === 'number' && code >= 400 && code <= 599 ? code : 500;
}

function mapStatusToProviderCode(status: number, message: string): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'provider_unavailable';
  if (status === 400 || status === 404 || status === 422) {
    const msg = message.toLowerCase();
    if (msg.includes('too long') || msg.includes('context') || msg.includes('token count')) {
      return 'context_length_exceeded';
    }
    return 'invalid_request';
  }
  return 'unknown';
}
