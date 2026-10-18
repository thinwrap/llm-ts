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
  ResponseFormat,
  ToolCall,
  ToolChoice,
  Usage,
} from '../../types';
import type { GeminiConfig } from './gemini.config';

const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';

/**
 * Google's documented value for a `thoughtSignature` the client does not have
 * (a hand-built history, or one from another provider). Gemini 3 rejects a
 * model turn whose function calls carry no signature at all.
 */
const SKIP_THOUGHT_SIGNATURE = 'skip_thought_signature_validator';

/** The id this connector invents for a function call that came back without one. */
const SYNTHETIC_CALL_ID = /^call_\d+$/;

/**
 * Native adapter for the Google Gemini (AI Studio) `generateContent` API.
 * Gemini uses `contents[]`/`parts[]` (roles `user`/`model`), a top-level
 * `systemInstruction`, `functionDeclarations`, and keys tool results by
 * function NAME (not id). This connector translates the normalized surface to
 * that shape and back, emitting the identical `ChatResult`/`ChatStreamDelta`/
 * `ConnectorError`. Unlike Bedrock, Gemini streaming IS SSE
 * (`:streamGenerateContent?alt=sse`), so streaming is truly incremental.
 *
 * A function call's `thoughtSignature` travels on
 * `ToolCall.providerMetadata.gemini` and goes back on the replayed
 * `functionCall` part, which Gemini 3 requires in a tool loop.
 *
 * Passthrough/raw (not emulated): reasoning CoT output (`raw`), prompt caching
 * (`cachedContent` via `_passthrough`), safety settings / video params
 * (`_passthrough`).
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
    // Only a prompt-level block legitimately omits `candidates` (parseResult
    // turns it into content_filter); any other body without them is not a result.
    if (!Array.isArray(json.candidates) && !json.promptFeedback?.blockReason) {
      throw this.missingPayloadError(response.status, json, 'no candidates');
    }
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
    this.requireEventStream(response);

    let sawFunctionCall = false;
    // Gemini yields each functionCall part in its own chunk, so assign a
    // monotonically-incrementing tool-call index (mirrors Anthropic's
    // `nextToolIndex++` and the non-stream `idx`) — a hardcoded 0 would make a
    // consumer keyed on `index` merge parallel/multiple calls into one.
    let nextToolIndex = 0;
    // Gemini's `usageMetadata` is cumulative and can appear on more than one
    // chunk, even after the finish. Buffer the latest and report it ONCE, on
    // the finish delta emitted at stream end, so a consumer summing
    // `delta.usage` never double-counts and finds usage beside the finish as on
    // every other provider.
    let latestUsage: GeminiResponse['usageMetadata'];
    let finish: { reason: FinishReason; chunk: GeminiResponse } | undefined;
    for await (const raw of parseSSEStream(response)) {
      const chunk = raw as GeminiResponse;
      // A provider error emitted mid-stream arrives as an `{"error": …}` frame —
      // throw instead of silently ending the stream with no finishReason.
      if (chunk.error != null) {
        throw this.mapStreamErrorFrame(response.status, chunk);
      }
      // A prompt-level safety block on a stream arrives as a chunk with no
      // candidate and only promptFeedback — emit a content_filter finish so the
      // consumer sees the block instead of a silent, empty end.
      if (chunk.candidates?.[0] == null && chunk.promptFeedback?.blockReason) {
        finish = { reason: 'content_filter', chunk };
        if (chunk.usageMetadata) latestUsage = chunk.usageMetadata;
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
          const index = nextToolIndex++;
          const providerMetadata = signatureMetadata(p);
          yield {
            toolCallDelta: {
              index,
              // Index-based, so parallel calls to one function stay distinct.
              id: p.functionCall.id || `call_${index}`,
              functionName: p.functionCall.name,
              argumentsDelta: JSON.stringify(p.functionCall.args ?? {}),
              ...(providerMetadata ? { providerMetadata } : {}),
            },
            raw: chunk,
          };
        }
      }
      if (chunk.usageMetadata) latestUsage = chunk.usageMetadata;
      if (cand?.finishReason) {
        finish = { reason: sawFunctionCall ? 'tool_calls' : mapFinishReason(cand.finishReason), chunk };
      }
    }
    if (!finish) throw this.streamEndedEarlyError();
    yield {
      finishReason: finish.reason,
      ...(latestUsage ? { usage: mapUsage(latestUsage) } : {}),
      raw: finish.chunk,
    };
  }

  /**
   * A mid-stream `{"error": …}` frame arrives inside a 200 stream, so the
   * error carries that status; the code comes from the frame's own `code` or
   * google.rpc `status` name, and a frame with neither is `unknown`.
   */
  private mapStreamErrorFrame(httpStatus: number, chunk: GeminiResponse): ConnectorError {
    const effective = errorFrameStatus(chunk.error);
    const message = extractGeminiError(chunk) ?? `HTTP ${httpStatus}`;
    return new ConnectorError({
      message,
      statusCode: httpStatus,
      providerCode: effective === null ? 'unknown' : mapStatusToProviderCode(effective, message, chunk),
      providerMessage: message,
      cause: { raw: chunk },
    });
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
            // `parameters` takes only Google's OpenAPI subset and 400s on plain
            // JSON Schema (`additionalProperties`, `$ref`); this field takes it.
            ...(t.function.parameters ? { parametersJsonSchema: t.function.parameters } : {}),
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
        const providerMetadata = signatureMetadata(p);
        toolCalls.push({
          id: p.functionCall.id || `call_${idx}`,
          type: 'function',
          function: {
            name: p.functionCall.name ?? '',
            arguments: JSON.stringify(p.functionCall.args ?? {}),
          },
          ...(providerMetadata ? { providerMetadata } : {}),
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
    const providerCode = mapStatusToProviderCode(status, message, body);
    const cause: Record<string, unknown> = { raw: body ?? null };
    applyRetryAfter(cause, headers);
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
  /** Opaque reasoning-state token Gemini 3 needs back on the replayed part. */
  thoughtSignature?: string;
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
  usageMetadata?: {
    promptTokenCount?: number;
    toolUsePromptTokenCount?: number;
    candidatesTokenCount?: number;
    thoughtsTokenCount?: number;
    totalTokenCount?: number;
  };
  candidates?: Array<{ content?: { parts?: GeminiPart[] }; finishReason?: string | null }>;
  /**
   * Prompt-level safety block: when the *input* is rejected, Gemini returns no
   * candidates and only `promptFeedback.blockReason` (e.g. `SAFETY`, `BLOCKLIST`,
   * `PROHIBITED_CONTENT`, `OTHER`). Without reading it the result would look like
   * a clean empty completion with `finishReason: unknown`.
   */
  promptFeedback?: { blockReason?: string | null };
  /** Mid-stream error frame: `{"error": {"code": 429, "message": …, "status": …}}`. */
  error?: { code?: number | string; message?: string; status?: string };
}

type GeminiContent = { role: string; parts: unknown[] };

// ---------------------------------------------------------------------------
// Mappers
// ---------------------------------------------------------------------------

function splitMessages(messages: ChatMessage[]): {
  systemInstruction?: { parts: Array<{ text: string }> };
  contents: GeminiContent[];
} {
  // Filled while walking, so a tool result resolves against the nearest
  // preceding assistant turn: synthetic `call_N` ids restart at 0 in every
  // response, and a whole-history map would hand an earlier result the name
  // of a later call.
  const nameById = new Map<string, string>();

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
        functionResponse: {
          ...(m.toolCallId && isEchoableId(m.toolCallId) ? { id: m.toolCallId } : {}),
          name,
          response: { result: parseJsonOrString(contentToText(m.content)) },
        },
      };
      const last = contents[contents.length - 1];
      if (last && last.role === 'user' && isFunctionResponseParts(last.parts)) {
        last.parts.push(part);
      } else {
        contents.push({ role: 'user', parts: [part] });
      }
      continue;
    }
    if (m.role === 'assistant' && m.toolCalls) {
      for (const tc of m.toolCalls) nameById.set(tc.id, tc.function.name);
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
    const signed = m.toolCalls.some((tc) => storedSignature(tc) !== undefined);
    m.toolCalls.forEach((tc, i) => {
      const part: Record<string, unknown> = {
        functionCall: {
          ...(isEchoableId(tc.id) ? { id: tc.id } : {}),
          name: tc.function.name,
          args: safeJsonParse(tc.function.arguments),
        },
      };
      const signature = storedSignature(tc);
      if (signature !== undefined) {
        part.thoughtSignature = signature;
      } else if (!signed && i === 0) {
        // No call in this turn has a real signature: Gemini checks only the
        // turn's first call, and Google documents this value for that case.
        part.thoughtSignature = SKIP_THOUGHT_SIGNATURE;
      }
      parts.push(part);
    });
  }
  if (parts.length === 0) parts.push({ text: '' });
  return parts;
}

/** The `thoughtSignature` this connector stored on a tool call, if any. Other providers' keys are never read. */
function storedSignature(tc: ToolCall): string | undefined {
  const signature = (tc.providerMetadata?.gemini as { thoughtSignature?: unknown } | undefined)?.thoughtSignature;
  return typeof signature === 'string' && signature.length > 0 ? signature : undefined;
}

function signatureMetadata(p: GeminiPart): Record<string, unknown> | undefined {
  return typeof p.thoughtSignature === 'string' && p.thoughtSignature.length > 0
    ? { gemini: { thoughtSignature: p.thoughtSignature } }
    : undefined;
}

/**
 * Whether an id goes back on `functionCall.id` / `functionResponse.id`: any
 * real id does — Gemini's own, and another provider's, which Gemini accepts
 * as long as the call and its response carry the same one. Only an id this
 * connector made up for a call Gemini returned without one stays off the wire.
 */
function isEchoableId(id: string): boolean {
  return id.length > 0 && !SYNTHETIC_CALL_ID.test(id);
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
    // JSON Schema proper; `responseSchema` takes only Google's OpenAPI subset.
    generationConfig.responseJsonSchema = rf.jsonSchema.schema;
  }
}

function mapFinishReason(fr: string | null | undefined): FinishReason {
  switch (fr) {
    case 'STOP':
      return 'stop';
    case 'MAX_TOKENS':
    case 'CONTINUATION':
      return 'length';
    case 'SAFETY':
    case 'RECITATION':
    case 'BLOCKLIST':
    case 'PROHIBITED_CONTENT':
    case 'SPII':
    case 'IMAGE_SAFETY':
    case 'IMAGE_PROHIBITED_CONTENT':
    case 'IMAGE_RECITATION':
      return 'content_filter';
    // MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, TOO_MANY_TOOL_CALLS,
    // LANGUAGE (an unsupported language, not a safety block), NO_IMAGE,
    // IMAGE_OTHER, OTHER, unspecified: no normalized counterpart (a malformed
    // call is not a usable tool call).
    default:
      return 'unknown';
  }
}

/**
 * Thoughts are billed as output and tool-use prompts as input, so both are
 * folded in; Gemini's own total is the same sum.
 */
function mapUsage(u: GeminiResponse['usageMetadata']): Usage {
  const inputTokens = (u?.promptTokenCount ?? 0) + (u?.toolUsePromptTokenCount ?? 0);
  const outputTokens = (u?.candidatesTokenCount ?? 0) + (u?.thoughtsTokenCount ?? 0);
  return { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens };
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

/** HTTP status for each google.rpc status name a mid-stream error frame can carry. */
const RPC_STATUS_HTTP: Record<string, number> = {
  INVALID_ARGUMENT: 400,
  FAILED_PRECONDITION: 400,
  OUT_OF_RANGE: 400,
  UNAUTHENTICATED: 401,
  PERMISSION_DENIED: 403,
  NOT_FOUND: 404,
  ABORTED: 409,
  ALREADY_EXISTS: 409,
  RESOURCE_EXHAUSTED: 429,
  CANCELLED: 499,
  INTERNAL: 500,
  UNKNOWN: 500,
  DATA_LOSS: 500,
  UNIMPLEMENTED: 501,
  UNAVAILABLE: 503,
  DEADLINE_EXCEEDED: 504,
};

/**
 * The HTTP status a mid-stream Gemini `{"error": …}` frame stands for:
 * `error.code` is the HTTP status; without a usable one, the google.rpc
 * `error.status` name decides; `null` when the frame names neither.
 */
function errorFrameStatus(error: GeminiResponse['error']): number | null {
  const code = error?.code;
  if (typeof code === 'number' && code >= 400 && code <= 599) return code;
  const byName = typeof error?.status === 'string' ? RPC_STATUS_HTTP[error.status] : undefined;
  return byName !== undefined && byName >= 400 && byName <= 599 ? byName : null;
}

function mapStatusToProviderCode(status: number, message: string, body: unknown): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  // 402 is credit/quota exhaustion, which OpenAI (429 insufficient_quota) and
  // Gemini (RESOURCE_EXHAUSTED) already report as rate-limited, and
  // @thinwrap/notifications maps Pusher Beams' 402 the same way.
  if (status === 429 || status === 402) return 'rate_limited';
  if (status >= 500 || status === 408) return 'provider_unavailable';
  if (status >= 400) {
    // A bad key is a 400 INVALID_ARGUMENT whose ErrorInfo reason names it.
    if (status === 400 && hasErrorReason(body, 'API_KEY_INVALID')) return 'auth_failed';
    // Only the overflow message ("The input token count (N) exceeds the maximum
    // number of tokens allowed (M)") counts: a looser match ("context", "too
    // long") would also catch ordinary 400s, and a caller that trims history on
    // this code would then loop forever.
    if (status === 400 && message.toLowerCase().includes('exceeds the maximum number of tokens')) {
      return 'context_length_exceeded';
    }
    // 404, 413, 422, …: the request cannot succeed as sent.
    return 'invalid_request';
  }
  return 'unknown';
}

/** Whether a google.rpc error body carries an `ErrorInfo` detail with this `reason`. */
function hasErrorReason(body: unknown, reason: string): boolean {
  const err = (body as { error?: { details?: unknown } } | null)?.error;
  const details = err?.details;
  return Array.isArray(details) && details.some((d) => (d as { reason?: unknown } | null)?.reason === reason);
}
