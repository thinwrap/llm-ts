import { applyRetryAfter, BaseConnector, parseSSEStream } from '../../base/base.connector';
import type { SSEStreamState } from '../../base/base.connector';
import { ConnectorError } from '../../types';
import type {
  ChatInput,
  ChatMessage,
  ChatResult,
  ChatStreamDelta,
  FinishReason,
  IChatConnector,
  OpenAICompatConfig,
  ProviderCode,
  ResponseFormat,
  ToolCall,
  Usage,
} from '../../types';
import type { OpenAICompatSpec } from './spec';

/**
 * The single connector shared by every first-class OpenAI-compatible provider.
 * It builds the `/chat/completions` request from the normalized `ChatInput`,
 * parses the response (and SSE stream) into the normalized shapes, and maps
 * vendor errors to `ConnectorError`. Provider differences are expressed by the
 * `OpenAICompatSpec`, not by subclassing.
 */
export class OpenAICompatConnector extends BaseConnector implements IChatConnector {
  public readonly id: string;
  private readonly baseUrl: string;
  private readonly spec: OpenAICompatSpec;
  private readonly config: OpenAICompatConfig;

  constructor(spec: OpenAICompatSpec, config: OpenAICompatConfig) {
    super(config.fetch);
    this.spec = spec;
    this.id = spec.id;
    this.config = config;
    const baseUrl = config.baseUrl ?? spec.defaultBaseUrl;
    if (!baseUrl) {
      throw new ConnectorError({
        message: `Provider '${spec.id}' requires an explicit \`baseUrl\` in config`,
        statusCode: null,
        providerCode: 'invalid_request',
      });
    }
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async complete(input: ChatInput): Promise<ChatResult> {
    const { url, body, headers } = this.buildRequest(input, false);
    const response = await this.sendPostJson(url, body, headers, input.signal);
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw this.mapVendorError(response.status, errBody, response.headers);
    }
    const json = this.requireDecodedBody<OpenAIChatResponse>((await response.json().catch(() => null)) as OpenAIChatResponse | null, response.status);
    this.assertCompletionPayload(json, response.status, response.headers);
    return this.parseResult(json, input.model);
  }

  /**
   * A 2xx is not proof of success. Gateways (OpenRouter documents this) answer
   * HTTP 200 when a failure happens after processing started: the body is
   * either only an `error` object with no `choices`, or `choices[0]` carrying
   * `finish_reason: "error"` plus an `error` object beside partial content.
   * Both are raised as in-body errors (see `inBodyError`).
   */
  private assertCompletionPayload(json: OpenAIChatResponse, status: number, headers: Headers): void {
    if (json.error != null) {
      throw this.inBodyError(status, json.error, json, headers);
    }
    const choices = json.choices;
    if (!Array.isArray(choices) || choices.length === 0) {
      throw this.missingPayloadError(status, json, 'no choices');
    }
    const choice = choices[0];
    if (choice?.error != null || choice?.finish_reason === 'error') {
      const error = choice?.error ?? { message: 'Provider reported finish_reason "error"' };
      throw this.inBodyError(status, error, { error }, headers, json);
    }
  }

  async *stream(input: ChatInput): AsyncGenerator<ChatStreamDelta> {
    const { url, body, headers } = this.buildRequest(input, true);
    const response = await this.invokeFetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...headers,
      },
      body: JSON.stringify(body),
      signal: input.signal,
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw this.mapVendorError(response.status, errBody, response.headers);
    }
    this.requireEventStream(response);
    // A complete stream ends on `[DONE]`, and its last choice chunk carries a
    // `finish_reason`; either one proves the answer was not cut off.
    const state: SSEStreamState = { sawDone: false };
    let sawFinish = false;
    for await (const chunk of parseSSEStream(response, state)) {
      for (const delta of this.mapStreamChunk(chunk as OpenAIStreamChunk, response.status)) {
        if (delta.finishReason !== undefined) sawFinish = true;
        yield delta;
      }
    }
    if (!state.sawDone && !sawFinish) throw this.streamEndedEarlyError();
  }

  private buildRequest(
    input: ChatInput,
    stream: boolean,
  ): { url: string; body: Record<string, unknown>; headers: Record<string, string> } {
    const body: Record<string, unknown> = {
      model: input.model,
      messages: input.messages.map(mapMessage),
    };

    const toolsAllowed = this.spec.supportsTools !== false;
    if (toolsAllowed && input.tools && input.tools.length > 0) {
      body.tools = input.tools.map((t) => ({
        type: 'function',
        function: {
          name: t.function.name,
          ...(t.function.description ? { description: t.function.description } : {}),
          ...(t.function.parameters ? { parameters: t.function.parameters } : {}),
        },
      }));
    }
    if (toolsAllowed && input.toolChoice !== undefined) body.tool_choice = input.toolChoice;
    if (input.responseFormat) body.response_format = mapResponseFormat(input.responseFormat);
    if (input.temperature !== undefined) body.temperature = input.temperature;
    if (input.topP !== undefined) body.top_p = input.topP;
    if (input.maxOutputTokens !== undefined) {
      body[this.spec.maxTokensField ?? 'max_tokens'] = input.maxOutputTokens;
    }
    if (input.stop !== undefined) body.stop = input.stop;
    if (input.reasoning) body.reasoning_effort = input.reasoning.effort;

    if (stream) {
      body.stream = true;
      if (!this.spec.strictParams) body.stream_options = { include_usage: true };
    }

    // Passthrough body is merged last and overrides connector-built fields.
    if (input._passthrough?.body) Object.assign(body, input._passthrough.body);

    const headers: Record<string, string> = {
      ...this.spec.buildAuthHeaders(this.config),
      ...this.config.headers,
      ...input._passthrough?.headers,
    };

    let url = `${this.baseUrl}/chat/completions`;
    const query = input._passthrough?.query;
    if (query && Object.keys(query).length > 0) {
      url += `?${new URLSearchParams(query).toString()}`;
    }
    return { url, body, headers };
  }

  private parseResult(json: OpenAIChatResponse | null, requestedModel: string): ChatResult {
    const choice = json?.choices?.[0];
    const msg = choice?.message;
    const toolCalls = msg?.tool_calls?.map(mapToolCall);
    return {
      message: {
        role: 'assistant',
        content: contentText(msg?.content),
        ...(toolCalls && toolCalls.length > 0 ? { toolCalls } : {}),
      },
      finishReason: mapFinishReason(choice?.finish_reason),
      usage: mapUsage(json?.usage),
      model: json?.model ?? requestedModel,
      raw: json,
    };
  }

  private mapStreamChunk(chunk: OpenAIStreamChunk, httpStatus: number): ChatStreamDelta[] {
    // A provider/gateway failure emitted mid-stream arrives as an `{"error": …}`
    // data frame (no `choices`). Surface it as a ConnectorError instead of
    // silently truncating the stream into a finish-less "success".
    if (chunk.error != null) {
      throw this.inBodyError(httpStatus, chunk.error, chunk, new Headers());
    }

    const deltas: ChatStreamDelta[] = [];
    const choice = chunk.choices?.[0];
    const delta = choice?.delta;
    const contentDelta = contentText(delta?.content);
    if (contentDelta !== null && contentDelta.length > 0) {
      deltas.push({ contentDelta, raw: chunk });
    }
    // Parallel tool-call fragments can arrive in a single chunk — emit one
    // toolCallDelta per entry (not just `tool_calls[0]`).
    for (const tc of delta?.tool_calls ?? []) {
      deltas.push({
        toolCallDelta: {
          index: tc.index ?? 0,
          ...(tc.id ? { id: tc.id } : {}),
          ...(tc.function?.name ? { functionName: tc.function.name } : {}),
          ...(tc.function?.arguments ? { argumentsDelta: tc.function.arguments } : {}),
        },
        raw: chunk,
      });
    }

    const trailer: ChatStreamDelta = { raw: chunk };
    let hasTrailer = false;
    if (choice?.finish_reason) {
      trailer.finishReason = mapFinishReason(choice.finish_reason);
      hasTrailer = true;
    }
    const usageRaw =
      chunk.usage ??
      (this.spec.streamUsagePath === 'x_groq' ? chunk.x_groq?.usage : undefined);
    if (usageRaw) {
      trailer.usage = mapUsage(usageRaw);
      hasTrailer = true;
    }
    if (hasTrailer) deltas.push(trailer);

    return deltas;
  }

  /**
   * An error delivered inside a 2xx body or a stream frame. It carries the HTTP
   * status it arrived with — the response really was a 2xx — while the code
   * comes from the embedded error: a numeric `code` classified as that HTTP
   * status, a string `code` / `type` by its name, and `unknown` with neither.
   * `raw` overrides `cause.raw` when `body` is only the error slice.
   */
  private inBodyError(httpStatus: number, error: unknown, body: unknown, headers: Headers, raw: unknown = body): ConnectorError {
    const effective = embeddedErrorStatus(error);
    const message = extractErrorMessage(body) ?? `HTTP ${httpStatus}`;
    const cause: Record<string, unknown> = { raw: raw ?? null };
    applyRetryAfter(cause, headers);
    return new ConnectorError({
      message,
      statusCode: httpStatus,
      providerCode: effective === null ? 'unknown' : mapStatusToProviderCode(effective, body),
      providerMessage: message,
      cause,
    });
  }

  /** `raw` overrides `cause.raw` when `body` is only the error slice of a larger response. */
  private mapVendorError(status: number, body: unknown, headers: Headers, raw: unknown = body): ConnectorError {
    const message = extractErrorMessage(body) ?? `HTTP ${status}`;
    const providerCode = mapStatusToProviderCode(status, body);
    const cause: Record<string, unknown> = { raw: raw ?? null };
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
// Vendor wire shapes (minimal — only the fields the connector reads)
// ---------------------------------------------------------------------------

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

/**
 * Mistral reasoning models return `content` as a list of chunks instead of a
 * string: `{"type":"thinking","thinking":[…]}` and `{"type":"text","text":"…"}`.
 */
type OpenAIContent = string | null | Array<{ type?: string; text?: unknown }>;

interface OpenAIChatResponse {
  model?: string;
  usage?: OpenAIUsage | null;
  /** Error delivered on a 2xx (OpenRouter: failure after processing started). */
  error?: unknown;
  choices?: Array<{
    /** Per-choice error beside `finish_reason: "error"` and any partial content. */
    error?: unknown;
    message?: {
      content?: OpenAIContent;
      tool_calls?: Array<{
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string | null;
  }>;
}

interface OpenAIStreamChunk {
  usage?: OpenAIUsage | null;
  x_groq?: { usage?: OpenAIUsage | null };
  /** Mid-stream error frame (gateway/upstream failure), e.g. `{"error": {"message": …}}`. */
  error?: unknown;
  choices?: Array<{
    delta?: {
      content?: OpenAIContent;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string | null;
  }>;
}

// ---------------------------------------------------------------------------
// Module-private mappers
// ---------------------------------------------------------------------------

function mapMessage(m: ChatMessage): Record<string, unknown> {
  const out: Record<string, unknown> = { role: m.role, content: mapContent(m.content) };
  if (m.name) out.name = m.name;
  if (m.toolCalls && m.toolCalls.length > 0) {
    out.tool_calls = m.toolCalls.map((tc) => ({
      id: tc.id,
      type: 'function',
      function: { name: tc.function.name, arguments: tc.function.arguments },
    }));
  }
  if (m.toolCallId) out.tool_call_id = m.toolCallId;
  return out;
}

function mapContent(content: ChatMessage['content']): unknown {
  if (content === null) return null;
  if (typeof content === 'string') return content;
  return content.map((p) =>
    p.type === 'text'
      ? { type: 'text', text: p.text }
      : { type: 'image_url', image_url: { url: `data:${p.mediaType};base64,${p.base64}` } },
  );
}

/**
 * Answer text of a message/delta `content`. For the chunk-list shape only
 * `type: "text"` chunks are answer text; `thinking` chunks are reasoning and
 * stay reachable via `raw`. A list with no text is no content.
 */
function contentText(content: OpenAIContent | undefined): string | null {
  if (!Array.isArray(content)) return content ?? null;
  let text = '';
  for (const chunk of content) {
    if (chunk?.type === 'text' && typeof chunk.text === 'string') text += chunk.text;
  }
  return text.length > 0 ? text : null;
}

function mapToolCall(tc: {
  id?: string;
  function?: { name?: string; arguments?: string };
}): ToolCall {
  return {
    id: tc.id ?? '',
    type: 'function',
    function: { name: tc.function?.name ?? '', arguments: tc.function?.arguments ?? '' },
  };
}

function mapFinishReason(fr: string | null | undefined): FinishReason {
  switch (fr) {
    case 'stop':
    case 'eos': // Together AI reports natural end-of-sequence as "eos"
      return 'stop';
    case 'length':
      return 'length';
    case 'tool_calls':
    case 'function_call':
      return 'tool_calls';
    case 'content_filter':
      return 'content_filter';
    default:
      return 'unknown';
  }
}

function mapUsage(u: OpenAIUsage | null | undefined): Usage {
  const inputTokens = u?.prompt_tokens ?? 0;
  const outputTokens = u?.completion_tokens ?? 0;
  const totalTokens = u?.total_tokens ?? inputTokens + outputTokens;
  return { inputTokens, outputTokens, totalTokens };
}

function mapResponseFormat(rf: ResponseFormat): Record<string, unknown> {
  if (rf.type === 'json_schema') {
    return {
      type: 'json_schema',
      json_schema: { name: rf.jsonSchema.name, schema: rf.jsonSchema.schema },
    };
  }
  return { type: rf.type };
}

function extractErrorMessage(body: unknown): string | null {
  if (typeof body === 'string') return body;
  if (!body || typeof body !== 'object') return null;
  const b = body as Record<string, unknown>;
  if (typeof b.error === 'string') return b.error;
  if (b.error && typeof b.error === 'object') {
    const em = (b.error as Record<string, unknown>).message;
    if (typeof em === 'string') return em;
  }
  if (typeof b.message === 'string') return b.message;
  const errs = b.errors;
  if (Array.isArray(errs) && errs[0] && typeof errs[0] === 'object') {
    const m = (errs[0] as Record<string, unknown>).message;
    if (typeof m === 'string') return m;
  }
  return null;
}

/** HTTP status for the OpenAI-style error `type` / `code` names an embedded error can carry instead of a number. */
const ERROR_NAME_STATUS: Record<string, number> = {
  invalid_request_error: 400,
  context_length_exceeded: 400,
  authentication_error: 401,
  invalid_api_key: 401,
  permission_error: 403,
  not_found_error: 404,
  rate_limit_error: 429,
  rate_limit_exceeded: 429,
  insufficient_quota: 429,
  server_error: 500,
  api_error: 500,
  overloaded_error: 529,
};

/**
 * Effective HTTP status for an `error` object delivered inside a 2xx body or a
 * mid-stream frame: its `code` when that is an HTTP error status (a number, or
 * a numeric string), else the status its `code` / `type` name stands for, else
 * `null`. Shared with the embeddings connector so both classify the same body
 * the same way.
 */
export function embeddedErrorStatus(error: unknown): number | null {
  if (error && typeof error === 'object') {
    const { code, type } = error as Record<string, unknown>;
    const numeric = typeof code === 'string' && /^\d{3}$/.test(code) ? Number(code) : code;
    if (typeof numeric === 'number' && Number.isInteger(numeric) && numeric >= 400 && numeric <= 599) return numeric;
    for (const name of [code, type]) {
      if (typeof name === 'string' && ERROR_NAME_STATUS[name] !== undefined) return ERROR_NAME_STATUS[name]!;
    }
  }
  return null;
}

function mapStatusToProviderCode(status: number, body: unknown): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  // 402 is credit/quota exhaustion, which OpenAI (429 insufficient_quota) and
  // Gemini (RESOURCE_EXHAUSTED) already report as rate-limited, and
  // @thinwrap/notifications maps Pusher Beams' 402 the same way.
  if (status === 429 || status === 402) return 'rate_limited';
  if (status >= 500 || status === 408) return 'provider_unavailable';
  if (status >= 400) {
    const msg = (extractErrorMessage(body) ?? '').toLowerCase();
    if (
      errorCode(body) === 'context_length_exceeded' ||
      msg.includes('context length') ||
      msg.includes('maximum context') ||
      msg.includes('context_length')
    ) {
      return 'context_length_exceeded';
    }
    if (
      (msg.includes('content') && msg.includes('filter')) ||
      msg.includes('content_policy') ||
      msg.includes('content policy') ||
      msg.includes('moderation')
    ) {
      return 'content_filtered';
    }
    // 404, 413, 422, …: the request cannot succeed as sent.
    return 'invalid_request';
  }
  return 'unknown';
}

function errorCode(body: unknown): unknown {
  const error = (body as { error?: unknown } | null)?.error;
  return error && typeof error === 'object' ? (error as Record<string, unknown>).code : undefined;
}
