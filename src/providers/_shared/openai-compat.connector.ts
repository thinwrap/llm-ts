import { BaseConnector, parseSSEStream } from '../../base/base.connector';
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
    return this.parseResult(json, input.model);
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
    for await (const chunk of parseSSEStream(response)) {
      for (const delta of this.mapStreamChunk(chunk as OpenAIStreamChunk)) yield delta;
    }
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
        content: msg?.content ?? null,
        ...(toolCalls && toolCalls.length > 0 ? { toolCalls } : {}),
      },
      finishReason: mapFinishReason(choice?.finish_reason),
      usage: mapUsage(json?.usage),
      model: json?.model ?? requestedModel,
      raw: json,
    };
  }

  private mapStreamChunk(chunk: OpenAIStreamChunk): ChatStreamDelta[] {
    // A provider/gateway failure emitted mid-stream arrives as an `{"error": …}`
    // data frame (no `choices`). Surface it as a ConnectorError instead of
    // silently truncating the stream into a finish-less "success".
    if (chunk.error != null) {
      throw this.mapVendorError(streamErrorStatus(chunk.error), chunk, new Headers());
    }

    const deltas: ChatStreamDelta[] = [];
    const choice = chunk.choices?.[0];
    const delta = choice?.delta;
    if (typeof delta?.content === 'string' && delta.content.length > 0) {
      deltas.push({ contentDelta: delta.content, raw: chunk });
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

  private mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
    const message = extractErrorMessage(body) ?? `HTTP ${status}`;
    const providerCode = mapStatusToProviderCode(status, body);
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
// Vendor wire shapes (minimal — only the fields the connector reads)
// ---------------------------------------------------------------------------

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

interface OpenAIChatResponse {
  model?: string;
  usage?: OpenAIUsage | null;
  choices?: Array<{
    message?: {
      content?: string | null;
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
      content?: string | null;
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

/** Best-effort HTTP status for a mid-stream `{"error": …}` frame (some gateways carry a numeric `code`). */
function streamErrorStatus(error: unknown): number {
  if (error && typeof error === 'object') {
    const code = (error as Record<string, unknown>).code;
    if (typeof code === 'number' && code >= 400 && code <= 599) return code;
  }
  return 500;
}

function mapStatusToProviderCode(status: number, body: unknown): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'provider_unavailable';
  if (status === 400 || status === 404 || status === 422) {
    const msg = (extractErrorMessage(body) ?? '').toLowerCase();
    if (msg.includes('context length') || msg.includes('maximum context') || msg.includes('context_length')) {
      return 'context_length_exceeded';
    }
    if (
      msg.includes('content_filter') ||
      msg.includes('content filter') ||
      msg.includes('content_policy') ||
      msg.includes('content policy') ||
      msg.includes('moderation')
    ) {
      return 'content_filtered';
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
