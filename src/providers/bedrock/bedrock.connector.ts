import { createHash, createHmac } from 'node:crypto';
import { applyRetryAfter, BaseConnector } from '../../base/base.connector';
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
import type { BedrockConfig } from './bedrock.config';

const DEFAULT_MAX_TOKENS = 4096;
const SERVICE = 'bedrock';

/**
 * An AWS region is a DNS label run (`us-east-1`, `eu-central-2`). It is spliced
 * into the default host, so anything else (`x.evil.example#`) could steer the
 * signed request — session token included — to another host.
 */
const REGION_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/**
 * Native adapter for AWS Bedrock's unified **Converse API**
 * (`POST /model/{modelId}/converse`). Requests are signed with hand-rolled AWS
 * SigV4 on `node:crypto` (zero third-party deps — same approach as the
 * notifications SES/SNS connectors). Converse is structurally different from
 * OpenAI (top-level `system[]`, typed content blocks, `inferenceConfig`,
 * `toolConfig`, tool results in a user turn), so this is a full translation
 * layer emitting the identical normalized `ChatResult` / `ChatStreamDelta` /
 * `ConnectorError`.
 *
 * Streaming NOTE: Bedrock's `converse-stream` uses AWS's binary event-stream
 * framing (`application/vnd.amazon.eventstream`), not SSE. This connector does
 * NOT parse that binary protocol — `stream()` makes a single non-incremental
 * Converse call and yields the full result as deltas.
 *
 * Not normalized here (passthrough/raw, never emulated): reasoning control
 * (model-dependent on Bedrock — pass `additionalModelRequestFields` via
 * `_passthrough.body`), `responseFormat`, prompt caching (`cachePoint` blocks
 * via `_passthrough`), and `top_k`.
 */
export class BedrockConnector extends BaseConnector implements IChatConnector {
  public readonly id = 'bedrock';
  private readonly config: BedrockConfig;
  private readonly origin: string;
  private readonly host: string;
  /** The `baseUrl` path prefix (e.g. a gateway's `/bedrock`), `''` for none, as signed. */
  private readonly canonicalBasePath: string;

  constructor(config: BedrockConfig) {
    super(config.fetch);
    if (typeof config.region !== 'string' || !REGION_RE.test(config.region)) {
      throw new ConnectorError({
        message: 'Bedrock: `region` must be an AWS region name such as `us-east-1`',
        statusCode: null,
        providerCode: 'invalid_request',
      });
    }
    this.config = config;
    this.origin = (config.baseUrl ?? `https://bedrock-runtime.${config.region}.amazonaws.com`).replace(
      /\/+$/,
      '',
    );
    try {
      const origin = new URL(this.origin);
      this.host = origin.host;
      // A gateway `baseUrl` with a path prefix sends `<prefix>/model/…`, and the
      // signature must cover that full path; each segment is encoded once more,
      // like the model id below.
      this.canonicalBasePath = origin.pathname
        .replace(/\/+$/, '')
        .split('/')
        .map((segment) => encodeURIComponent(segment))
        .join('/');
    } catch {
      throw new ConnectorError({
        message: `Bedrock: invalid \`baseUrl\`/\`region\` — could not derive a host from '${this.origin}'`,
        statusCode: null,
        providerCode: 'invalid_request',
      });
    }
  }

  async complete(input: ChatInput): Promise<ChatResult> {
    const serialized = JSON.stringify(this.buildRequest(input));
    // The wire path is single-encoded, but the SigV4 canonical URI must be
    // DOUBLE-encoded: every non-S3 AWS service URI-encodes the received path a
    // second time before recomputing the signature, so a model id containing `:`
    // (all on-demand IDs like `…-v2:0`, cross-region profiles `us.…-v2:0`, ARNs)
    // must be signed as `%253A` while `%3A` travels on the wire — otherwise the
    // server's signature never matches ours → 403 SignatureDoesNotMatch. For a
    // colon-free id the two are identical, so nothing regresses.
    const wirePath = `/model/${encodeURIComponent(input.model)}/converse`;
    const canonicalPath = `${this.canonicalBasePath}/model/${encodeURIComponent(encodeURIComponent(input.model))}/converse`;
    // `_passthrough.query` must be folded into the SigV4 canonical query BEFORE
    // signing (the query string is part of the signed canonical request); the
    // same canonical form then travels on the wire, so the server's recomputed
    // signature matches. Appending it post-signing would 403 SignatureDoesNotMatch.
    const canonicalQuery = buildCanonicalQuery(input._passthrough?.query);
    const headers = this.buildSignedHeaders('POST', canonicalPath, serialized, input, canonicalQuery);
    const wireUrl = `${this.origin}${wirePath}${canonicalQuery ? `?${canonicalQuery}` : ''}`;
    const response = await this.invokeFetch(wireUrl, {
      method: 'POST',
      headers,
      body: serialized,
      signal: input.signal,
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw this.mapVendorError(response.status, errBody, response.headers);
    }
    const json = this.requireDecodedBody<ConverseResponse>((await response.json().catch(() => null)) as ConverseResponse | null, response.status);
    // Converse always returns the `output` union; without it there is no message.
    if (json.output == null || typeof json.output !== 'object') {
      throw this.missingPayloadError(response.status, json, 'no output');
    }
    return this.parseResult(json, input.model);
  }

  /**
   * Non-incremental: see the class NOTE — Bedrock streaming is a binary
   * event-stream, not SSE, so this issues one Converse call and yields the
   * result as a content delta + tool-call deltas + a terminal finish/usage delta.
   */
  async *stream(input: ChatInput): AsyncGenerator<ChatStreamDelta> {
    const result = await this.complete(input);
    if (result.message.content) yield { contentDelta: result.message.content };
    if (result.message.toolCalls) {
      let index = 0;
      for (const tc of result.message.toolCalls) {
        yield {
          toolCallDelta: {
            index,
            id: tc.id,
            functionName: tc.function.name,
            argumentsDelta: tc.function.arguments,
          },
        };
        index++;
      }
    }
    yield { finishReason: result.finishReason, usage: result.usage, raw: result.raw };
  }

  private buildSignedHeaders(
    method: string,
    canonicalPath: string,
    serializedBody: string,
    input: ChatInput,
    canonicalQuery?: string,
  ): Record<string, string> {
    // Merged case-insensitively: a config/passthrough `content-type` beside the
    // default `Content-Type` would otherwise reach fetch as two entries, which
    // it joins into one comma-separated value that the signature never saw →
    // SignatureDoesNotMatch. Later sources win (default < config < passthrough).
    const extra: Record<string, string> = { 'Content-Type': 'application/json' };
    for (const source of [this.config.headers, input._passthrough?.headers]) {
      for (const [name, value] of Object.entries(source ?? {})) setHeaderCI(extra, name, value);
    }
    const signed = signAwsRequest({
      method,
      path: canonicalPath,
      service: SERVICE,
      region: this.config.region,
      host: this.host,
      accessKeyId: this.config.accessKeyId,
      secretAccessKey: this.config.secretAccessKey,
      sessionToken: this.config.sessionToken,
      serializedBody,
      additionalSignedHeaders: extra,
      isoTimestamp: isoBasicTimestamp(),
      canonicalQuery,
    });
    const headers = { ...extra };
    for (const [name, value] of Object.entries(signed)) setHeaderCI(headers, name, value);
    return headers;
  }

  private buildRequest(input: ChatInput): Record<string, unknown> {
    const { system, messages } = splitMessages(input.messages);

    const body: Record<string, unknown> = { messages };
    if (system.length > 0) body.system = system;

    const inferenceConfig: Record<string, unknown> = {
      maxTokens: input.maxOutputTokens ?? this.config.defaultMaxTokens ?? DEFAULT_MAX_TOKENS,
    };
    if (input.temperature !== undefined) inferenceConfig.temperature = input.temperature;
    if (input.topP !== undefined) inferenceConfig.topP = input.topP;
    if (input.stop !== undefined) {
      inferenceConfig.stopSequences = Array.isArray(input.stop) ? input.stop : [input.stop];
    }
    body.inferenceConfig = inferenceConfig;

    // toolChoice 'none' has no Converse counterpart, so the tools are left out
    // entirely: offering them would let the model call one.
    if (input.tools && input.tools.length > 0 && input.toolChoice !== 'none') {
      const toolConfig: Record<string, unknown> = {
        tools: input.tools.map((t) => ({
          toolSpec: {
            name: t.function.name,
            ...(t.function.description ? { description: t.function.description } : {}),
            inputSchema: { json: t.function.parameters ?? { type: 'object', properties: {} } },
          },
        })),
      };
      const tc = mapToolChoice(input.toolChoice);
      if (tc) toolConfig.toolChoice = tc;
      body.toolConfig = toolConfig;
    }
    // Reasoning control is model-dependent on Bedrock, so it is not mapped;
    // pass `additionalModelRequestFields` via `_passthrough.body`.

    if (input._passthrough?.body) Object.assign(body, input._passthrough.body);
    return body;
  }

  private parseResult(json: ConverseResponse | null, requestedModel: string): ChatResult {
    const blocks = json?.output?.message?.content ?? [];
    let text = '';
    const toolCalls: ToolCall[] = [];
    for (const b of blocks) {
      if (typeof b.text === 'string') {
        text += b.text;
      } else if (b.toolUse) {
        toolCalls.push({
          id: b.toolUse.toolUseId ?? '',
          type: 'function',
          function: { name: b.toolUse.name ?? '', arguments: JSON.stringify(b.toolUse.input ?? {}) },
        });
      }
    }
    // Cache reads and writes are billed input but reported beside `inputTokens`.
    const usage = json?.usage;
    const inputTokens =
      (usage?.inputTokens ?? 0) + (usage?.cacheReadInputTokens ?? 0) + (usage?.cacheWriteInputTokens ?? 0);
    const outputTokens = usage?.outputTokens ?? 0;
    return {
      message: {
        role: 'assistant',
        content: text.length > 0 ? text : null,
        ...(toolCalls.length > 0 ? { toolCalls } : {}),
      },
      finishReason: mapStopReason(json?.stopReason),
      usage: { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens },
      model: requestedModel,
      raw: json,
    };
  }

  private mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
    const message = extractBedrockError(body) ?? `HTTP ${status}`;
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
// Converse wire shapes (minimal)
// ---------------------------------------------------------------------------

interface ConverseResponse {
  stopReason?: string | null;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    totalTokens?: number;
    cacheReadInputTokens?: number;
    cacheWriteInputTokens?: number;
  };
  output?: {
    message?: {
      content?: Array<{ text?: string; toolUse?: { toolUseId?: string; name?: string; input?: unknown } }>;
    };
  };
}

type ConverseMessage = { role: string; content: unknown[] };

// ---------------------------------------------------------------------------
// Hand-rolled AWS SigV4 (node:crypto; adapted from the notifications SES signer)
// ---------------------------------------------------------------------------

const UNSIGNABLE_HEADERS = new Set([
  'authorization',
  'cache-control',
  'connection',
  'content-length',
  'expect',
  'from',
  'keep-alive',
  'max-forwards',
  'pragma',
  'proxy-authenticate',
  'proxy-authorization',
  'referer',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
  'user-agent',
  'x-amzn-trace-id',
]);

/**
 * Hand-rolled AWS SigV4 signer (exported for white-box testing of the
 * canonical-path encoding; NOT re-exported from the package barrel).
 */
export function signAwsRequest(opts: {
  method: string;
  path: string;
  service: string;
  region: string;
  host: string;
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
  serializedBody: string;
  additionalSignedHeaders: Record<string, string>;
  isoTimestamp: string;
  canonicalQuery?: string;
}): Record<string, string> {
  const canonicalQuery = opts.canonicalQuery ?? '';
  const dateStamp = opts.isoTimestamp.slice(0, 8);
  const hashedPayload = createHash('sha256').update(opts.serializedBody).digest('hex');

  const canonicalHeaders: Record<string, string> = {
    ...opts.additionalSignedHeaders,
    Host: opts.host,
    'X-Amz-Date': opts.isoTimestamp,
    'X-Amz-Content-Sha256': hashedPayload,
  };
  if (opts.sessionToken) canonicalHeaders['X-Amz-Security-Token'] = opts.sessionToken;

  const normalized: Record<string, string> = {};
  for (const [name, value] of Object.entries(canonicalHeaders)) {
    const lowerName = name.toLowerCase();
    if (UNSIGNABLE_HEADERS.has(lowerName)) continue;
    normalized[lowerName] = String(value).trim().replace(/\s+/g, ' ');
  }
  const sortedNames = Object.keys(normalized).sort();

  const canonicalHeadersString = sortedNames.map((name) => `${name}:${normalized[name]}\n`).join('');
  const signedHeadersList = sortedNames.join(';');

  const canonicalRequest = `${opts.method}\n${opts.path}\n${canonicalQuery}\n${canonicalHeadersString}\n${signedHeadersList}\n${hashedPayload}`;

  const credentialScope = `${dateStamp}/${opts.region}/${opts.service}/aws4_request`;
  const stringToSign =
    `AWS4-HMAC-SHA256\n${opts.isoTimestamp}\n${credentialScope}\n` +
    createHash('sha256').update(canonicalRequest).digest('hex');

  const hmac = (key: string | Buffer, data: string): Buffer =>
    createHmac('sha256', key).update(data).digest();
  const kDate = hmac('AWS4' + opts.secretAccessKey, dateStamp);
  const kRegion = hmac(kDate, opts.region);
  const kService = hmac(kRegion, opts.service);
  const kSigning = hmac(kService, 'aws4_request');
  const signature = createHmac('sha256', kSigning).update(stringToSign).digest('hex');

  const authorization = `AWS4-HMAC-SHA256 Credential=${opts.accessKeyId}/${credentialScope}, SignedHeaders=${signedHeadersList}, Signature=${signature}`;

  const out: Record<string, string> = {
    Host: opts.host,
    'X-Amz-Date': opts.isoTimestamp,
    'X-Amz-Content-Sha256': hashedPayload,
    Authorization: authorization,
  };
  if (opts.sessionToken) out['X-Amz-Security-Token'] = opts.sessionToken;
  return out;
}

/** Set `headers[name] = value`, first dropping any entry whose name differs only in case. */
function setHeaderCI(headers: Record<string, string>, name: string, value: string): void {
  const lower = name.toLowerCase();
  for (const existing of Object.keys(headers)) {
    if (existing.toLowerCase() === lower) delete headers[existing];
  }
  headers[name] = value;
}

function isoBasicTimestamp(): string {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
}

/**
 * RFC-3986 percent-encode a query key/value for the SigV4 canonical query
 * string. `encodeURIComponent` leaves `!'()*` unescaped, but SigV4's unreserved
 * set is exactly `A-Za-z0-9-_.~`, so those must also be escaped or the
 * signature won't match.
 */
function awsUriEncodeComponent(s: string): string {
  return encodeURIComponent(s).replace(
    /[!'()*]/g,
    (c) => '%' + c.charCodeAt(0).toString(16).toUpperCase(),
  );
}

/**
 * Build the SigV4 canonical query string from `_passthrough.query`: encode each
 * key/value, then sort by encoded key (then value). The same string is both
 * signed and sent on the wire. Returns '' when there are no params.
 */
function buildCanonicalQuery(query?: Record<string, string>): string {
  if (!query) return '';
  const pairs = Object.entries(query).map(
    ([k, v]) => [awsUriEncodeComponent(k), awsUriEncodeComponent(v)] as const,
  );
  if (pairs.length === 0) return '';
  pairs.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
  return pairs.map(([k, v]) => `${k}=${v}`).join('&');
}

// ---------------------------------------------------------------------------
// Converse mappers
// ---------------------------------------------------------------------------

function splitMessages(messages: ChatMessage[]): {
  system: Array<{ text: string }>;
  messages: ConverseMessage[];
} {
  const system: Array<{ text: string }> = [];
  const out: ConverseMessage[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const t = contentToText(m.content);
      if (t) system.push({ text: t });
      continue;
    }
    if (m.role === 'tool') {
      const block = {
        toolResult: { toolUseId: m.toolCallId ?? '', content: [{ text: contentToText(m.content) }] },
      };
      const last = out[out.length - 1];
      if (last && last.role === 'user' && isToolResultContent(last.content)) {
        last.content.push(block);
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
    if (m.content.length > 0) blocks.push({ text: m.content });
  } else if (Array.isArray(m.content)) {
    for (const p of m.content) {
      if (p.type === 'text') {
        // Converse rejects empty text blocks; an empty part carries nothing.
        if (p.text.length > 0) blocks.push({ text: p.text });
      } else {
        blocks.push({ image: { format: imageFormat(p.mediaType), source: { bytes: p.base64 } } });
      }
    }
  }
  if (m.role === 'assistant' && m.toolCalls) {
    for (const tc of m.toolCalls) {
      blocks.push({
        toolUse: { toolUseId: tc.id, name: tc.function.name, input: safeJsonParse(tc.function.arguments) },
      });
    }
  }
  if (blocks.length === 0) blocks.push({ text: '' });
  return blocks;
}

function isToolResultContent(content: unknown[]): boolean {
  return (
    content.length > 0 &&
    typeof content[0] === 'object' &&
    content[0] !== null &&
    'toolResult' in (content[0] as object)
  );
}

function mapToolChoice(tc: ToolChoice | undefined): Record<string, unknown> | null {
  if (tc === undefined) return null;
  if (tc === 'auto') return { auto: {} };
  if (tc === 'required') return { any: {} };
  if (tc === 'none') return null; // unreachable: buildRequest omits toolConfig for 'none'
  if (typeof tc === 'object' && tc.type === 'function') return { tool: { name: tc.function.name } };
  return null;
}

function mapStopReason(sr: string | null | undefined): FinishReason {
  switch (sr) {
    case 'end_turn':
    case 'stop_sequence':
      return 'stop';
    case 'max_tokens':
    case 'model_context_window_exceeded':
      return 'length';
    case 'tool_use':
      return 'tool_calls';
    case 'content_filtered':
    case 'guardrail_intervened':
      return 'content_filter';
    // malformed_model_output / malformed_tool_use: no usable output to name.
    default:
      return 'unknown';
  }
}

function imageFormat(mediaType: string): string {
  const m = mediaType.toLowerCase();
  if (m.includes('png')) return 'png';
  if (m.includes('jpeg') || m.includes('jpg')) return 'jpeg';
  if (m.includes('gif')) return 'gif';
  if (m.includes('webp')) return 'webp';
  return 'png';
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return {};
  }
}

function extractBedrockError(body: unknown): string | null {
  if (typeof body === 'string') return body;
  if (!body || typeof body !== 'object') return null;
  const b = body as Record<string, unknown>;
  if (typeof b.message === 'string') return b.message;
  if (typeof b.Message === 'string') return b.Message;
  if (typeof b.__type === 'string') return b.__type;
  return null;
}

function mapStatusToProviderCode(status: number, message: string): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  // 402 is credit/quota exhaustion, which OpenAI (429 insufficient_quota) and
  // Gemini (RESOURCE_EXHAUSTED) already report as rate-limited, and
  // @thinwrap/notifications maps Pusher Beams' 402 the same way.
  if (status === 429 || status === 402) return 'rate_limited';
  // 408 ModelTimeoutException and 424 ModelErrorException are failures on the
  // model side of the call, retryable like a 5xx.
  if (status >= 500 || status === 408 || status === 424) return 'provider_unavailable';
  if (status >= 400) {
    const msg = message.toLowerCase();
    if (msg.includes('throttl')) return 'rate_limited';
    // Only overflow phrasings count: a looser match ("token", "max", "context")
    // would turn ordinary ValidationExceptions (e.g. "maxTokens must be …")
    // into overflows, and a caller that trims history on this code would loop.
    if (status === 400 && CONTEXT_OVERFLOW_PHRASES.some((p) => msg.includes(p))) {
      return 'context_length_exceeded';
    }
    return 'invalid_request';
  }
  return 'unknown';
}

/**
 * Overflow wording of the ValidationException across Bedrock's model families;
 * Bedrock relays the model provider's text ("Input is too long for requested
 * model", Anthropic's "prompt is too long" / "… exceed context limit",
 * Llama/Mistral's "maximum context length").
 */
const CONTEXT_OVERFLOW_PHRASES = [
  'input is too long',
  'prompt is too long',
  'too many input tokens',
  'maximum context length',
  'exceed context limit',
];
