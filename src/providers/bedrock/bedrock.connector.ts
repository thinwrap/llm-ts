import { createHash, createHmac } from 'node:crypto';
import { BaseConnector } from '../../base/base.connector';
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
 * Native adapter for AWS Bedrock's unified **Converse API**
 * (`POST /model/{modelId}/converse`). Requests are signed with hand-rolled AWS
 * SigV4 on `node:crypto` (zero third-party deps — same approach as the
 * notifications SES/SNS connectors). Converse is structurally different from
 * OpenAI (top-level `system[]`, typed content blocks, `inferenceConfig`,
 * `toolConfig`, tool results in a user turn), so this is a full translation
 * layer emitting the identical normalized `ChatResult` / `ChatStreamDelta` /
 * `ConnectorError`.
 *
 * v1 streaming NOTE: Bedrock's `converse-stream` uses AWS's binary
 * event-stream framing (`application/vnd.amazon.eventstream`), not SSE. v1 does
 * NOT parse that binary protocol — `stream()` falls back to a single
 * non-incremental Converse call and yields the full result as deltas. True
 * incremental streaming (a binary event-stream parser) is a documented
 * follow-up.
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

  constructor(config: BedrockConfig) {
    super(config.fetch);
    this.config = config;
    this.origin = (config.baseUrl ?? `https://bedrock-runtime.${config.region}.amazonaws.com`).replace(
      /\/+$/,
      '',
    );
    try {
      this.host = new URL(this.origin).host;
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
    const canonicalPath = `/model/${encodeURIComponent(encodeURIComponent(input.model))}/converse`;
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
    return this.parseResult(json, input.model);
  }

  /**
   * v1 fallback: non-incremental. See the class NOTE — Bedrock streaming is a
   * binary event-stream, not SSE; v1 issues one Converse call and yields the
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
    const extra: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.headers,
      ...input._passthrough?.headers,
    };
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
    return { ...extra, ...signed };
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

    if (input.tools && input.tools.length > 0) {
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
    // reasoning control is model-dependent on Bedrock → not auto-mapped in v1;
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
    const inputTokens = json?.usage?.inputTokens ?? 0;
    const outputTokens = json?.usage?.outputTokens ?? 0;
    return {
      message: {
        role: 'assistant',
        content: text.length > 0 ? text : null,
        ...(toolCalls.length > 0 ? { toolCalls } : {}),
      },
      finishReason: mapStopReason(json?.stopReason),
      usage: {
        inputTokens,
        outputTokens,
        totalTokens: json?.usage?.totalTokens ?? inputTokens + outputTokens,
      },
      model: requestedModel,
      raw: json,
    };
  }

  private mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
    const message = extractBedrockError(body) ?? `HTTP ${status}`;
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
// Converse wire shapes (minimal)
// ---------------------------------------------------------------------------

interface ConverseResponse {
  stopReason?: string | null;
  usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number };
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
        blocks.push({ text: p.text });
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
  if (tc === 'none') return null; // Bedrock has no "none"; omit toolChoice
  if (typeof tc === 'object' && tc.type === 'function') return { tool: { name: tc.function.name } };
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
    case 'content_filtered':
    case 'guardrail_intervened':
      return 'content_filter';
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
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'provider_unavailable';
  if (status === 400 || status === 404 || status === 422) {
    const msg = message.toLowerCase();
    if (msg.includes('throttl')) return 'rate_limited';
    if (msg.includes('too long') || msg.includes('context') || msg.includes('token') || msg.includes('max')) {
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
