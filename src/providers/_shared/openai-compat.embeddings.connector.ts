import { appendPassthroughQuery, applyRetryAfter, BaseConnector } from '../../base/base.connector';
import { ConnectorError } from '../../types';
import type {
  EmbeddingsInput,
  EmbeddingsResult,
  IEmbeddingsConnector,
  OpenAICompatConfig,
  ProviderCode,
} from '../../types';
import { embeddedErrorStatus } from './openai-compat.connector';
import type { OpenAICompatSpec } from './spec';

/**
 * The shared embeddings connector for OpenAI-float-shaped providers
 * (`POST /embeddings`). Decoupled from chat (a separate operation surface).
 * Requests `encoding_format: 'float'` and normalizes `data[].embedding` into
 * `number[][]` in input order. Only wired for providers that expose a
 * float-shaped embeddings endpoint (see `EMBEDDINGS_PROVIDER_IDS`).
 */
export class OpenAICompatEmbeddingsConnector extends BaseConnector implements IEmbeddingsConnector {
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

  async create(input: EmbeddingsInput): Promise<EmbeddingsResult> {
    const body: Record<string, unknown> = {
      model: input.model,
      input: input.input,
      encoding_format: 'float',
    };
    if (input.dimensions !== undefined) body[this.spec.dimensionsField ?? 'dimensions'] = input.dimensions;
    if (input._passthrough?.body) Object.assign(body, input._passthrough.body);

    const headers: Record<string, string> = {
      ...this.spec.buildAuthHeaders(this.config),
      ...this.config.headers,
      ...input._passthrough?.headers,
    };

    const url = appendPassthroughQuery(`${this.baseUrl}/embeddings`, input._passthrough?.query);
    const response = await this.sendPostJson(url, body, headers, input.signal);
    if (!response.ok) {
      const errBody = await response.json().catch(() => null);
      throw mapVendorError(response.status, errBody, response.headers);
    }
    const json = this.requireDecodedBody<EmbeddingsResponse>((await response.json().catch(() => null)) as EmbeddingsResponse | null, response.status);
    const embeddings = this.requireVectors(json, body.input, response.status, response.headers);
    const inputTokens = json.usage?.prompt_tokens ?? 0;
    return {
      embeddings,
      usage: { inputTokens, totalTokens: json.usage?.total_tokens ?? inputTokens },
      model: json.model ?? input.model,
      raw: json,
    };
  }

  /**
   * Validate a 2xx body into one vector per input, in input order. A gateway
   * can answer 200 with an `error` object, and a short or malformed `data`
   * array would silently misalign vectors with inputs — a caller storing them
   * by position would index the wrong text. A `null` cell is never coerced to
   * 0 for the same reason: it would be a plausible-looking wrong vector.
   * `sentInput` is the input actually sent (a passthrough body may replace it).
   */
  private requireVectors(json: EmbeddingsResponse, sentInput: unknown, status: number, headers: Headers): number[][] {
    if (json.error != null) throw inBodyError(status, json.error, json, headers);
    const data = json.data;
    if (!Array.isArray(data)) throw this.missingPayloadError(status, json, 'no data');
    const expected = Array.isArray(sentInput) ? sentInput.length : 1;
    if (data.length !== expected) {
      throw this.missingPayloadError(status, json, `${data.length} embeddings for ${expected} inputs`);
    }
    const rows = [...data].sort((a, b) => (a?.index ?? 0) - (b?.index ?? 0));
    return rows.map((row, i) => {
      const vector = row?.embedding;
      if (!Array.isArray(vector) || !vector.every((v) => typeof v === 'number' && Number.isFinite(v))) {
        throw this.missingPayloadError(status, json, `embedding ${i} is not an array of finite numbers`);
      }
      return vector as number[];
    });
  }
}

interface EmbeddingsResponse {
  model?: string;
  /** Error delivered on a 2xx by a gateway. */
  error?: unknown;
  data?: Array<{ index?: number; embedding?: unknown[] }>;
  usage?: { prompt_tokens?: number; total_tokens?: number };
}

/**
 * An error inside a 2xx body keeps the status it arrived with; its code comes
 * from the embedded error the way chat classifies it (`unknown` with no usable
 * code or type).
 */
function inBodyError(httpStatus: number, error: unknown, body: unknown, headers: Headers): ConnectorError {
  const effective = embeddedErrorStatus(error);
  const message = extractErrorMessage(body) ?? `HTTP ${httpStatus}`;
  const cause: Record<string, unknown> = { raw: body ?? null };
  applyRetryAfter(cause, headers);
  return new ConnectorError({
    message,
    statusCode: httpStatus,
    providerCode: effective === null ? 'unknown' : mapStatusToProviderCode(effective),
    providerMessage: message,
    cause,
  });
}

function mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
  const message = extractErrorMessage(body) ?? `HTTP ${status}`;
  const cause: Record<string, unknown> = { raw: body ?? null };
  applyRetryAfter(cause, headers);
  return new ConnectorError({
    message,
    statusCode: status,
    providerCode: mapStatusToProviderCode(status),
    providerMessage: message,
    cause,
  });
}

function extractErrorMessage(body: unknown): string | null {
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

function mapStatusToProviderCode(status: number): ProviderCode {
  if (status === 401 || status === 403) return 'auth_failed';
  // 402 is credit/quota exhaustion, which OpenAI (429 insufficient_quota) and
  // Gemini (RESOURCE_EXHAUSTED) already report as rate-limited, and
  // @thinwrap/notifications maps Pusher Beams' 402 the same way.
  if (status === 429 || status === 402) return 'rate_limited';
  if (status >= 500 || status === 408) return 'provider_unavailable';
  // 404, 413, 422, …: the request cannot succeed as sent.
  if (status >= 400) return 'invalid_request';
  return 'unknown';
}
