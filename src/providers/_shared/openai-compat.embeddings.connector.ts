import { appendPassthroughQuery, BaseConnector } from '../../base/base.connector';
import { ConnectorError } from '../../types';
import type {
  EmbeddingsInput,
  EmbeddingsResult,
  IEmbeddingsConnector,
  OpenAICompatConfig,
  ProviderCode,
} from '../../types';
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
    if (input.dimensions !== undefined) body.dimensions = input.dimensions;
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
    const rows = [...(json?.data ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    const inputTokens = json?.usage?.prompt_tokens ?? 0;
    return {
      embeddings: rows.map((d) => d.embedding ?? []),
      usage: { inputTokens, totalTokens: json?.usage?.total_tokens ?? inputTokens },
      model: json?.model ?? input.model,
      raw: json,
    };
  }
}

interface EmbeddingsResponse {
  model?: string;
  data?: Array<{ index?: number; embedding?: number[] }>;
  usage?: { prompt_tokens?: number; total_tokens?: number };
}

function mapVendorError(status: number, body: unknown, headers: Headers): ConnectorError {
  const message = extractErrorMessage(body) ?? `HTTP ${status}`;
  const cause: Record<string, unknown> = { raw: body ?? null };
  const retryAfter = headers.get('retry-after');
  if (retryAfter != null) {
    cause.retryAfter = retryAfter;
    const secs = Number(retryAfter);
    if (Number.isFinite(secs)) cause.retryAfterSeconds = secs;
  }
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
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'provider_unavailable';
  if (status === 400 || status === 404 || status === 422) return 'invalid_request';
  return 'unknown';
}
