import { OpenAICompatEmbeddingsConnector } from '../providers/_shared/openai-compat.embeddings.connector';
import { SPECS } from '../providers/_shared/spec';
import type { SpecId } from '../providers/_shared/spec';
import { ConnectorError } from '../types';
import type {
  EmbeddingsInput,
  EmbeddingsResult,
  IEmbeddingsConnector,
  OpenAICompatConfig,
} from '../types';

/**
 * First-class providers that expose an OpenAI-float-shaped `/embeddings`
 * surface (FR-AI-E1). Deliberately excludes: providers with no native
 * embeddings (deepseek, xai), quantized/non-float shapes (perplexity int8),
 * unstable coverage (groq), and the native-adapter providers (Anthropic →
 * Voyage; Bedrock/Gemini native embeddings have a different shape — a future
 * per-connector embeddings surface).
 */
export const EMBEDDINGS_PROVIDER_IDS = [
  'openai',
  'azure-openai',
  'openrouter',
  'together',
  'fireworks',
  'mistral',
  'deepinfra',
  'cloudflare',
  'vllm',
  'ollama',
  'lmstudio',
] as const;

export type EmbeddingsProviderId = (typeof EMBEDDINGS_PROVIDER_IDS)[number];

const SUPPORTED = new Set<string>(EMBEDDINGS_PROVIDER_IDS);

/**
 * The embeddings facade — a separate operation from `Chat`. Construct by
 * provider id + config, or with a connector instance. Only the OpenAI-float
 * subset is accepted (type-restricted); other ids throw `ConnectorError`.
 */
export class Embeddings<P extends EmbeddingsProviderId = EmbeddingsProviderId> {
  public readonly id: string;
  private readonly connector: IEmbeddingsConnector;

  constructor(providerId: P, config: OpenAICompatConfig);
  constructor(connector: IEmbeddingsConnector);
  constructor(arg: P | IEmbeddingsConnector, config?: OpenAICompatConfig) {
    if (typeof arg === 'object' && arg !== null) {
      this.connector = arg;
      this.id = arg.id;
      return;
    }
    const providerId = arg;
    this.id = providerId;
    if (!SUPPORTED.has(providerId)) {
      throw new ConnectorError({
        message: `Provider '${String(providerId)}' has no OpenAI-compatible embeddings surface`,
        statusCode: null,
        providerCode: 'invalid_request',
      });
    }
    if (!config) {
      throw new ConnectorError({
        message: 'Embeddings facade requires `config` when constructed with a provider id',
        statusCode: null,
        providerCode: 'invalid_request',
      });
    }
    this.connector = new OpenAICompatEmbeddingsConnector(SPECS[providerId as SpecId], config);
  }

  create(input: EmbeddingsInput): Promise<EmbeddingsResult> {
    return this.connector.create(input);
  }
}
