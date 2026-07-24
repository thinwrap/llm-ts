import { OpenAICompatConnector } from '../providers/_shared/openai-compat.connector';
import { SPECS } from '../providers/_shared/spec';
import { AnthropicConnector } from '../providers/anthropic';
import { BedrockConnector } from '../providers/bedrock';
import { GeminiConnector } from '../providers/gemini';
import { ConnectorError } from '../types';
import type {
  LlmProviderId,
  ChatInput,
  ChatResult,
  ChatStreamDelta,
  IChatConnector,
  OpenAICompatConfig,
  ProviderConfigMap,
} from '../types';
import type { AnthropicConfig } from '../providers/anthropic';
import type { BedrockConfig } from '../providers/bedrock';
import type { GeminiConfig } from '../providers/gemini';

export type ChatConfig<P extends LlmProviderId> = ProviderConfigMap[P];

/**
 * The chat-completion facade. Construct by provider id + config — the facade
 * dispatches to the shared OpenAI-compatible connector (native adapters slot in
 * here as they land) — or pass a connector instance directly for advanced use.
 * Switching vendor is a change of provider id + `model` only; `ChatInput` /
 * `ChatResult` are identical across providers.
 */
export class Chat<P extends LlmProviderId = LlmProviderId> {
  public readonly id: string;
  private readonly connector: IChatConnector;

  constructor(providerId: P, config: ChatConfig<P>);
  constructor(connector: IChatConnector);
  constructor(arg: P | IChatConnector, config?: ChatConfig<P>) {
    if (typeof arg === 'object' && arg !== null) {
      this.connector = arg;
      this.id = arg.id;
      return;
    }
    const providerId = arg;
    this.id = providerId;
    if (!config) {
      throw new ConnectorError({
        message: 'Chat facade requires `config` when constructed with a provider id',
        statusCode: null,
        providerCode: 'invalid_request',
      });
    }
    // Native adapters — structurally different wire, same normalized surface.
    if (providerId === 'anthropic') {
      this.connector = new AnthropicConnector(config as AnthropicConfig);
      return;
    }
    if (providerId === 'bedrock') {
      this.connector = new BedrockConnector(config as BedrockConfig);
      return;
    }
    if (providerId === 'gemini') {
      this.connector = new GeminiConnector(config as GeminiConfig);
      return;
    }
    const spec = SPECS[providerId as keyof typeof SPECS];
    if (!spec) {
      throw new ConnectorError({
        message: `Unsupported LLM provider: ${String(providerId)}`,
        statusCode: null,
        providerCode: 'invalid_request',
      });
    }
    this.connector = new OpenAICompatConnector(spec, config as OpenAICompatConfig);
  }

  complete(input: ChatInput): Promise<ChatResult> {
    return this.connector.complete(input);
  }

  stream(input: ChatInput): AsyncIterable<ChatStreamDelta> {
    return this.connector.stream(input);
  }
}
