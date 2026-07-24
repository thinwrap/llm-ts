import type { LlmProviderId } from './provider-id.enum';
import type { AnthropicConfig } from '../providers/anthropic/anthropic.config';
import type { BedrockConfig } from '../providers/bedrock/bedrock.config';
import type { GeminiConfig } from '../providers/gemini/gemini.config';

export interface OpenAICompatConfig {
  /** Bearer API key. Optional for self-host providers (vLLM / Ollama / LM Studio) that run without auth. */
  apiKey?: string;
  /**
   * Override the provider's default base URL (e.g. `https://api.openai.com/v1`).
   * Required for providers with no fixed public default: `azure-openai`,
   * `cloudflare`, and self-host providers on a non-default host.
   */
  baseUrl?: string;
  /** Bring-your-own fetch. Defaults to `globalThis.fetch`. */
  fetch?: typeof fetch;
  /** Extra headers merged onto every request (e.g. OpenRouter `HTTP-Referer` / `X-Title`). */
  headers?: Record<string, string>;
}

/**
 * Per-provider config type. The 15 first-class providers share
 * `OpenAICompatConfig`; native-adapter providers carry their own config.
 */
export type ProviderConfigMap = {
  [K in Exclude<LlmProviderId, 'anthropic' | 'bedrock' | 'gemini'>]: OpenAICompatConfig;
} & {
  anthropic: AnthropicConfig;
  bedrock: BedrockConfig;
  gemini: GeminiConfig;
};
