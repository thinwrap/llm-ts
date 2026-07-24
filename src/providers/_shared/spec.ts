import type { OpenAICompatConfig } from '../../types';

/**
 * Per-provider parameterization of the one shared OpenAI-compatible connector.
 * Base URLs are current-as-of-2026 defaults and are the exact fact each
 * per-connector README pins; treat as overridable via `config.baseUrl`.
 */
export interface OpenAICompatSpec {
  id: string;
  /** Default base URL (no trailing slash). Undefined ⇒ consumer must supply `config.baseUrl`. */
  defaultBaseUrl?: string;
  buildAuthHeaders: (config: OpenAICompatConfig) => Record<string, string>;
  /** Providers with no tool support on the primary chat surface (Perplexity Sonar). */
  supportsTools?: boolean;
  /** Vendor field name for max output tokens (default `max_tokens`). */
  maxTokensField?: string;
  /** Groq relocates streaming usage under `x_groq.usage` on the final chunk (baseline-exception). */
  streamUsagePath?: 'x_groq';
  /** Mistral rejects unknown params with HTTP 422 — skip OpenAI-only extras like `stream_options`. */
  strictParams?: boolean;
}

const bearer = (config: OpenAICompatConfig): Record<string, string> =>
  config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {};

export const SPECS = {
  openai: {
    id: 'openai',
    defaultBaseUrl: 'https://api.openai.com/v1',
    buildAuthHeaders: bearer,
  },
  'azure-openai': {
    id: 'azure-openai',
    // no default: consumer supplies https://<resource>.openai.azure.com/openai/v1
    buildAuthHeaders: (config: OpenAICompatConfig): Record<string, string> =>
      config.apiKey ? { 'api-key': config.apiKey } : {},
  },
  openrouter: {
    id: 'openrouter',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    buildAuthHeaders: bearer,
  },
  groq: {
    id: 'groq',
    defaultBaseUrl: 'https://api.groq.com/openai/v1',
    buildAuthHeaders: bearer,
    streamUsagePath: 'x_groq',
  },
  together: {
    id: 'together',
    defaultBaseUrl: 'https://api.together.ai/v1',
    buildAuthHeaders: bearer,
  },
  fireworks: {
    id: 'fireworks',
    defaultBaseUrl: 'https://api.fireworks.ai/inference/v1',
    buildAuthHeaders: bearer,
  },
  deepseek: {
    id: 'deepseek',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    buildAuthHeaders: bearer,
  },
  xai: {
    id: 'xai',
    defaultBaseUrl: 'https://api.x.ai/v1',
    buildAuthHeaders: bearer,
  },
  mistral: {
    id: 'mistral',
    defaultBaseUrl: 'https://api.mistral.ai/v1',
    buildAuthHeaders: bearer,
    strictParams: true,
  },
  perplexity: {
    id: 'perplexity',
    defaultBaseUrl: 'https://api.perplexity.ai',
    buildAuthHeaders: bearer,
    supportsTools: false,
  },
  deepinfra: {
    id: 'deepinfra',
    defaultBaseUrl: 'https://api.deepinfra.com/v1/openai',
    buildAuthHeaders: bearer,
  },
  cloudflare: {
    id: 'cloudflare',
    // no default: consumer supplies https://api.cloudflare.com/client/v4/accounts/<id>/ai/v1
    buildAuthHeaders: bearer,
  },
  vllm: {
    id: 'vllm',
    defaultBaseUrl: 'http://localhost:8000/v1',
    buildAuthHeaders: bearer,
  },
  ollama: {
    id: 'ollama',
    defaultBaseUrl: 'http://localhost:11434/v1',
    buildAuthHeaders: bearer,
  },
  lmstudio: {
    id: 'lmstudio',
    defaultBaseUrl: 'http://localhost:1234/v1',
    buildAuthHeaders: bearer,
  },
} satisfies Record<string, OpenAICompatSpec>;

export type SpecId = keyof typeof SPECS;
