/**
 * Provider ids implemented so far — the 15 first-class OpenAI-compatible
 * providers (one shared connector) plus native-adapter providers as they land
 * (`anthropic`). Remaining natives (bedrock, gemini, vertex, cohere) are added
 * when their adapters ship.
 */
export const LLM_PROVIDER_IDS = [
  'openai',
  'azure-openai',
  'openrouter',
  'groq',
  'together',
  'fireworks',
  'deepseek',
  'xai',
  'mistral',
  'perplexity',
  'deepinfra',
  'cloudflare',
  'vllm',
  'ollama',
  'lmstudio',
  'anthropic',
  'bedrock',
  'gemini',
] as const;

export type LlmProviderId = (typeof LLM_PROVIDER_IDS)[number];
