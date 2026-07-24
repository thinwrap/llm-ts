export interface GeminiConfig {
  /** Google AI Studio API key — sent as the `x-goog-api-key` header. */
  apiKey: string;
  /** Override the base URL. Default `https://generativelanguage.googleapis.com/v1beta`. */
  baseUrl?: string;
  /** `generationConfig.maxOutputTokens` used when `ChatInput.maxOutputTokens` is omitted. */
  defaultMaxTokens?: number;
  /** Bring-your-own fetch. Defaults to `globalThis.fetch`. */
  fetch?: typeof fetch;
  /** Extra headers merged onto every request. */
  headers?: Record<string, string>;
}
