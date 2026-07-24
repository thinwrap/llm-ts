export interface AnthropicConfig {
  /** Anthropic API key — sent as the `x-api-key` header (not Bearer). */
  apiKey: string;
  /** Override the base URL. Default `https://api.anthropic.com/v1`. */
  baseUrl?: string;
  /** `anthropic-version` header value. Default `2023-06-01`. */
  anthropicVersion?: string;
  /**
   * Anthropic requires `max_tokens` on every request. Used when
   * `ChatInput.maxOutputTokens` is omitted. Default 4096.
   */
  defaultMaxTokens?: number;
  /** Bring-your-own fetch. Defaults to `globalThis.fetch`. */
  fetch?: typeof fetch;
  /** Extra headers merged onto every request. */
  headers?: Record<string, string>;
}
