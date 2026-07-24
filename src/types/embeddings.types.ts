import type { Passthrough } from './passthrough.type';

export interface EmbeddingsInput {
  model: string;
  /** A single string or a batch of strings. */
  input: string | string[];
  /** Optional output dimensionality (providers that support `dimensions` / Matryoshka). */
  dimensions?: number;
  /** Optional cancellation signal, forwarded to the underlying `fetch`. */
  signal?: AbortSignal;
  /** Escape hatch for provider-specific request fields (e.g. `input_type`). Never emulated. */
  _passthrough?: Passthrough;
}

export interface EmbeddingsUsage {
  inputTokens: number;
  totalTokens: number;
}

export interface EmbeddingsResult {
  /** One float vector per input, in input order. */
  embeddings: number[][];
  usage: EmbeddingsUsage;
  /** The model id the provider reported serving. */
  model: string;
  /** Verbatim vendor response body. */
  raw: unknown;
}

export interface IEmbeddingsConnector {
  readonly id: string;
  create(input: EmbeddingsInput): Promise<EmbeddingsResult>;
}
