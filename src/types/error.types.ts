export type ProviderCode =
  | 'rate_limited'
  | 'auth_failed'
  | 'provider_unavailable'
  | 'invalid_request'
  | 'context_length_exceeded'
  | 'content_filtered'
  | 'unknown';

/**
 * Canonical error thrown by every connector on vendor HTTP non-2xx (and on
 * network/abort failures). `providerCode` is narrowed to the `ProviderCode`
 * union; the verbatim vendor error payload is preserved on `cause.raw`, and any
 * retry hint on `cause.retryAfter` / `cause.retryAfterSeconds`. There is no
 * top-level `retryAfterSeconds` field (umbrella invariant).
 */
export class ConnectorError extends Error {
  public readonly statusCode: number | null;
  public readonly providerCode?: ProviderCode;
  public readonly providerMessage: string | null;

  constructor(options: {
    message?: string;
    statusCode: number | null;
    providerCode?: ProviderCode;
    providerMessage?: string | null;
    cause?: unknown;
  }) {
    super(options.message ?? options.providerMessage ?? 'Connector error', {
      cause: options.cause,
    });
    this.name = 'ConnectorError';
    this.statusCode = options.statusCode;
    this.providerCode = options.providerCode;
    this.providerMessage = options.providerMessage ?? null;
  }
}
