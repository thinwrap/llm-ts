export interface Passthrough {
  /**
   * Merged into the outbound request JSON body; overrides connector-built fields
   * on key collision. Never transformed or validated — if you populate this from
   * untrusted input, sanitize it yourself (you own the keys that reach the wire).
   */
  body?: Record<string, unknown>;
  /** Merged into request headers. */
  headers?: Record<string, string>;
  /** Appended to the request URL query string. */
  query?: Record<string, string>;
}

export type WithPassthrough<T> = T & { _passthrough?: Passthrough };
