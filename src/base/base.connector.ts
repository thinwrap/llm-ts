import { ConnectorError } from '../types';

/**
 * Shared HTTP plumbing for every connector. Holds the BYO `fetch` and maps
 * transport-level failures (abort, network) to `ConnectorError`. Applies no
 * automatic key transformation and holds no state (umbrella invariants).
 */
export abstract class BaseConnector {
  protected readonly fetchImpl: typeof fetch;

  protected constructor(fetchImpl?: typeof fetch) {
    const baseFetch = fetchImpl ?? globalThis.fetch;
    // Force `redirect: 'error'` on every request at the transport seam so a 3xx
    // can never silently re-send credentials to the redirect target. `fetch`
    // strips only `Authorization`/`Cookie` cross-origin, so provider key headers
    // (x-api-key, x-goog-api-key, api-key, …) would otherwise leak. No LLM
    // endpoint legitimately redirects a POST.
    this.fetchImpl = ((input: Parameters<typeof fetch>[0], init?: RequestInit) =>
      baseFetch(input, { ...init, redirect: 'error' })) as typeof fetch;
  }

  protected async sendPostJson(
    url: string,
    body: unknown,
    headers: Record<string, string>,
    signal?: AbortSignal,
  ): Promise<Response> {
    return this.invokeFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
      signal,
    });
  }

  protected async invokeFetch(url: string, init: RequestInit): Promise<Response> {
    try {
      return await this.fetchImpl(url, init);
    } catch (err) {
      throw transportErrorFrom(err);
    }
  }

  /**
   * Guard a decoded 2xx body: a `null` here means the body was empty or not
   * JSON (a proxy/captive-portal HTML page, a truncated read). Parsing that into
   * an empty "successful" result is silent data loss, so raise instead.
   */
  protected requireDecodedBody<T>(json: T | null, status: number): T {
    if (json === null) {
      const message =
        'Provider returned a successful status with an empty or non-JSON body';
      throw new ConnectorError({
        message,
        statusCode: status,
        providerCode: 'provider_unavailable',
        providerMessage: message,
        cause: null,
      });
    }
    return json;
  }
}

/**
 * Map a caught transport-layer failure to a `ConnectorError`. Shared by the
 * request path (`invokeFetch`) and the streaming read loop (`parseSSEStream`) so
 * a mid-stream TCP drop / abort surfaces with the same taxonomy as a failure
 * before the first byte — never a raw `TypeError`/`DOMException`.
 *
 * A manual `AbortController.abort()` raises an `AbortError` (consumer cancelled
 * → `invalid_request`); `AbortSignal.timeout()` raises a distinct `TimeoutError`
 * (the provider failed to respond in time → `provider_unavailable`). Any other
 * rejection (network reset, DNS) is `provider_unavailable`.
 */
export function transportErrorFrom(err: unknown): ConnectorError {
  if (err instanceof ConnectorError) return err;
  const name = (err as Error)?.name;
  if (name === 'AbortError') {
    return new ConnectorError({
      message: (err as Error).message || 'Request cancelled',
      statusCode: null,
      providerCode: 'invalid_request',
      cause: { raw: err },
    });
  }
  return new ConnectorError({
    message: (err as Error)?.message || 'Network error',
    statusCode: null,
    providerCode: 'provider_unavailable',
    cause: { raw: err },
  });
}

/**
 * Append `_passthrough.query` params to a request URL. EVERY request path must
 * honor this — not just OpenAI-compat chat — because some providers require a
 * query param the facade doesn't model (e.g. Azure OpenAI's `api-version`).
 * Preserves any query string the URL already carries.
 */
export function appendPassthroughQuery(url: string, query?: Record<string, string>): string {
  if (!query || Object.keys(query).length === 0) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}${new URLSearchParams(query).toString()}`;
}

/**
 * Upper bound on the residual (not-yet-terminated) SSE buffer. A stream that
 * never emits an event boundary would otherwise grow `buffer` without limit
 * (memory-exhaustion DoS from a malicious/broken upstream). Generous enough for
 * any single legitimate LLM event; complete events are drained before this is
 * checked, so many-small-event streams never approach it.
 */
const MAX_SSE_EVENT_CHARS = 16 * 1024 * 1024;

/**
 * Parse a Server-Sent-Events response body into a sequence of JSON-decoded
 * `data:` payloads. Skips comments/blank lines, stops at the `[DONE]` sentinel,
 * and ignores any `data:` line that isn't valid JSON. Transport-agnostic over
 * `\n\n` and `\r\n\r\n` event boundaries.
 */
export async function* parseSSEStream(response: Response): AsyncGenerator<unknown> {
  const body = response.body;
  if (!body) return;
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let drained = false;
  try {
    for (;;) {
      let done: boolean;
      let value: Uint8Array | undefined;
      try {
        ({ done, value } = await reader.read());
      } catch (err) {
        // A mid-stream read rejection (TCP reset, abort, undici error) must
        // surface with the ConnectorError taxonomy, not as a raw DOMException —
        // streams are the failures consumers are least able to catch generically.
        throw transportErrorFrom(err);
      }
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary: number;
      while ((boundary = indexOfEventBoundary(buffer)) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary).replace(/^(?:\r\n|\r|\n)+/, '');
        const payload = extractData(rawEvent);
        if (payload === null) continue;
        if (payload === '[DONE]') {
          // Leave `drained = false` so the `finally` cancels the reader. The
          // sentinel does NOT mean the body is fully consumed — every
          // OpenAI-compat stream ends here, and without a cancel undici can't
          // release the socket (leak on every successful stream).
          return;
        }
        const parsed = tryParseJson(payload);
        if (parsed !== undefined) yield parsed;
      }

      if (buffer.length > MAX_SSE_EVENT_CHARS) {
        throw new ConnectorError({
          message: `SSE event exceeded ${MAX_SSE_EVENT_CHARS} characters without a boundary`,
          statusCode: null,
          providerCode: 'provider_unavailable',
        });
      }
    }
    const tail = buffer.trim();
    if (tail) {
      const payload = extractData(tail);
      if (payload !== null && payload !== '[DONE]') {
        const parsed = tryParseJson(payload);
        if (parsed !== undefined) yield parsed;
      }
    }
    drained = true;
  } finally {
    // If the consumer broke out early (or we threw), the body is not fully read —
    // cancel it so the underlying connection is released instead of leaking.
    if (!drained) {
      try {
        await reader.cancel();
      } catch {
        /* best-effort */
      }
    }
    reader.releaseLock();
  }
}

// An SSE event boundary is a blank line: two consecutive line terminators. The
// spec permits CRLF, CR, or LF as a terminator, so accept any pairing
// (`\n\n`, `\r\n\r\n`, `\r\r`, and mixed `\n\r\n` / `\r\n\n`) rather than only
// the two common forms — a CR-only or mixed upstream would otherwise never
// split and buffer until MAX_SSE_EVENT_CHARS.
const EVENT_BOUNDARY_RE = /(?:\r\n|\r|\n)(?:\r\n|\r|\n)/;

function indexOfEventBoundary(buffer: string): number {
  const m = EVENT_BOUNDARY_RE.exec(buffer);
  return m ? m.index : -1;
}

function extractData(rawEvent: string): string | null {
  const dataParts: string[] = [];
  for (const line of rawEvent.split(/\r\n|\r|\n/)) {
    if (line.startsWith('data:')) {
      dataParts.push(line.slice(5).replace(/^ /, ''));
    }
  }
  return dataParts.length === 0 ? null : dataParts.join('\n');
}

function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}
