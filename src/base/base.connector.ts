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
      // The provider may have answered and the dispatcher turned that answer
      // into a rejection. Rebuilding the `Response` — rather than classifying
      // here — keeps the per-connector `mapVendorError` the single owner of
      // status→code translation. Without it a 429 reads as `provider_unavailable`
      // with a null status, which is the single most consequential misreport an
      // LLM caller can get.
      const rebuilt = responseFromThrownHttpError(err);
      if (rebuilt !== null) return rebuilt;
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
 * → `invalid_request`); any other rejection (network reset, DNS, timeout) is
 * `provider_unavailable`.
 *
 * Neither the raw message nor the raw error object is propagated: a BYO
 * `fetchImpl` (and undici itself) can embed the request URL in its message, and
 * for query-key providers that URL **is** the credential. Surfacing it would
 * re-expose the secret through `error.message`, `error.cause` and every logger
 * that serializes them (CWE-532). The PHP sibling has always used a fixed
 * message here; this brings TypeScript in line.
 */
export function transportErrorFrom(err: unknown): ConnectorError {
  if (err instanceof ConnectorError) return err;
  const name = (err as Error)?.name;
  if (name === 'AbortError') {
    return new ConnectorError({
      message: 'Request cancelled',
      statusCode: null,
      providerCode: 'invalid_request',
      cause: { raw: sanitizeThrownError(err) },
    });
  }
  return new ConnectorError({
    message: 'Network error',
    statusCode: null,
    providerCode: 'provider_unavailable',
    cause: { raw: sanitizeThrownError(err) },
  });
}

// ---------------------------------------------------------------------------
// Transport-rejection normalization (duck-typed; never an `undici` import)
// ---------------------------------------------------------------------------

/** Walk `err`, `err.cause`, `err.cause.cause` — deep enough for a wrapped rejection. */
function* errorChain(err: unknown, depth = 3): Generator<Record<string, unknown>> {
  let current = err;
  for (let i = 0; i < depth; i++) {
    if (current === null || typeof current !== 'object') return;
    yield current as Record<string, unknown>;
    current = (current as { cause?: unknown }).cause;
  }
}

/**
 * Rebuild a `Response` from a rejection carrying an HTTP status, or `null` when
 * the rejection is a genuine transport failure.
 *
 * Fetch resolves a non-2xx; a dispatcher may reject instead. On Node,
 * `globalThis.fetch` runs through undici's **process-global** dispatcher, which
 * the host application can replace at any time — `setGlobalDispatcher(new
 * Agent().compose(retry(), interceptors.responseError()))` is ordinary app code
 * — with no signal reaching this library. Under it, `fetch` rejects with
 * `TypeError: fetch failed` whose `.cause` carries `statusCode`/`headers`/`body`.
 *
 * Only `body` is read as a body. undici's retry-exhaustion error carries a
 * `data` field that holds `{ count }` retry metadata, NOT the response payload;
 * treating it as one would fabricate a body the provider never sent.
 */
function responseFromThrownHttpError(err: unknown): Response | null {
  if (typeof Response !== 'function') return null;

  for (const link of errorChain(err)) {
    const status = link.statusCode ?? link.status;
    if (typeof status !== 'number' || !Number.isInteger(status) || status < 200 || status > 599) {
      continue;
    }
    // 204/205/304 are null-body statuses; `new Response(body, …)` throws for them.
    const body = status === 204 || status === 205 || status === 304 ? null : readErrorBody(link.body, link.headers);
    try {
      return new Response(body, { status, headers: readErrorHeaders(link.headers) });
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Normalize a buffered error body to something the `Response` constructor accepts.
 *
 * A `content-encoding` on the error means the body was buffered **still
 * compressed** and is unrecoverable — drop it rather than hand a connector
 * mojibake. A dispatcher-level interceptor sits *below* fetch's content-decoding,
 * so it buffers raw wire bytes and text-decodes them; gzip is not valid UTF-8, so
 * every invalid byte becomes U+FFFD before we ever see the value. Measured on a
 * real gzipped 400: 16 of 66 bytes replaced, the `1f 8b` magic destroyed, and
 * re-encoding as latin1/binary/utf8 all fail to inflate (`Z_DATA_ERROR`). There
 * is nothing here to salvage — not with `zlib`, not with anything.
 *
 * (Verified against undici 8.9: this is driven by `content-encoding`, not by the
 * `;charset=` suffix — its content-type check is a prefix match. Its own
 * `decompress()` interceptor does not help in any composition order.)
 */
function readErrorBody(body: unknown, headers: unknown): string | Uint8Array | ArrayBuffer | null {
  if (typeof body === 'string') return isContentEncoded(headers) ? null : body;
  if (body instanceof Uint8Array || body instanceof ArrayBuffer) return body;
  if (body === null || body === undefined) return null;
  if (typeof body === 'object') {
    // Already-decoded JSON (undici parses `application/json` before throwing).
    try {
      return JSON.stringify(body);
    } catch {
      return null;
    }
  }
  return null;
}

/** Whether the error's headers declare a body encoding other than `identity`. */
function isContentEncoded(headers: unknown): boolean {
  if (headers === null || typeof headers !== 'object') return false;
  for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
    if (key.toLowerCase() !== 'content-encoding') continue;
    const encoding = (Array.isArray(value) ? value.join(',') : String(value)).trim().toLowerCase();
    return encoding !== '' && encoding !== 'identity';
  }
  return false;
}

/**
 * Copy the string-valued headers off a thrown error. `content-length` and
 * `content-encoding` are dropped: they describe a wire encoding that no longer
 * applies once the body has been decoded and re-serialized.
 */
function readErrorHeaders(headers: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (headers === null || typeof headers !== 'object') return out;
  for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
    const name = key.toLowerCase();
    if (name === 'content-length' || name === 'content-encoding') continue;
    if (typeof value === 'string') out[name] = value;
    else if (Array.isArray(value) && value.every((v) => typeof v === 'string')) {
      out[name] = value.join(', ');
    }
  }
  return out;
}

/**
 * A diagnostic-code shape: identifier characters only. A credential would reach
 * here inside a URL, which cannot match this pattern (no `/`, `:`, `?`, `=` or
 * whitespace). Length-capped so an opaque token cannot masquerade as one.
 */
const DIAGNOSTIC_CODE = /^[A-Za-z0-9_.-]{1,64}$/;

function readDiagnosticCode(value: unknown): string | undefined {
  return typeof value === 'string' && DIAGNOSTIC_CODE.test(value) ? value : undefined;
}

/**
 * The non-secret, structured fields of a transport rejection. `name` alone is
 * `'TypeError'` for every undici failure — DNS, reset, TLS — and identifies
 * nothing; `code` (`ECONNRESET`, `UND_ERR_SOCKET`, …) is the field that names it.
 */
function sanitizeThrownError(err: unknown): Record<string, unknown> {
  const raw: Record<string, unknown> = { name: (err as Error)?.name };
  const [self, cause] = [...errorChain(err, 2)];

  const code = readDiagnosticCode(self?.code) ?? readDiagnosticCode(cause?.code);
  if (code !== undefined) raw.code = code;

  const causeName = readDiagnosticCode(cause?.name);
  if (causeName !== undefined) raw.causeName = causeName;

  const statusCode = cause?.statusCode ?? self?.statusCode;
  if (typeof statusCode === 'number') raw.statusCode = statusCode;

  return raw;
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
