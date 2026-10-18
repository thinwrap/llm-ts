import { describe, it, expect } from 'vitest';
import { parseRetryAfterSeconds, parseSSEStream } from './base.connector';

function sseResponse(body: string | ReadableStream<Uint8Array>): Response {
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}

describe('parseSSEStream', () => {
  it('yields JSON payloads and stops at [DONE]', async () => {
    const sse = 'data: {"n":1}\n\ndata: {"n":2}\n\ndata: [DONE]\n\ndata: {"n":3}\n\n';
    const out: unknown[] = [];
    for await (const evt of parseSSEStream(sseResponse(sse))) out.push(evt);
    expect(out).toEqual([{ n: 1 }, { n: 2 }]); // n:3 after [DONE] is not read
  });

  it('throws a ConnectorError when a single event exceeds the buffer cap (DoS guard)', async () => {
    // one `data:` line, no `\n\n` boundary, larger than MAX_SSE_EVENT_CHARS (16 MiB)
    const huge = 'data: ' + 'x'.repeat(16 * 1024 * 1024 + 100);
    await expect(
      (async () => {
        for await (const evt of parseSSEStream(sseResponse(huge))) void evt;
      })(),
    ).rejects.toMatchObject({ name: 'ConnectorError', providerCode: 'provider_unavailable' });
  });

  it('cancels the underlying body on the [DONE] sentinel (no socket leak on a successful stream)', async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        // one event then [DONE]; stream is left OPEN (undici keeps the socket until cancel)
        controller.enqueue(new TextEncoder().encode('data: {"n":1}\n\ndata: [DONE]\n\n'));
      },
      cancel() {
        cancelled = true;
      },
    });
    const out: unknown[] = [];
    for await (const evt of parseSSEStream(sseResponse(stream))) out.push(evt);
    expect(out).toEqual([{ n: 1 }]); // stopped at [DONE]
    expect(cancelled).toBe(true); // and cancelled the body rather than leaking it
  });

  it('maps a mid-stream read failure to a ConnectorError (not a raw error)', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"n":1}\n\n'));
      },
      pull() {
        // second read rejects: a TCP reset / broken upstream mid-stream
        throw new Error('connection reset by peer');
      },
    });
    const out: unknown[] = [];
    await expect(
      (async () => {
        for await (const evt of parseSSEStream(sseResponse(stream))) out.push(evt);
      })(),
    ).rejects.toMatchObject({ name: 'ConnectorError', providerCode: 'provider_unavailable' });
    expect(out).toEqual([{ n: 1 }]); // the good event before the drop still surfaced
  });

  it('splits events on CR-only and mixed line terminators', async () => {
    // \r\r (CR-only) and \n\r\n (mixed) boundaries — a legit-but-uncommon upstream
    const sse = 'data: {"n":1}\r\rdata: {"n":2}\n\r\ndata: [DONE]\r\r';
    const out: unknown[] = [];
    for await (const evt of parseSSEStream(sseResponse(sse))) out.push(evt);
    expect(out).toEqual([{ n: 1 }, { n: 2 }]);
  });

  it('cancels the underlying body when the consumer breaks out early (no socket leak)', async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        // two complete events, stream left open (never closed)
        controller.enqueue(new TextEncoder().encode('data: {"n":1}\n\ndata: {"n":2}\n\n'));
      },
      cancel() {
        cancelled = true;
      },
    });
    const gen = parseSSEStream(sseResponse(stream));
    const first = await gen.next();
    expect(first.value).toEqual({ n: 1 });
    // break out early → generator .return() runs the finally, which must cancel
    await gen.return(undefined as never);
    expect(cancelled).toBe(true);
  });
});

describe('parseSSEStream CRLF handling', () => {
  it('treats a single CRLF as a line end, not an event boundary', async () => {
    const sse = 'data: {"a":\r\ndata: 1}\r\n\r\ndata: {"b":2}\r\n\r\n';
    const out: unknown[] = [];
    for await (const evt of parseSSEStream(sseResponse(sse))) out.push(evt);
    expect(out).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it('handles a CRLF split across reads', async () => {
    const chunks = ['data: {"a":\r', '\ndata: 1}\r', '\n\r', '\ndata: {"b":2}\r\n\r\n'];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const c of chunks) controller.enqueue(new TextEncoder().encode(c));
        controller.close();
      },
    });
    const out: unknown[] = [];
    for await (const evt of parseSSEStream(sseResponse(stream))) out.push(evt);
    expect(out).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it('records the [DONE] sentinel on the state object', async () => {
    const state = { sawDone: false };
    for await (const evt of parseSSEStream(sseResponse('data: {"n":1}\n\ndata: [DONE]'), state)) void evt;
    expect(state.sawDone).toBe(true);
  });
});

describe('parseRetryAfterSeconds', () => {
  it.each([
    ['', null],
    ['  ', null],
    ['-1', null],
    ['nonsense', null],
    ['0', 0],
    ['30', 30],
    ['Wed, 21 Oct 2015 07:28:00 GMT', 0],
  ])('parses %j → %j', (header, expected) => {
    expect(parseRetryAfterSeconds(header)).toBe(expected);
  });
});

describe('parseRetryAfterSeconds (RFC 9110 forms only)', () => {
  it.each(['1.5', '+5', '1e3', 'tomorrow', '+1 week', '2026-10-18T10:00:00Z', 'Sun, 32 Nov 1994 08:49:37 GMT', 'Sun, 06 Nov 1994 25:49:37 GMT'])(
    'ignores %j',
    (header) => {
      expect(parseRetryAfterSeconds(header)).toBeNull();
    },
  );

  it.each(['Sun, 06 Nov 1994 08:49:37 GMT', 'Sunday, 06-Nov-94 08:49:37 GMT', 'Sun Nov  6 08:49:37 1994'])(
    'reads the past HTTP-date %j as 0',
    (header) => {
      expect(parseRetryAfterSeconds(header)).toBe(0);
    },
  );

  it('reads a future IMF-fixdate as the seconds until it', () => {
    const at = new Date(Date.now() + 120_000);
    at.setUTCMilliseconds(0);
    const header = at.toUTCString();
    const secs = parseRetryAfterSeconds(header)!;
    expect(secs).toBeGreaterThanOrEqual(118);
    expect(secs).toBeLessThanOrEqual(120);
  });
});
