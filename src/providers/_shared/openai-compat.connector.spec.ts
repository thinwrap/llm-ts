import { describe, it, expect, vi } from 'vitest';
import { OpenAICompatConnector } from './openai-compat.connector';
import { SPECS } from './spec';
import { ConnectorError } from '../../types';
import type { ChatInput, ChatStreamDelta } from '../../types';

const input: ChatInput = {
  model: 'test-model',
  messages: [{ role: 'user', content: 'hi' }],
};

function jsonResponse(
  status: number,
  body: unknown,
  headers?: Record<string, string>,
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

function mockFetch(factory: () => Response) {
  return vi.fn((_url: string, _init?: RequestInit): Promise<Response> =>
    Promise.resolve(factory()),
  );
}

const asFetch = (m: ReturnType<typeof mockFetch>): typeof fetch =>
  m as unknown as typeof fetch;

describe('OpenAICompatConnector.complete', () => {
  it('maps an OpenAI chat response to ChatResult and builds the request', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        model: 'test-model-v2',
        choices: [{ message: { content: 'hello' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      }),
    );
    const c = new OpenAICompatConnector(SPECS.openai, {
      apiKey: 'sk-test',
      fetch: asFetch(fetchMock),
    });
    const res = await c.complete(input);

    expect(res.message.content).toBe('hello');
    expect(res.finishReason).toBe('stop');
    expect(res.usage).toEqual({ inputTokens: 3, outputTokens: 1, totalTokens: 4 });
    expect(res.model).toBe('test-model-v2');

    expect(fetchMock).toHaveBeenCalledOnce();
    const call = fetchMock.mock.calls[0]!;
    expect(call[0]).toBe('https://api.openai.com/v1/chat/completions');
    const init = call[1]!;
    const sentBody = JSON.parse(init.body as string);
    expect(sentBody).toMatchObject({
      model: 'test-model',
      messages: [{ role: 'user', content: 'hi' }],
    });
    expect(init.headers).toMatchObject({ Authorization: 'Bearer sk-test' });
  });

  it('produces an identical result shape when switching provider (openai → groq)', async () => {
    const payload = {
      model: 'llama',
      choices: [{ message: { content: 'yo' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    };
    const openaiFetch = mockFetch(() => jsonResponse(200, payload));
    const groqFetch = mockFetch(() => jsonResponse(200, payload));

    const openai = await new OpenAICompatConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: asFetch(openaiFetch),
    }).complete(input);
    const groq = await new OpenAICompatConnector(SPECS.groq, {
      apiKey: 'k',
      fetch: asFetch(groqFetch),
    }).complete(input);

    expect(groq).toEqual(openai);
    expect(openaiFetch.mock.calls[0]![0]).toContain('api.openai.com');
    expect(groqFetch.mock.calls[0]![0]).toContain('api.groq.com');
  });

  it('throws ConnectorError with rate_limited + retry-after on 429', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(429, { error: { message: 'slow down' } }, { 'retry-after': '30' }),
    );
    const c = new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(fetchMock) });
    try {
      await c.complete(input);
      expect.unreachable('should have thrown');
    } catch (e) {
      const err = e as ConnectorError;
      expect(err).toBeInstanceOf(ConnectorError);
      expect(err.statusCode).toBe(429);
      expect(err.providerCode).toBe('rate_limited');
      const cause = err.cause as Record<string, unknown>;
      expect(cause.retryAfter).toBe('30');
      expect(cause.retryAfterSeconds).toBe(30);
    }
  });

  it('maps 401 → auth_failed and a 400 context-length error → context_length_exceeded', async () => {
    const authFetch = mockFetch(() => jsonResponse(401, { error: { message: 'bad key' } }));
    await expect(
      new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(authFetch) }).complete(input),
    ).rejects.toMatchObject({ providerCode: 'auth_failed' });

    const ctxFetch = mockFetch(() =>
      jsonResponse(400, { error: { message: 'This model’s maximum context length is 8192 tokens' } }),
    );
    await expect(
      new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(ctxFetch) }).complete(input),
    ).rejects.toMatchObject({ providerCode: 'context_length_exceeded' });
  });

  it('throws when a provider without a default base URL gets no baseUrl', () => {
    expect(
      () => new OpenAICompatConnector(SPECS.cloudflare, { apiKey: 'k' }),
    ).toThrow(ConnectorError);
  });
});

describe('OpenAICompatConnector.stream', () => {
  it('parses SSE content deltas, finish reason, and final usage', async () => {
    const sse =
      'data: {"choices":[{"delta":{"content":"He"}}]}\n\n' +
      'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n' +
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n' +
      'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n' +
      'data: [DONE]\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(fetchMock) });

    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream(input)) deltas.push(d);

    expect(deltas.map((d) => d.contentDelta ?? '').join('')).toBe('Hello');
    expect(deltas.some((d) => d.finishReason === 'stop')).toBe(true);
    expect(deltas.find((d) => d.usage)?.usage).toEqual({
      inputTokens: 2,
      outputTokens: 1,
      totalTokens: 3,
    });

    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.stream).toBe(true);
    expect(body.stream_options).toEqual({ include_usage: true });
  });

  it('reads Groq streaming usage from x_groq.usage (baseline-exception)', async () => {
    const sse =
      'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n' +
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"x_groq":{"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}}\n\n' +
      'data: [DONE]\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new OpenAICompatConnector(SPECS.groq, { apiKey: 'k', fetch: asFetch(fetchMock) });

    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream(input)) deltas.push(d);

    expect(deltas.find((d) => d.usage)?.usage).toEqual({
      inputTokens: 5,
      outputTokens: 2,
      totalTokens: 7,
    });
  });

  it('emits one toolCallDelta per parallel tool_call fragment in a single chunk', async () => {
    const sse =
      'data: {"choices":[{"delta":{"tool_calls":[' +
      '{"index":0,"id":"c0","function":{"name":"f0","arguments":"{\\"a\\":1}"}},' +
      '{"index":1,"id":"c1","function":{"name":"f1","arguments":"{\\"b\\":2}"}}' +
      ']}}]}\n\n' +
      'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n' +
      'data: [DONE]\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(fetchMock) });

    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream(input)) deltas.push(d);

    const toolDeltas = deltas.filter((d) => d.toolCallDelta).map((d) => d.toolCallDelta!);
    // both parallel fragments surface (the old code dropped tool_calls[1])
    expect(toolDeltas.map((t) => t.index)).toEqual([0, 1]);
    expect(toolDeltas.map((t) => t.functionName)).toEqual(['f0', 'f1']);
    expect(toolDeltas.map((t) => t.id)).toEqual(['c0', 'c1']);
    expect(deltas.some((d) => d.finishReason === 'tool_calls')).toBe(true);
  });

  it('throws a ConnectorError on a mid-stream {"error": …} frame instead of truncating silently', async () => {
    const sse =
      'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n' +
      'data: {"error":{"message":"upstream exploded","code":503,"type":"server_error"}}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(fetchMock) });

    const seen: ChatStreamDelta[] = [];
    await expect(
      (async () => {
        for await (const d of c.stream(input)) seen.push(d);
      })(),
    ).rejects.toMatchObject({
      name: 'ConnectorError',
      statusCode: 503,
      providerCode: 'provider_unavailable',
      providerMessage: 'upstream exploded',
    });
    // the content before the error frame was still delivered
    expect(seen.map((d) => d.contentDelta ?? '').join('')).toBe('partial');
  });
});

describe('OpenAICompatConnector AbortSignal', () => {
  it('forwards `signal` to fetch and maps an aborted request to a ConnectorError', async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn((_url: string, init?: RequestInit): Promise<Response> => {
      expect(init?.signal).toBe(controller.signal); // signal threaded into RequestInit
      const err = new Error('The operation was aborted');
      err.name = 'AbortError';
      return Promise.reject(err);
    });
    const c = new OpenAICompatConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: fetchMock as unknown as typeof fetch,
    });
    await expect(c.complete({ ...input, signal: controller.signal })).rejects.toMatchObject({
      name: 'ConnectorError',
      providerCode: 'invalid_request',
    });
  });

  it('forwards `signal` to fetch on the streaming path too', async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn(
      (_url: string, init?: RequestInit): Promise<Response> => {
        expect(init?.signal).toBe(controller.signal);
        return Promise.resolve(
          new Response('data: [DONE]\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } }),
        );
      },
    );
    const c = new OpenAICompatConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: fetchMock as unknown as typeof fetch,
    });
    for await (const d of c.stream({ ...input, signal: controller.signal })) void d;
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});

/**
 * A host application can replace Node's process-global undici dispatcher at any
 * time (`setGlobalDispatcher(new Agent().compose(retry(), responseError()))`),
 * after which a non-2xx REJECTS instead of resolving to a `Response`. That is
 * invisible to the `fetch` handed to the connector, so it is handled at the
 * transport seam. Duck-typed here exactly as the library duck-types it.
 */
describe('a fetch that rejects instead of returning a non-2xx', () => {
  function responseErrorRejection(statusCode: number, body: unknown, headers: Record<string, string> = {}) {
    const wrapper = new TypeError('fetch failed');
    (wrapper as { cause?: unknown }).cause = {
      name: 'ResponseError',
      code: 'UND_ERR_RESPONSE',
      statusCode,
      body,
      headers: { 'content-type': 'application/json', ...headers },
    };
    return wrapper;
  }

  function rejectingFetch(err: unknown): typeof fetch {
    return vi.fn((_url: string, _init?: RequestInit) => Promise.reject(err)) as unknown as typeof fetch;
  }

  // Rate limiting is the error an LLM caller most needs classified correctly —
  // it is the difference between backing off and failing the request.
  it('classifies a rejected 429 as rate_limited, not provider_unavailable', async () => {
    const c = new OpenAICompatConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: rejectingFetch(
        responseErrorRejection(429, { error: { message: 'Rate limit reached' } }, { 'retry-after': '20' }),
      ),
    });
    const err = (await c.complete(input).catch((e) => e)) as ConnectorError;
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.statusCode).toBe(429);
    expect(err.providerCode).toBe('rate_limited');
    expect(err.providerMessage).toContain('Rate limit reached');
    expect(JSON.stringify(err.cause)).toContain('20');
  });

  it('classifies a rejected 401 as auth_failed', async () => {
    const c = new OpenAICompatConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: rejectingFetch(responseErrorRejection(401, { error: { message: 'Invalid API key' } })),
    });
    const err = (await c.complete(input).catch((e) => e)) as ConnectorError;
    expect(err.statusCode).toBe(401);
    expect(err.providerCode).toBe('auth_failed');
  });

  it('recovers the status when retry exhaustion carries no body', async () => {
    // undici's RequestRetryError holds `data: { count }` — retry metadata, not the
    // response payload. Reading it as a body would fabricate one.
    const wrapper = new TypeError('fetch failed');
    (wrapper as { cause?: unknown }).cause = {
      name: 'RequestRetryError',
      code: 'UND_ERR_REQ_RETRY',
      statusCode: 503,
      data: { count: 3 },
      headers: {},
    };
    const c = new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: rejectingFetch(wrapper) });
    const err = (await c.complete(input).catch((e) => e)) as ConnectorError;
    expect(err.statusCode).toBe(503);
    expect(JSON.stringify(err.cause)).not.toContain('count');
  });

  it('still reports a genuine transport failure as provider_unavailable', async () => {
    const netErr = new TypeError('fetch failed');
    (netErr as { cause?: unknown }).cause = { name: 'SocketError', code: 'UND_ERR_SOCKET' };
    const c = new OpenAICompatConnector(SPECS.openai, { apiKey: 'k', fetch: rejectingFetch(netErr) });
    const err = (await c.complete(input).catch((e) => e)) as ConnectorError;
    expect(err.statusCode).toBeNull();
    expect(err.providerCode).toBe('provider_unavailable');
    expect(JSON.stringify(err.cause)).toContain('UND_ERR_SOCKET');
  });

  it('never surfaces a key-bearing URL from a leaky transport error', async () => {
    const leaky = new TypeError('request to https://api.openai.com/v1/chat?api_key=sk-SECRET failed');
    const c = new OpenAICompatConnector(SPECS.openai, { apiKey: 'sk-SECRET', fetch: rejectingFetch(leaky) });
    const err = (await c.complete(input).catch((e) => e)) as ConnectorError;
    expect(JSON.stringify({ m: err.message, c: err.cause })).not.toContain('sk-SECRET');
    expect(err.message).toBe('Network error');
  });
});
