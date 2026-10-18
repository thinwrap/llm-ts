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

const expectedFailure = (): never => {
  throw new Error('expected the call to fail with a ConnectorError');
};

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
      statusCode: 200, // the status the stream arrived with
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

describe('max output tokens field', () => {
  const withMax: ChatInput = { ...input, maxOutputTokens: 256 };
  const ok = () => jsonResponse(200, { choices: [{ message: { content: 'x' }, finish_reason: 'stop' }] });

  it.each(['openai', 'azure-openai'] as const)(
    'sends max_completion_tokens (never max_tokens) for %s',
    async (id) => {
      const fetchMock = mockFetch(ok);
      await new OpenAICompatConnector(SPECS[id], {
        apiKey: 'k',
        baseUrl: 'https://res.openai.azure.com/openai/v1',
        fetch: asFetch(fetchMock),
      }).complete(withMax);
      const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
      expect(body.max_completion_tokens).toBe(256);
      expect(body).not.toHaveProperty('max_tokens');
    },
  );

  it('keeps max_tokens for other OpenAI-compatible providers (groq)', async () => {
    const fetchMock = mockFetch(ok);
    await new OpenAICompatConnector(SPECS.groq, { apiKey: 'k', fetch: asFetch(fetchMock) }).complete(withMax);
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.max_tokens).toBe(256);
    expect(body).not.toHaveProperty('max_completion_tokens');
  });
});

describe('2xx bodies that are not a completion', () => {
  const complete = (body: unknown) =>
    new OpenAICompatConnector(SPECS.openrouter, {
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, body))),
    })
      .complete(input)
      .catch((e: unknown) => e) as Promise<ConnectorError>;

  it('raises an OpenRouter 200 carrying only an error object, classified by error.code but keeping the 200', async () => {
    const err = await complete({ error: { code: 429, message: 'Rate limit exceeded upstream' } });
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('rate_limited');
    expect(err.message).toBe('Rate limit exceeded upstream');
  });

  it('raises an OpenRouter 200 whose choice has finish_reason "error" beside partial content', async () => {
    const body = {
      id: 'gen-1',
      model: 'openai/gpt-4o',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: 'Partial ans' },
          finish_reason: 'error',
          error: { code: 502, message: 'Provider disconnected' },
        },
      ],
    };
    const err = await complete(body);
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('provider_unavailable');
    expect(err.message).toBe('Provider disconnected');
    // the partial content stays inspectable
    expect((err.cause as Record<string, unknown>).raw).toEqual(body);
  });

  it('classifies a string error code by its name', async () => {
    const err = await complete({ error: { code: 'server_error', message: 'boom' } });
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('provider_unavailable');
  });

  it('classifies an error object with no usable code or type as unknown', async () => {
    const err = await complete({ error: { message: 'something went wrong' } });
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('unknown');
  });

  it('raises finish_reason "error" even when the choice carries no error object, as unknown', async () => {
    const err = await complete({ choices: [{ message: { content: '' }, finish_reason: 'error' }] });
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('unknown');
  });

  it.each([[{}], [{ choices: [] }], [{ choices: null }]])('raises the empty-body error for %j', async (body) => {
    const err = await complete(body);
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('provider_unavailable');
  });
});

describe('chunk-list message content (Mistral reasoning)', () => {
  it('returns only the text chunks of an array-shaped message.content', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        model: 'magistral-medium-latest',
        choices: [
          {
            message: {
              role: 'assistant',
              content: [
                { type: 'thinking', thinking: [{ type: 'text', text: 'Let me compute 2+2.' }] },
                { type: 'text', text: 'The answer ' },
                { type: 'text', text: 'is 4.' },
              ],
            },
            finish_reason: 'stop',
          },
        ],
      }),
    );
    const res = await new OpenAICompatConnector(SPECS.mistral, { apiKey: 'k', fetch: asFetch(fetchMock) }).complete(input);
    expect(res.message.content).toBe('The answer is 4.');
  });

  it('treats a chunk list with only thinking as no content', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        choices: [
          {
            message: { content: [{ type: 'thinking', thinking: [{ type: 'text', text: 'hmm' }] }] },
            finish_reason: 'length',
          },
        ],
      }),
    );
    const res = await new OpenAICompatConnector(SPECS.mistral, { apiKey: 'k', fetch: asFetch(fetchMock) }).complete(input);
    expect(res.message.content).toBeNull();
  });

  it('streams only answer text across the thinking → transition → string delta phases', async () => {
    const frames = [
      { choices: [{ delta: { content: [{ type: 'thinking', thinking: [{ type: 'text', text: 'Thinking…' }] }] } }] },
      {
        choices: [
          {
            delta: {
              content: [
                { type: 'thinking', thinking: [{ type: 'text', text: ' done.' }] },
                { type: 'text', text: 'The answer' },
              ],
            },
          },
        ],
      },
      { choices: [{ delta: { content: ' is 4.' } }] },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ];
    const sse = frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join('') + 'data: [DONE]\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const deltas: ChatStreamDelta[] = [];
    for await (const d of new OpenAICompatConnector(SPECS.mistral, { apiKey: 'k', fetch: asFetch(fetchMock) }).stream(input)) {
      deltas.push(d);
    }
    expect(deltas.filter((d) => d.contentDelta !== undefined).map((d) => d.contentDelta)).toEqual(['The answer', ' is 4.']);
    expect(deltas.some((d) => d.finishReason === 'stop')).toBe(true);
  });
});

describe('OpenAICompatConnector stream integrity', () => {
  const streamOf = (body: string, contentType = 'text/event-stream') =>
    new OpenAICompatConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => new Response(body, { status: 200, headers: { 'content-type': contentType } }))),
    });
  const drain = async (c: OpenAICompatConnector): Promise<ChatStreamDelta[]> => {
    const out: ChatStreamDelta[] = [];
    for await (const d of c.stream(input)) out.push(d);
    return out;
  };

  it('raises when the body ends with neither [DONE] nor a finish_reason', async () => {
    const sse = 'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n';
    await expect(drain(streamOf(sse))).rejects.toMatchObject({ name: 'ConnectorError', providerCode: 'provider_unavailable' });
  });

  it('accepts a stream that ends on a finish_reason without [DONE]', async () => {
    const sse = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n';
    const deltas = await drain(streamOf(sse));
    expect(deltas.some((d) => d.finishReason === 'stop')).toBe(true);
  });

  it('raises on a 2xx JSON body for a streaming request', async () => {
    const json = JSON.stringify({ choices: [{ message: { content: 'x' }, finish_reason: 'stop' }] });
    await expect(drain(streamOf(json, 'application/json'))).rejects.toMatchObject({
      name: 'ConnectorError',
      statusCode: 200,
      providerCode: 'provider_unavailable',
    });
  });

  it('keeps every line of a CRLF multi-line data event', async () => {
    const sse = 'data: {"choices":[{"delta":\r\ndata: {"content":"Hi"},"finish_reason":"stop"}]}\r\n\r\ndata: [DONE]\r\n\r\n';
    const deltas = await drain(streamOf(sse));
    expect(deltas.map((d) => d.contentDelta ?? '').join('')).toBe('Hi');
  });

  it('honours the string code of a mid-stream error frame', async () => {
    const sse = 'data: {"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}\n\n';
    await expect(drain(streamOf(sse))).rejects.toMatchObject({ statusCode: 200, providerCode: 'rate_limited' });
  });
});

describe('OpenAICompatConnector status classification', () => {
  const fail = (status: number, body: unknown, headers?: Record<string, string>) =>
    new OpenAICompatConnector(SPECS.openrouter, { apiKey: 'k', fetch: asFetch(mockFetch(() => jsonResponse(status, body, headers))) })
      .complete(input)
      .then(expectedFailure, (e: unknown) => e as ConnectorError);

  it.each([
    [402, { error: { code: 402, message: 'Insufficient credits' } }, 'rate_limited'],
    [413, { error: { message: 'Request Entity Too Large' } }, 'invalid_request'],
    [408, { error: { message: 'Request timeout' } }, 'provider_unavailable'],
    [400, { error: { code: 'context_length_exceeded', message: 'Your input exceeds the window.' } }, 'context_length_exceeded'],
  ])('maps %i → %s', async (status, body, expected) => {
    expect((await fail(status, body)).providerCode).toBe(expected);
  });

  it.each([
    ['', undefined],
    ['-5', undefined],
    ['abc', undefined],
    ['7', 7],
    ['1.5', undefined],
  ])('reads Retry-After %j as retryAfterSeconds %j', async (header, expected) => {
    const err = await fail(429, { error: { message: 'slow down' } }, { 'retry-after': header });
    expect((err.cause as Record<string, unknown>).retryAfterSeconds).toBe(expected);
  });
});

describe('OpenAICompatConnector review-5', () => {
  it('reads a message naming both "content" and "filter" as content_filtered', async () => {
    const err = (await new OpenAICompatConnector(SPECS["azure-openai"], {
      apiKey: 'k',
      baseUrl: 'https://r.openai.azure.com/openai/v1',
      fetch: asFetch(
        mockFetch(() =>
          jsonResponse(400, { error: { code: 'content_filter', message: "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." } }),
        ),
      ),
    })
      .complete(input)
      .then(expectedFailure, (e: unknown) => e as ConnectorError)) as ConnectorError;
    expect(err.providerCode).toBe('content_filtered');
  });

  it('classifies a mid-stream error frame with no usable code or type as unknown, keeping the stream status', async () => {
    const c = new OpenAICompatConnector(SPECS.openrouter, {
      apiKey: 'k',
      fetch: asFetch(
        mockFetch(() => new Response('data: {"error":{"message":"something broke"}}\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } })),
      ),
    });
    await expect(
      (async () => {
        for await (const d of c.stream(input)) void d;
      })(),
    ).rejects.toMatchObject({ statusCode: 200, providerCode: 'unknown', providerMessage: 'something broke' });
  });
});
