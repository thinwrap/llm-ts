import { describe, it, expect, vi } from 'vitest';
import { GeminiConnector } from './gemini.connector';
import { Chat } from '../../facades/chat.facade';
import { ConnectorError } from '../../types';
import type { ChatInput, ChatStreamDelta } from '../../types';

function jsonResponse(status: number, body: unknown, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

function mockFetch(factory: () => Response) {
  return vi.fn((_url: string, _init?: RequestInit): Promise<Response> => Promise.resolve(factory()));
}

const asFetch = (m: ReturnType<typeof mockFetch>): typeof fetch => m as unknown as typeof fetch;

const expectedFailure = (): never => {
  throw new Error('expected the call to fail with a ConnectorError');
};

describe('GeminiConnector.complete', () => {
  it('translates generateContent, hoists systemInstruction, and sends x-goog-api-key', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        modelVersion: 'gemini-3.8-flash',
        candidates: [{ content: { parts: [{ text: 'hello' }] }, finishReason: 'STOP' }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 },
      }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const input: ChatInput = {
      model: 'gemini-3.8-flash',
      maxOutputTokens: 100,
      messages: [
        { role: 'system', content: 'be brief' },
        { role: 'user', content: 'hi' },
      ],
    };
    const res = await c.complete(input);

    expect(res.message.content).toBe('hello');
    expect(res.finishReason).toBe('stop');
    expect(res.usage).toEqual({ inputTokens: 5, outputTokens: 2, totalTokens: 7 });
    expect(res.model).toBe('gemini-3.8-flash');

    const call = fetchMock.mock.calls[0]!;
    expect(call[0]).toBe('https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent');
    const init = call[1]!;
    expect((init.headers as Record<string, string>)['x-goog-api-key']).toBe('k');
    const body = JSON.parse(init.body as string);
    expect(body.systemInstruction).toEqual({ parts: [{ text: 'be brief' }] });
    expect(body.contents).toEqual([{ role: 'user', parts: [{ text: 'hi' }] }]);
    expect(body.generationConfig.maxOutputTokens).toBe(100);
  });

  it('URL-encodes the model id in the request path (path/query-injection guard)', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }], usageMetadata: {} }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({ model: 'models/x?alt=json#../evil', messages: [{ role: 'user', content: 'hi' }] });
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toBe(
      'https://generativelanguage.googleapis.com/v1beta/models/models%2Fx%3Falt%3Djson%23..%2Fevil:generateContent',
    );
    // the injected separators must not survive raw into the path/query
    expect(url).not.toContain('?alt=json');
    expect(url).not.toContain('#');
  });

  it('maps functionCall parts → toolCalls and tools → functionDeclarations + functionCallingConfig', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        candidates: [{ content: { parts: [{ functionCall: { name: 'get_weather', args: { city: 'SF' } } }] }, finishReason: 'STOP' }],
        usageMetadata: { promptTokenCount: 8, candidatesTokenCount: 4, totalTokenCount: 12 },
      }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const res = await c.complete({
      model: 'gemini-3.8-pro',
      messages: [{ role: 'user', content: 'weather?' }],
      tools: [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object' } } }],
      toolChoice: 'required',
      responseFormat: { type: 'json_schema', jsonSchema: { name: 'w', schema: { type: 'object' } } },
    });

    expect(res.message.content).toBeNull();
    expect(res.finishReason).toBe('tool_calls');
    expect(res.message.toolCalls).toEqual([
      { id: 'call_0', type: 'function', function: { name: 'get_weather', arguments: '{"city":"SF"}' } },
    ]);

    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.tools).toEqual([
      { functionDeclarations: [{ name: 'get_weather', parametersJsonSchema: { type: 'object' } }] },
    ]);
    expect(body.toolConfig).toEqual({ functionCallingConfig: { mode: 'ANY' } });
    expect(body.generationConfig.responseMimeType).toBe('application/json');
    expect(body.generationConfig.responseJsonSchema).toEqual({ type: 'object' });
    expect(body.generationConfig.responseSchema).toBeUndefined();
  });

  it('resolves a tool message → functionResponse keyed by function NAME (via the prior tool call)', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }], usageMetadata: {} }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'gemini-3.8-flash',
      messages: [
        { role: 'user', content: 'weather?' },
        { role: 'assistant', content: null, toolCalls: [{ id: 'call-xyz', type: 'function', function: { name: 'get_weather', arguments: '{}' } }] },
        { role: 'tool', toolCallId: 'call-xyz', content: 'sunny' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    // assistant turn → role 'model' with functionCall; tool result → user turn with functionResponse keyed by NAME.
    // A hand-built turn has no signature, so it carries Google's validator-skip value; its
    // caller-chosen id goes out on both the call and the response.
    expect(body.contents[1]).toEqual({
      role: 'model',
      parts: [
        { functionCall: { id: 'call-xyz', name: 'get_weather', args: {} }, thoughtSignature: 'skip_thought_signature_validator' },
      ],
    });
    expect(body.contents[2]).toEqual({
      role: 'user',
      parts: [{ functionResponse: { id: 'call-xyz', name: 'get_weather', response: { result: 'sunny' } } }],
    });
  });

  it('maps a prompt-level safety block (no candidates, only promptFeedback) → content_filter', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        promptFeedback: { blockReason: 'SAFETY' },
        usageMetadata: { promptTokenCount: 9, totalTokenCount: 9 },
      }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const res = await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] });
    expect(res.message.content).toBeNull();
    expect(res.finishReason).toBe('content_filter');
  });

  it('drops reasoning parts marked thought:true from normalized content (they stay in raw)', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        candidates: [
          {
            content: { parts: [{ text: 'secret chain of thought', thought: true }, { text: 'the answer' }] },
            finishReason: 'STOP',
          },
        ],
        usageMetadata: {},
      }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const res = await c.complete({ model: 'gemini-3-pro', messages: [{ role: 'user', content: 'hi' }] });
    expect(res.message.content).toBe('the answer');
  });

  it('appends _passthrough.query to the request URL', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }], usageMetadata: {} }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'gemini-3.8-flash',
      messages: [{ role: 'user', content: 'hi' }],
      _passthrough: { query: { extra: 'v' } },
    });
    expect(fetchMock.mock.calls[0]![0]).toBe(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?extra=v',
    );
  });

  it('maps 429 with body RetryInfo → rate_limited + cause.retryAfterSeconds', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(429, {
        error: {
          code: 429,
          message: 'quota exceeded',
          status: 'RESOURCE_EXHAUSTED',
          details: [{ '@type': 'type.googleapis.com/google.rpc.RetryInfo', retryDelay: '30s' }],
        },
      }),
    );
    try {
      await new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) }).complete({
        model: 'gemini-3.8-flash',
        messages: [{ role: 'user', content: 'x' }],
      });
      expect.unreachable('should throw');
    } catch (e) {
      const err = e as ConnectorError;
      expect(err.providerCode).toBe('rate_limited');
      expect((err.cause as Record<string, unknown>).retryAfterSeconds).toBe(30);
    }
  });
});

describe('GeminiConnector.stream (real SSE)', () => {
  it('parses incremental SSE chunks into content deltas + finish + usage', async () => {
    const sse =
      'data: {"candidates":[{"content":{"parts":[{"text":"He"}]}}]}\n\n' +
      'data: {"candidates":[{"content":{"parts":[{"text":"llo"}]}}]}\n\n' +
      'data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });

    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'hi' }] })) {
      deltas.push(d);
    }
    expect(deltas.map((d) => d.contentDelta ?? '').join('')).toBe('Hello');
    expect(deltas.some((d) => d.finishReason === 'stop')).toBe(true);
    expect(deltas.find((d) => d.usage)?.usage).toEqual({ inputTokens: 2, outputTokens: 1, totalTokens: 3 });
    expect(fetchMock.mock.calls[0]![0]).toContain(':streamGenerateContent?alt=sse');
  });

  it('assigns distinct tool-call indices and emits a single terminal usage delta', async () => {
    // Two parallel functionCall parts arrive in separate chunks; usageMetadata is
    // cumulative and repeated across chunks. Each tool call must get a distinct
    // `index`, and exactly ONE usage delta (the last, cumulative value) must be
    // emitted so a consumer summing `delta.usage` does not double-count.
    const sse =
      'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}}\n\n' +
      'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_time","args":{"tz":"PT"}}}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":9}}\n\n' +
      'data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":6,"totalTokenCount":11}}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });

    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'hi' }] })) {
      deltas.push(d);
    }

    const toolDeltas = deltas.filter((d) => d.toolCallDelta);
    expect(toolDeltas.map((d) => d.toolCallDelta!.index)).toEqual([0, 1]);
    expect(toolDeltas.map((d) => d.toolCallDelta!.functionName)).toEqual(['get_weather', 'get_time']);

    const usageDeltas = deltas.filter((d) => d.usage);
    expect(usageDeltas).toHaveLength(1);
    expect(usageDeltas[0]!.usage).toEqual({ inputTokens: 5, outputTokens: 6, totalTokens: 11 });
    expect(deltas.some((d) => d.finishReason === 'tool_calls')).toBe(true);
  });

  it('emits a content_filter finish when a stream chunk carries a prompt-level block', async () => {
    const sse = 'data: {"promptFeedback":{"blockReason":"SAFETY"}}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })) {
      deltas.push(d);
    }
    expect(deltas.some((d) => d.finishReason === 'content_filter')).toBe(true);
  });

  it('URL-encodes the model id in the streaming path too', async () => {
    const fetchMock = mockFetch(
      () =>
        new Response('data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}\n\n', {
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
        }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    for await (const d of c.stream({ model: 'a/b:c', messages: [{ role: 'user', content: 'hi' }] })) {
      void d;
    }
    expect(fetchMock.mock.calls[0]![0]).toBe(
      'https://generativelanguage.googleapis.com/v1beta/models/a%2Fb%3Ac:streamGenerateContent?alt=sse',
    );
  });

  it('throws a ConnectorError on a mid-stream {"error": …} frame instead of ending silently', async () => {
    const sse =
      'data: {"candidates":[{"content":{"parts":[{"text":"He"}]}}]}\n\n' +
      'data: {"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });

    const seen: ChatStreamDelta[] = [];
    await expect(
      (async () => {
        for await (const d of c.stream({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'hi' }] })) {
          seen.push(d);
        }
      })(),
    ).rejects.toMatchObject({
      name: 'ConnectorError',
      statusCode: 200, // the status the stream arrived with
      providerCode: 'rate_limited',
      providerMessage: 'quota exceeded',
    });
    expect(seen.map((d) => d.contentDelta ?? '').join('')).toBe('He');
  });
});

describe('Chat facade → gemini dispatch', () => {
  it('constructs the native Gemini adapter by provider id', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 } }),
    );
    const chat = new Chat('gemini', { apiKey: 'k', fetch: asFetch(fetchMock) });
    expect(chat.id).toBe('gemini');
    const res = await chat.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'hi' }] });
    expect(res.message.content).toBe('ok');
    expect(fetchMock.mock.calls[0]![0]).toContain('generativelanguage.googleapis.com');
  });
});

describe('GeminiConnector 2xx without candidates', () => {
  it.each([[{}], [{ usageMetadata: { promptTokenCount: 3, totalTokenCount: 3 }, modelVersion: 'gemini-3.8-flash' }]])(
    'raises the empty-body error for %j (no promptFeedback.blockReason)',
    async (body) => {
      const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(mockFetch(() => jsonResponse(200, body))) });
      const err = (await c
        .complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })
        .catch((e: unknown) => e)) as ConnectorError;
      expect(err).toBeInstanceOf(ConnectorError);
      expect(err.statusCode).toBe(200);
      expect(err.providerCode).toBe('provider_unavailable');
    },
  );
});

const sseOf = (chunks: unknown[]): string => chunks.map((c) => `data: ${JSON.stringify(c)}\r\n\r\n`).join('');
const sseResponse = (chunks: unknown[]) => () =>
  new Response(sseOf(chunks), { status: 200, headers: { 'content-type': 'text/event-stream' } });

async function drain(c: GeminiConnector, input: ChatInput): Promise<ChatStreamDelta[]> {
  const out: ChatStreamDelta[] = [];
  for await (const d of c.stream(input)) out.push(d);
  return out;
}

// Gemini 3 parallel calls: only the first functionCall part carries the signature.
const PARALLEL_CALLS = {
  candidates: [
    {
      content: {
        role: 'model',
        parts: [
          { functionCall: { name: 'get_weather', args: { city: 'SF' } }, thoughtSignature: 'sig-1' },
          { functionCall: { name: 'get_weather', args: { city: 'LA' } } },
        ],
      },
      finishReason: 'STOP',
    },
  ],
  usageMetadata: { promptTokenCount: 8, candidatesTokenCount: 4, totalTokenCount: 12 },
  modelVersion: 'gemini-3.8-flash',
};

describe('GeminiConnector thought signatures', () => {
  it('captures the signature on ToolCall.providerMetadata.gemini with unique index-based ids', async () => {
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(mockFetch(() => jsonResponse(200, PARALLEL_CALLS))) });
    const res = await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'weather?' }] });
    expect(res.message.toolCalls).toEqual([
      { id: 'call_0', type: 'function', function: { name: 'get_weather', arguments: '{"city":"SF"}' }, providerMetadata: { gemini: { thoughtSignature: 'sig-1' } } },
      { id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"LA"}' } },
    ]);
  });

  it('captures the signature from a stream, with distinct ids for parallel same-name calls', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(
        mockFetch(
          sseResponse([
            { candidates: [{ content: { parts: [{ functionCall: { name: 'get_weather', args: { city: 'SF' } }, thoughtSignature: 'sig-1' }] } }] },
            { candidates: [{ content: { parts: [{ functionCall: { name: 'get_weather', args: { city: 'LA' } } }] }, finishReason: 'STOP' }] },
          ]),
        ),
      ),
    });
    const calls = (await drain(c, { model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] }))
      .filter((d) => d.toolCallDelta)
      .map((d) => d.toolCallDelta);
    expect(calls).toEqual([
      { index: 0, id: 'call_0', functionName: 'get_weather', argumentsDelta: '{"city":"SF"}', providerMetadata: { gemini: { thoughtSignature: 'sig-1' } } },
      { index: 1, id: 'call_1', functionName: 'get_weather', argumentsDelta: '{"city":"LA"}' },
    ]);
  });

  it('replays stored signatures and adds no dummy when the turn has one', async () => {
    const fetchMock = mockFetch(() => jsonResponse(200, PARALLEL_CALLS));
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const first = await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'weather?' }] });
    await c.complete({
      model: 'gemini-3.8-flash',
      messages: [
        { role: 'user', content: 'weather?' },
        first.message,
        { role: 'tool', toolCallId: 'call_0', content: 'sunny' },
        { role: 'tool', toolCallId: 'call_1', content: 'rain' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[1]![1]!.body as string);
    expect(body.contents[1]).toEqual({
      role: 'model',
      parts: [
        { functionCall: { name: 'get_weather', args: { city: 'SF' } }, thoughtSignature: 'sig-1' },
        { functionCall: { name: 'get_weather', args: { city: 'LA' } } },
      ],
    });
    // Synthetic ids were never Gemini's, so they stay off the wire.
    expect(body.contents[2].parts).toEqual([
      { functionResponse: { name: 'get_weather', response: { result: 'sunny' } } },
      { functionResponse: { name: 'get_weather', response: { result: 'rain' } } },
    ]);
    expect(JSON.stringify(body)).not.toContain('providerMetadata');
  });

  it('echoes Gemini’s own functionCall ids on the replayed call and the functionResponse', async () => {
    const withIds = structuredClone(PARALLEL_CALLS);
    withIds.candidates[0]!.content.parts[0]!.functionCall = { id: 'fc-a', name: 'get_weather', args: { city: 'SF' } } as never;
    withIds.candidates[0]!.content.parts[1]!.functionCall = { id: 'fc-b', name: 'get_weather', args: { city: 'LA' } } as never;
    const fetchMock = mockFetch(() => jsonResponse(200, withIds));
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const first = await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'weather?' }] });
    expect(first.message.toolCalls?.map((tc) => tc.id)).toEqual(['fc-a', 'fc-b']);
    await c.complete({
      model: 'gemini-3.8-flash',
      messages: [
        { role: 'user', content: 'weather?' },
        first.message,
        { role: 'tool', toolCallId: 'fc-a', content: 'sunny' },
        { role: 'tool', toolCallId: 'fc-b', content: 'rain' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[1]![1]!.body as string);
    expect(body.contents[1].parts.map((p: { functionCall: { id?: string } }) => p.functionCall.id)).toEqual(['fc-a', 'fc-b']);
    expect(body.contents[2].parts).toEqual([
      { functionResponse: { id: 'fc-a', name: 'get_weather', response: { result: 'sunny' } } },
      { functionResponse: { id: 'fc-b', name: 'get_weather', response: { result: 'rain' } } },
    ]);
  });

  it('sets the validator-skip value on the first call of an unsigned turn only, echoing foreign ids but not foreign metadata', async () => {
    const fetchMock = mockFetch(() => jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] }));
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'gemini-3.8-flash',
      messages: [
        { role: 'user', content: 'weather?' },
        {
          role: 'assistant',
          content: null,
          providerMetadata: { anthropic: { content: [{ type: 'thinking', thinking: 't', signature: 'a-sig' }] } },
          toolCalls: [
            { id: 'toolu_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"SF"}' }, providerMetadata: { openai: { x: 1 } } },
            { id: 'toolu_2', type: 'function', function: { name: 'get_time', arguments: '{}' } },
          ],
        },
        { role: 'tool', toolCallId: 'toolu_1', content: 'sunny' },
        { role: 'tool', toolCallId: 'toolu_2', content: 'noon' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.contents[1].parts).toEqual([
      { functionCall: { id: 'toolu_1', name: 'get_weather', args: { city: 'SF' } }, thoughtSignature: 'skip_thought_signature_validator' },
      { functionCall: { id: 'toolu_2', name: 'get_time', args: {} } },
    ]);
    expect(body.contents[2].parts).toEqual([
      { functionResponse: { id: 'toolu_1', name: 'get_weather', response: { result: 'sunny' } } },
      { functionResponse: { id: 'toolu_2', name: 'get_time', response: { result: 'noon' } } },
    ]);
    const wire = JSON.stringify(body);
    expect(wire).not.toContain('a-sig');
    expect(wire).not.toContain('openai');
  });
});

describe('GeminiConnector usage', () => {
  it('counts thoughts as output tokens', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(
        mockFetch(() =>
          jsonResponse(200, {
            candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }],
            usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, thoughtsTokenCount: 40, totalTokenCount: 55 },
          }),
        ),
      ),
    });
    const res = await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] });
    expect(res.usage).toEqual({ inputTokens: 10, outputTokens: 45, totalTokens: 55 });
  });

  it('emits the cumulative stream usage once, thoughts included', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(
        mockFetch(
          sseResponse([
            { candidates: [{ content: { parts: [{ text: 'a' }] } }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 1, thoughtsTokenCount: 30 } },
            { candidates: [{ content: { parts: [{ text: 'b' }] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, thoughtsTokenCount: 30, totalTokenCount: 42 } },
          ]),
        ),
      ),
    });
    const usages = (await drain(c, { model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })).filter((d) => d.usage);
    expect(usages.map((d) => d.usage)).toEqual([{ inputTokens: 10, outputTokens: 32, totalTokens: 42 }]);
  });
});

describe('GeminiConnector finish reasons and errors', () => {
  it.each([
    ['MALFORMED_FUNCTION_CALL', 'unknown'],
    ['UNEXPECTED_TOOL_CALL', 'unknown'],
    ['IMAGE_SAFETY', 'content_filter'],
    ['MAX_TOKENS', 'length'],
  ])('maps finishReason %s → %s', async (finishReason, expected) => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'x' }] }, finishReason }] }))),
    });
    expect((await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })).finishReason).toBe(expected);
  });

  const fail = (status: number, error: Record<string, unknown>, headers?: Record<string, string>) =>
    new GeminiConnector({ apiKey: 'k', fetch: asFetch(mockFetch(() => jsonResponse(status, { error }, headers))) })
      .complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })
      .then(expectedFailure, (e: unknown) => e as ConnectorError);

  it('maps a 400 API_KEY_INVALID to auth_failed', async () => {
    const err = await fail(400, {
      code: 400,
      message: 'API key not valid. Please pass a valid API key.',
      status: 'INVALID_ARGUMENT',
      details: [{ '@type': 'type.googleapis.com/google.rpc.ErrorInfo', reason: 'API_KEY_INVALID', domain: 'googleapis.com' }],
    });
    expect(err.providerCode).toBe('auth_failed');
    expect(err.statusCode).toBe(400);
  });

  it.each([
    ['The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).', 'context_length_exceeded'],
    ["* GenerateContentRequest.contents: contents is not specified", 'invalid_request'],
    ['Invalid value at cached_content (context cache not found)', 'invalid_request'],
    ['Request payload size too long', 'invalid_request'],
  ])('classifies a 400 "%s" as %s', async (message, expected) => {
    expect((await fail(400, { code: 400, message, status: 'INVALID_ARGUMENT' })).providerCode).toBe(expected);
  });

  it.each([
    [402, 'rate_limited'],
    [413, 'invalid_request'],
    [408, 'provider_unavailable'],
    [504, 'provider_unavailable'],
  ])('maps HTTP %i → %s', async (status, expected) => {
    expect((await fail(status, { code: status, message: 'x' })).providerCode).toBe(expected);
  });

  it('ignores a blank Retry-After and falls back to the body RetryInfo', async () => {
    const err = await fail(
      429,
      { code: 429, message: 'quota', status: 'RESOURCE_EXHAUSTED', details: [{ retryDelay: '7s' }] },
      { 'retry-after': '' },
    );
    expect((err.cause as Record<string, unknown>).retryAfterSeconds).toBe(7);
  });

  it('honours a mid-stream error frame that has only a status name', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(sseResponse([{ error: { message: 'Resource has been exhausted', status: 'RESOURCE_EXHAUSTED' } }]))),
    });
    await expect(drain(c, { model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })).rejects.toMatchObject({
      statusCode: 200,
      providerCode: 'rate_limited',
    });
  });

  it('raises when the stream ends before a finishReason', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(sseResponse([{ candidates: [{ content: { parts: [{ text: 'partial' }] } }] }]))),
    });
    await expect(drain(c, { model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })).rejects.toMatchObject({
      name: 'ConnectorError',
      providerCode: 'provider_unavailable',
    });
  });

  it('raises on a 2xx JSON body for a streaming request', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'x' }] }, finishReason: 'STOP' }] }))),
    });
    await expect(drain(c, { model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })).rejects.toMatchObject({
      statusCode: 200,
      providerCode: 'provider_unavailable',
    });
  });
});

describe('GeminiConnector review-5', () => {
  it('names each tool result after the nearest preceding call, even when synthetic ids repeat across turns', async () => {
    const fetchMock = mockFetch(() => jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] }));
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'gemini-3.8-flash',
      messages: [
        { role: 'user', content: 'q' },
        { role: 'assistant', content: null, toolCalls: [{ id: 'call_0', type: 'function', function: { name: 'get_weather', arguments: '{}' } }] },
        { role: 'tool', toolCallId: 'call_0', content: 'sunny' },
        { role: 'assistant', content: null, toolCalls: [{ id: 'call_0', type: 'function', function: { name: 'get_time', arguments: '{}' } }] },
        { role: 'tool', toolCallId: 'call_0', content: 'noon' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    const names = body.contents
      .filter((c: { role: string; parts: Array<{ functionResponse?: unknown }> }) => c.parts[0]?.functionResponse)
      .map((c: { parts: Array<{ functionResponse: { name: string } }> }) => c.parts[0]!.functionResponse.name);
    expect(names).toEqual(['get_weather', 'get_time']);
  });

  it('synthesizes an id for an empty functionCall.id', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, { candidates: [{ content: { parts: [{ functionCall: { id: '', name: 'f', args: {} } }] }, finishReason: 'STOP' }] }))),
    });
    const res = await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] });
    expect(res.message.toolCalls?.[0]?.id).toBe('call_0');
  });

  it('reports stream usage once, on the finish delta, including a restatement after the finish', async () => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(
        mockFetch(
          sseResponse([
            { candidates: [{ content: { parts: [{ text: 'a' }] } }], usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 1 } },
            { candidates: [{ content: { parts: [{ text: 'b' }] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 2 } },
            { usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 2, thoughtsTokenCount: 4 } },
          ]),
        ),
      ),
    });
    const deltas = await drain(c, { model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] });
    const withUsage = deltas.filter((d) => d.usage);
    expect(withUsage).toHaveLength(1);
    expect(withUsage[0]!.finishReason).toBe('stop');
    expect(withUsage[0]!.usage).toEqual({ inputTokens: 3, outputTokens: 6, totalTokens: 9 });
    expect(deltas.at(-1)).toBe(withUsage[0]);
  });

  it.each([
    ['LANGUAGE', 'unknown'],
    ['NO_IMAGE', 'unknown'],
    ['IMAGE_OTHER', 'unknown'],
    ['IMAGE_PROHIBITED_CONTENT', 'content_filter'],
    ['IMAGE_RECITATION', 'content_filter'],
    ['CONTINUATION', 'length'],
    ['TOO_MANY_TOOL_CALLS', 'unknown'],
    ['FINISH_REASON_UNSPECIFIED', 'unknown'],
  ])('maps the SDK FinishReason %s → %s', async (finishReason, expected) => {
    const c = new GeminiConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'x' }] }, finishReason }] }))),
    });
    expect((await c.complete({ model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })).finishReason).toBe(expected);
  });

  it('classifies a mid-stream error frame with neither code nor status as unknown, keeping the stream status', async () => {
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(mockFetch(sseResponse([{ error: { message: 'odd failure' } }]))) });
    await expect(drain(c, { model: 'gemini-3.8-flash', messages: [{ role: 'user', content: 'x' }] })).rejects.toMatchObject({
      statusCode: 200,
      providerCode: 'unknown',
    });
  });
});
