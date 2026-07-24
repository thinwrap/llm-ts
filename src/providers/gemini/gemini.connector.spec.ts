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

describe('GeminiConnector.complete', () => {
  it('translates generateContent, hoists systemInstruction, and sends x-goog-api-key', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        modelVersion: 'gemini-2.5-flash',
        candidates: [{ content: { parts: [{ text: 'hello' }] }, finishReason: 'STOP' }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 },
      }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const input: ChatInput = {
      model: 'gemini-2.5-flash',
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
    expect(res.model).toBe('gemini-2.5-flash');

    const call = fetchMock.mock.calls[0]!;
    expect(call[0]).toBe('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent');
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
      model: 'gemini-2.5-pro',
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
      { functionDeclarations: [{ name: 'get_weather', parameters: { type: 'object' } }] },
    ]);
    expect(body.toolConfig).toEqual({ functionCallingConfig: { mode: 'ANY' } });
    expect(body.generationConfig.responseMimeType).toBe('application/json');
    expect(body.generationConfig.responseSchema).toEqual({ type: 'object' });
  });

  it('resolves a tool message → functionResponse keyed by function NAME (via the prior tool call)', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }], usageMetadata: {} }),
    );
    const c = new GeminiConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'gemini-2.5-flash',
      messages: [
        { role: 'user', content: 'weather?' },
        { role: 'assistant', content: null, toolCalls: [{ id: 'call-xyz', type: 'function', function: { name: 'get_weather', arguments: '{}' } }] },
        { role: 'tool', toolCallId: 'call-xyz', content: 'sunny' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    // assistant turn → role 'model' with functionCall; tool result → user turn with functionResponse keyed by NAME
    expect(body.contents[1]).toEqual({ role: 'model', parts: [{ functionCall: { name: 'get_weather', args: {} } }] });
    expect(body.contents[2]).toEqual({
      role: 'user',
      parts: [{ functionResponse: { name: 'get_weather', response: { result: 'sunny' } } }],
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
    const res = await c.complete({ model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'x' }] });
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
      model: 'gemini-2.5-flash',
      messages: [{ role: 'user', content: 'hi' }],
      _passthrough: { query: { extra: 'v' } },
    });
    expect(fetchMock.mock.calls[0]![0]).toBe(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?extra=v',
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
        model: 'gemini-2.5-flash',
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
    for await (const d of c.stream({ model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'hi' }] })) {
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
    for await (const d of c.stream({ model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'hi' }] })) {
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
    for await (const d of c.stream({ model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'x' }] })) {
      deltas.push(d);
    }
    expect(deltas.some((d) => d.finishReason === 'content_filter')).toBe(true);
  });

  it('URL-encodes the model id in the streaming path too', async () => {
    const fetchMock = mockFetch(
      () => new Response('data: [DONE]\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } }),
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
        for await (const d of c.stream({ model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'hi' }] })) {
          seen.push(d);
        }
      })(),
    ).rejects.toMatchObject({
      name: 'ConnectorError',
      statusCode: 429,
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
    const res = await chat.complete({ model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'hi' }] });
    expect(res.message.content).toBe('ok');
    expect(fetchMock.mock.calls[0]![0]).toContain('generativelanguage.googleapis.com');
  });
});
