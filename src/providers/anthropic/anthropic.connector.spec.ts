import { describe, it, expect, vi } from 'vitest';
import { AnthropicConnector } from './anthropic.connector';
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

describe('AnthropicConnector.complete', () => {
  it('translates the native Messages request/response and hoists the system message', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        model: 'claude-x',
        content: [{ type: 'text', text: 'hello' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 5, output_tokens: 2 },
      }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const input: ChatInput = {
      model: 'claude-x',
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
    expect(res.model).toBe('claude-x');

    const call = fetchMock.mock.calls[0]!;
    expect(call[0]).toBe('https://api.anthropic.com/v1/messages');
    const init = call[1]!;
    const headers = init.headers as Record<string, string>;
    expect(headers['x-api-key']).toBe('k');
    expect(headers['anthropic-version']).toBe('2023-06-01');
    const body = JSON.parse(init.body as string);
    expect(body.system).toBe('be brief');
    expect(body.max_tokens).toBe(100);
    expect(body.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]);
  });

  it('maps tool_use blocks → normalized toolCalls with content null', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        model: 'claude-x',
        content: [{ type: 'tool_use', id: 't1', name: 'get_weather', input: { city: 'SF' } }],
        stop_reason: 'tool_use',
        usage: { input_tokens: 8, output_tokens: 4 },
      }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const res = await c.complete({
      model: 'claude-x',
      messages: [{ role: 'user', content: 'weather?' }],
      tools: [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object' } } }],
      toolChoice: 'required',
    });

    expect(res.message.content).toBeNull();
    expect(res.finishReason).toBe('tool_calls');
    expect(res.message.toolCalls).toEqual([
      { id: 't1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"SF"}' } },
    ]);

    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.tools).toEqual([
      { name: 'get_weather', input_schema: { type: 'object' } },
    ]);
    expect(body.tool_choice).toEqual({ type: 'any' });
    // Anthropic requires max_tokens even when the caller omitted it.
    expect(body.max_tokens).toBe(4096);
  });

  it('rejects an explicit maxOutputTokens below the reasoning budget; inflates the default', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { model: 'm', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    // Explicit cap (100) below the "low" budget (1024) → throw, don't silently inflate.
    await expect(
      c.complete({ model: 'm', messages: [{ role: 'user', content: 'hi' }], maxOutputTokens: 100, reasoning: { effort: 'low' } }),
    ).rejects.toMatchObject({ name: 'ConnectorError', providerCode: 'invalid_request' });

    // No explicit cap → default is inflated above the budget to keep the request valid.
    await c.complete({ model: 'm', messages: [{ role: 'user', content: 'hi' }], reasoning: { effort: 'low' } });
    const body = JSON.parse(fetchMock.mock.calls.at(-1)![1]!.body as string);
    expect(body.max_tokens).toBeGreaterThan(1024);
  });

  it('maps 401 → auth_failed and 529 overloaded → provider_unavailable', async () => {
    const authFetch = mockFetch(() =>
      jsonResponse(401, { type: 'error', error: { type: 'authentication_error', message: 'bad key' } }),
    );
    await expect(
      new AnthropicConnector({ apiKey: 'k', fetch: asFetch(authFetch) }).complete({
        model: 'claude-x',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ name: 'ConnectorError', providerCode: 'auth_failed' });

    const overFetch = mockFetch(() =>
      jsonResponse(529, { type: 'error', error: { type: 'overloaded_error', message: 'overloaded' } }),
    );
    await expect(
      new AnthropicConnector({ apiKey: 'k', fetch: asFetch(overFetch) }).complete({
        model: 'claude-x',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ providerCode: 'provider_unavailable' });
  });

  it('coalesces consecutive tool messages into one user turn of tool_result blocks', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { model: 'claude-x', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: {} }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'claude-x',
      messages: [
        { role: 'user', content: 'run tools' },
        {
          role: 'assistant',
          content: null,
          toolCalls: [
            { id: 'a', type: 'function', function: { name: 'f1', arguments: '{}' } },
            { id: 'b', type: 'function', function: { name: 'f2', arguments: '{}' } },
          ],
        },
        { role: 'tool', toolCallId: 'a', content: 'res-a' },
        { role: 'tool', toolCallId: 'b', content: 'res-b' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    // assistant turn carries two tool_use blocks; the two tool results collapse into ONE user turn.
    expect(body.messages).toHaveLength(3);
    expect(body.messages[2]).toEqual({
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 'a', content: 'res-a' },
        { type: 'tool_result', tool_use_id: 'b', content: 'res-b' },
      ],
    });
    expect(body.messages[1].content).toEqual([
      { type: 'tool_use', id: 'a', name: 'f1', input: {} },
      { type: 'tool_use', id: 'b', name: 'f2', input: {} },
    ]);
  });
});

describe('AnthropicConnector.stream', () => {
  it('parses Anthropic typed SSE events into normalized deltas + usage', async () => {
    const sse =
      'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-x","usage":{"input_tokens":10}}}\n\n' +
      'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"He"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"llo"}}\n\n' +
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n\n' +
      'event: message_stop\ndata: {"type":"message_stop"}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });

    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream({ model: 'claude-x', messages: [{ role: 'user', content: 'hi' }] })) {
      deltas.push(d);
    }

    expect(deltas.map((d) => d.contentDelta ?? '').join('')).toBe('Hello');
    expect(deltas.some((d) => d.finishReason === 'stop')).toBe(true);
    expect(deltas.find((d) => d.usage)?.usage).toEqual({ inputTokens: 10, outputTokens: 3, totalTokens: 13 });

    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.stream).toBe(true);
  });

  it('normalizes tool-call index to 0-based tool-call-relative (past the content-block offset)', async () => {
    // text block at index 0, then two tool_use blocks at indices 1 and 2 →
    // normalized tool indices must be 0 and 1 (matching OpenAI-compat), not 1 and 2.
    const sse =
      'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n' +
      'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"f1"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":1}"}}\n\n' +
      'event: content_block_start\ndata: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t2","name":"f2"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}\n\n' +
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });

    const toolDeltas: ChatStreamDelta[] = [];
    for await (const d of c.stream({ model: 'claude-x', messages: [{ role: 'user', content: 'go' }] })) {
      if (d.toolCallDelta) toolDeltas.push(d);
    }
    // first tool call (block index 1) → normalized index 0; second (block 2) → 1
    const startT1 = toolDeltas.find((d) => d.toolCallDelta!.id === 't1');
    const startT2 = toolDeltas.find((d) => d.toolCallDelta!.id === 't2');
    expect(startT1!.toolCallDelta!.index).toBe(0);
    expect(startT2!.toolCallDelta!.index).toBe(1);
    // the argument-fragment deltas inherit the SAME normalized index as their start
    const argIndexes = toolDeltas
      .filter((d) => d.toolCallDelta!.argumentsDelta !== undefined)
      .map((d) => d.toolCallDelta!.index);
    expect(argIndexes).toEqual([0, 1]);
  });

  it('throws a ConnectorError on a mid-stream error event (e.g. overloaded_error) instead of ending silently', async () => {
    const sse =
      'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-x","usage":{"input_tokens":3}}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"He"}}\n\n' +
      'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n';
    const fetchMock = mockFetch(
      () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });

    const seen: ChatStreamDelta[] = [];
    await expect(
      (async () => {
        for await (const d of c.stream({ model: 'claude-x', messages: [{ role: 'user', content: 'hi' }] })) {
          seen.push(d);
        }
      })(),
    ).rejects.toMatchObject({
      name: 'ConnectorError',
      statusCode: 529,
      providerCode: 'provider_unavailable',
      providerMessage: 'Overloaded',
    });
    expect(seen.map((d) => d.contentDelta ?? '').join('')).toBe('He');
  });
});

describe('Chat facade → anthropic dispatch', () => {
  it('constructs the native Anthropic adapter by provider id', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { model: 'claude-x', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } }),
    );
    const chat = new Chat('anthropic', { apiKey: 'k', fetch: asFetch(fetchMock) });
    expect(chat.id).toBe('anthropic');
    const res = await chat.complete({ model: 'claude-x', messages: [{ role: 'user', content: 'hi' }] });
    expect(res.message.content).toBe('ok');
    expect(fetchMock.mock.calls[0]![0]).toContain('api.anthropic.com');
  });
});
