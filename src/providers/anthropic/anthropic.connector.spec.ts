import { describe, it, expect, vi } from 'vitest';
import { AnthropicConnector } from './anthropic.connector';
import { Chat } from '../../facades/chat.facade';
import { ConnectorError } from '../../types';
import type { ChatInput, ChatMessage, ChatStreamDelta } from '../../types';

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
    expect(body.max_tokens).toBe(16000);
  });

  it.each(['low', 'medium', 'high'] as const)(
    'maps reasoning.effort %s to adaptive thinking + output_config.effort, with no budget and no sampling params',
    async (effort) => {
      const fetchMock = mockFetch(() =>
        jsonResponse(200, { model: 'm', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } }),
      );
      const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
      // The same shape for every model id: no model-family switch.
      for (const model of ['claude-opus-5-5', 'claude-haiku-4-5']) {
        await c.complete({
          model,
          messages: [{ role: 'user', content: 'hi' }],
          maxOutputTokens: 100,
          temperature: 0.2,
          topP: 0.9,
          reasoning: { effort },
        });
        const body = JSON.parse(fetchMock.mock.calls.at(-1)![1]!.body as string);
        expect(body.thinking).toEqual({ type: 'adaptive' });
        expect(body.output_config).toEqual({ effort });
        // A small explicit cap is sent as-is: adaptive thinking has no budget to exceed.
        expect(body.max_tokens).toBe(100);
        expect(body).not.toHaveProperty('temperature');
        expect(body).not.toHaveProperty('top_p');
        expect(JSON.stringify(body)).not.toContain('budget_tokens');
      }
    },
  );

  it('keeps the default max_tokens under reasoning and omits thinking without it', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { model: 'm', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'hi' }], reasoning: { effort: 'high' } });
    expect(JSON.parse(fetchMock.mock.calls.at(-1)![1]!.body as string).max_tokens).toBe(16000);

    await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'hi' }], temperature: 0.5 });
    const plain = JSON.parse(fetchMock.mock.calls.at(-1)![1]!.body as string);
    expect(plain).not.toHaveProperty('thinking');
    expect(plain).not.toHaveProperty('output_config');
    expect(plain.temperature).toBe(0.5);
  });

  it('lets a budget-mode caller send thinking through _passthrough.body', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { model: 'm', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } }),
    );
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'claude-haiku-4-5',
      maxOutputTokens: 8000,
      messages: [{ role: 'user', content: 'hi' }],
      _passthrough: { body: { thinking: { type: 'enabled', budget_tokens: 4000 } } },
    });
    const body = JSON.parse(fetchMock.mock.calls.at(-1)![1]!.body as string);
    expect(body.thinking).toEqual({ type: 'enabled', budget_tokens: 4000 });
    expect(body).not.toHaveProperty('output_config');
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
      statusCode: 200, // the status the stream arrived with
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

describe('AnthropicConnector 2xx without a message', () => {
  it.each([[{}], [{ id: 'msg_1', type: 'message', role: 'assistant', content: null }]])(
    'raises the empty-body error for %j',
    async (body) => {
      const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(mockFetch(() => jsonResponse(200, body))) });
      const err = (await c
        .complete({ model: 'claude-sonnet-4-6', messages: [{ role: 'user', content: 'x' }] })
        .catch((e: unknown) => e)) as ConnectorError;
      expect(err).toBeInstanceOf(ConnectorError);
      expect(err.statusCode).toBe(200);
      expect(err.providerCode).toBe('provider_unavailable');
    },
  );
});

// A tool-loop turn as the Messages API returns it: a thinking block (with its
// signature), a redacted thinking block, text, and a tool call.
const TURN_CONTENT = [
  { type: 'thinking', thinking: 'Need the weather first.', signature: 'sig-abc' },
  { type: 'redacted_thinking', data: 'ENCRYPTED' },
  { type: 'text', text: 'Checking.' },
  { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: { city: 'SF' } },
];

function turnResponse(): Response {
  return jsonResponse(200, {
    model: 'claude-opus-5-5',
    content: TURN_CONTENT,
    stop_reason: 'tool_use',
    usage: { input_tokens: 10, output_tokens: 20 },
  });
}

const sse = (events: unknown[]): string =>
  events.map((e) => `event: ${(e as { type: string }).type}\ndata: ${JSON.stringify(e)}\n\n`).join('');

async function drain(c: AnthropicConnector, input: ChatInput): Promise<ChatStreamDelta[]> {
  const out: ChatStreamDelta[] = [];
  for await (const d of c.stream(input)) out.push(d);
  return out;
}

describe('AnthropicConnector providerMetadata (thinking round-trip)', () => {
  it('stores the response content blocks verbatim on message.providerMetadata.anthropic', async () => {
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(mockFetch(turnResponse)) });
    const res = await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'weather?' }] });
    expect(res.message.providerMetadata).toEqual({ anthropic: { content: TURN_CONTENT } });
    expect(res.message.content).toBe('Checking.');
    expect(res.message.toolCalls?.map((tc) => tc.id)).toEqual(['toolu_1']);
  });

  it('replays a pushed result.message verbatim, thinking blocks included', async () => {
    const fetchMock = mockFetch(turnResponse);
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const first = await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'weather?' }] });
    await c.complete({
      model: 'claude-opus-5-5',
      messages: [
        { role: 'user', content: 'weather?' },
        first.message,
        { role: 'tool', toolCallId: 'toolu_1', content: 'sunny' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[1]![1]!.body as string);
    expect(body.messages[1]).toEqual({ role: 'assistant', content: TURN_CONTENT });
    expect(JSON.stringify(body)).not.toContain('providerMetadata');
  });

  it.each([
    ['edited text', { content: 'Something else.' }],
    ['different tool-call ids', { toolCalls: [{ id: 'toolu_9', type: 'function', function: { name: 'get_weather', arguments: '{"city":"SF"}' } }] }],
    ['an extra tool call', {
      toolCalls: [
        { id: 'toolu_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"SF"}' } },
        { id: 'toolu_2', type: 'function', function: { name: 'get_weather', arguments: '{"city":"LA"}' } },
      ],
    }],
    ['a non-list content slot', { providerMetadata: { anthropic: { content: 'nope' } } }],
  ] as const)('rebuilds the turn (dropping thinking) when the stored blocks no longer match: %s', async (_label, edit) => {
    const fetchMock = mockFetch(turnResponse);
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    const first = await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'weather?' }] });
    const edited = { ...first.message, ...edit } as ChatMessage;
    await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'weather?' }, edited] });
    const sent = JSON.parse(fetchMock.mock.calls[1]![1]!.body as string).messages[1];
    const types = (sent.content as Array<{ type: string }>).map((b) => b.type);
    expect(types).not.toContain('thinking');
    expect(types).not.toContain('redacted_thinking');
    expect(sent.content).toEqual([
      ...(typeof edited.content === 'string' ? [{ type: 'text', text: edited.content }] : []),
      ...(edited.toolCalls ?? []).map((tc) => ({ type: 'tool_use', id: tc.id, name: tc.function.name, input: JSON.parse(tc.function.arguments) })),
    ]);
  });

  it('ignores another provider’s metadata and never sends it', async () => {
    const fetchMock = mockFetch(turnResponse);
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'claude-opus-5-5',
      messages: [
        { role: 'user', content: 'weather?' },
        {
          role: 'assistant',
          content: null,
          providerMetadata: { gemini: { note: 'x' } },
          toolCalls: [
            { id: 'call_0', type: 'function', function: { name: 'get_weather', arguments: '{}' }, providerMetadata: { gemini: { thoughtSignature: 'g-sig' } } },
          ],
        },
        { role: 'tool', toolCallId: 'call_0', content: 'sunny' },
      ],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.messages[1]).toEqual({ role: 'assistant', content: [{ type: 'tool_use', id: 'call_0', name: 'get_weather', input: {} }] });
    expect(JSON.stringify(body)).not.toContain('g-sig');
  });

  it('rebuilds the same content array from a stream and emits it on the terminal delta', async () => {
    const events = [
      { type: 'message_start', message: { model: 'claude-opus-5-5', usage: { input_tokens: 10, output_tokens: 1 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Need the weather ' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'first.' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'sig-abc' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'content_block_start', index: 1, content_block: { type: 'redacted_thinking', data: 'ENCRYPTED' } },
      { type: 'content_block_stop', index: 1 },
      { type: 'content_block_start', index: 2, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 2, delta: { type: 'text_delta', text: 'Checking.' } },
      { type: 'content_block_stop', index: 2 },
      { type: 'content_block_start', index: 3, content_block: { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: {} } },
      { type: 'content_block_delta', index: 3, delta: { type: 'input_json_delta', partial_json: '{"city":' } },
      { type: 'content_block_delta', index: 3, delta: { type: 'input_json_delta', partial_json: '"SF"}' } },
      { type: 'content_block_stop', index: 3 },
      { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 20 } },
      { type: 'message_stop' },
    ];
    const c = new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => new Response(sse(events), { status: 200, headers: { 'content-type': 'text/event-stream' } }))),
    });
    const deltas = await drain(c, { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'weather?' }] });
    const terminal = deltas.find((d) => d.finishReason !== undefined)!;
    expect(terminal.finishReason).toBe('tool_calls');
    expect(terminal.providerMetadata).toEqual({ anthropic: { content: TURN_CONTENT } });
    expect(deltas.filter((d) => d.providerMetadata !== undefined)).toHaveLength(1);
  });
});

describe('AnthropicConnector usage', () => {
  it('folds cache reads and writes into inputTokens', async () => {
    const c = new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(
        mockFetch(() =>
          jsonResponse(200, {
            content: [{ type: 'text', text: 'ok' }],
            stop_reason: 'end_turn',
            usage: { input_tokens: 50, cache_read_input_tokens: 100000, cache_creation_input_tokens: 200, output_tokens: 7 },
          }),
        ),
      ),
    });
    const res = await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] });
    expect(res.usage).toEqual({ inputTokens: 100250, outputTokens: 7, totalTokens: 100257 });
  });

  it('reads stream usage from message_start, superseded by cumulative message_delta counts', async () => {
    const events = [
      { type: 'message_start', message: { usage: { input_tokens: 5, cache_read_input_tokens: 90, output_tokens: 1 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hi' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 5, cache_read_input_tokens: 90, cache_creation_input_tokens: 5, output_tokens: 12 } },
    ];
    const c = new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => new Response(sse(events), { status: 200, headers: { 'content-type': 'text/event-stream' } }))),
    });
    const deltas = await drain(c, { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] });
    expect(deltas.find((d) => d.usage)?.usage).toEqual({ inputTokens: 100, outputTokens: 12, totalTokens: 112 });
  });
});

describe('AnthropicConnector finish reasons', () => {
  it.each([
    ['pause_turn', 'unknown'],
    ['refusal', 'content_filter'],
    ['model_context_window_exceeded', 'length'],
  ])('maps stop_reason %s → %s', async (stopReason, expected) => {
    const c = new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, { content: [{ type: 'text', text: 'x' }], stop_reason: stopReason, usage: {} }))),
    });
    const res = await c.complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] });
    expect(res.finishReason).toBe(expected);
  });
});

describe('AnthropicConnector error classification', () => {
  const fail = (status: number, type: string, message: string, headers?: Record<string, string>) =>
    new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(status, { type: 'error', error: { type, message } }, headers))),
    })
      .complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] })
      .then(expectedFailure, (e: unknown) => e as ConnectorError);

  it.each([
    [400, 'invalid_request_error', 'prompt is too long: 210000 tokens > 200000 maximum', 'context_length_exceeded'],
    [400, 'invalid_request_error', 'input length and `max_tokens` exceed context limit: 190000 + 20000 > 200000', 'context_length_exceeded'],
    [400, 'invalid_request_error', 'max_tokens: Field required', 'invalid_request'],
    [400, 'invalid_request_error', 'tool_choice: type "tool" and "any" are not supported for this model.', 'invalid_request'],
    [400, 'invalid_request_error', 'adaptive thinking is not supported on this model', 'invalid_request'],
    [402, 'billing_error', 'Your credit balance is too low', 'rate_limited'],
    [413, 'request_too_large', 'Request exceeds the maximum allowed number of bytes.', 'invalid_request'],
    [504, 'timeout_error', 'Request timed out', 'provider_unavailable'],
  ])('maps %i %s (%s) → %s', async (status, type, message, expected) => {
    expect((await fail(status, type, message)).providerCode).toBe(expected);
  });

  it.each([
    ['request_too_large', 413, 'invalid_request'],
    ['billing_error', 402, 'rate_limited'],
    ['timeout_error', 504, 'provider_unavailable'],
    ['rate_limit_error', 429, 'rate_limited'],
  ])('maps a mid-stream %s error event (HTTP %i when not streamed) → %s on a 200 stream', async (type, _status, code) => {
    const body = sse([{ type: 'error', error: { type, message: 'boom' } }]);
    const c = new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } }))),
    });
    await expect(drain(c, { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] })).rejects.toMatchObject({
      statusCode: 200,
      providerCode: code,
    });
  });

  it.each([
    ['', undefined],
    ['   ', undefined],
    ['-3', undefined],
    ['12', 12],
  ])('reads Retry-After %j as retryAfterSeconds %j', async (header, expected) => {
    const err = await fail(429, 'rate_limit_error', 'slow down', { 'retry-after': header });
    const cause = err.cause as Record<string, unknown>;
    expect(cause.retryAfterSeconds).toBe(expected);
  });
});

describe('AnthropicConnector stream integrity', () => {
  it('raises when the stream ends before the message_delta stop', async () => {
    const body = sse([
      { type: 'message_start', message: { usage: { input_tokens: 3 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hel' } },
    ]);
    const c = new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } }))),
    });
    await expect(drain(c, { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] })).rejects.toMatchObject({
      name: 'ConnectorError',
      providerCode: 'provider_unavailable',
    });
  });

  it('raises on a 2xx that is not an event stream', async () => {
    const c = new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, { content: [{ type: 'text', text: 'x' }], stop_reason: 'end_turn' }))),
    });
    await expect(drain(c, { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] })).rejects.toMatchObject({
      name: 'ConnectorError',
      statusCode: 200,
      providerCode: 'provider_unavailable',
    });
  });
});

describe('AnthropicConnector empty text parts', () => {
  it('drops empty text parts, which the Messages API rejects', async () => {
    const fetchMock = mockFetch(() => jsonResponse(200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: {} }));
    const c = new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) });
    await c.complete({
      model: 'claude-opus-5-5',
      messages: [{ role: 'user', content: [{ type: 'text', text: '' }, { type: 'text', text: 'hi' }] }],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.messages[0].content).toEqual([{ type: 'text', text: 'hi' }]);
  });
});

describe('AnthropicConnector review-5 replay rules', () => {
  const thinkingTurn = (id: string, text: string, input: Record<string, unknown>) => ({
    role: 'assistant' as const,
    content: text,
    toolCalls: [{ id, type: 'function' as const, function: { name: 'get_weather', arguments: JSON.stringify(input) } }],
    providerMetadata: {
      anthropic: {
        content: [
          { type: 'thinking', thinking: `why ${id}`, signature: `sig-${id}` },
          { type: 'text', text },
          { type: 'tool_use', id, name: 'get_weather', input },
        ],
      },
    },
  });
  const send = async (messages: ChatMessage[]) => {
    const fetchMock = mockFetch(() => jsonResponse(200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: {} }));
    await new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) }).complete({ model: 'claude-opus-5-5', messages });
    const sent = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string).messages as Array<{ role: string; content: Array<{ type: string }> }>;
    return sent.filter((m) => m.role === 'assistant').map((m) => m.content.map((b) => b.type));
  };

  it('rebuilds every assistant turn after one that is rebuilt, so no later thinking block is voided', async () => {
    const t1 = { ...thinkingTurn('toolu_1', 'One.', { city: 'SF' }), content: 'Edited one.' };
    const t2 = thinkingTurn('toolu_2', 'Two.', { city: 'LA' });
    const types = await send([
      { role: 'user', content: 'q' },
      t1,
      { role: 'tool', toolCallId: 'toolu_1', content: 'r1' },
      t2,
      { role: 'tool', toolCallId: 'toolu_2', content: 'r2' },
    ]);
    expect(types).toEqual([
      ['text', 'tool_use'],
      ['text', 'tool_use'],
    ]);
  });

  it('keeps earlier turns verbatim when only a later one is rebuilt', async () => {
    const t1 = thinkingTurn('toolu_1', 'One.', { city: 'SF' });
    const t2 = { ...thinkingTurn('toolu_2', 'Two.', { city: 'LA' }), content: 'Edited two.' };
    const types = await send([
      { role: 'user', content: 'q' },
      t1,
      { role: 'tool', toolCallId: 'toolu_1', content: 'r1' },
      t2,
      { role: 'tool', toolCallId: 'toolu_2', content: 'r2' },
    ]);
    expect(types).toEqual([
      ['thinking', 'text', 'tool_use'],
      ['text', 'tool_use'],
    ]);
  });

  it('rebuilds a turn whose tool arguments were edited, so the edit reaches the wire', async () => {
    const fetchMock = mockFetch(() => jsonResponse(200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: {} }));
    const turn = thinkingTurn('toolu_1', 'One.', { city: 'Paris' });
    turn.toolCalls[0]!.function.arguments = '{"city":"Berlin"}';
    await new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) }).complete({
      model: 'claude-opus-5-5',
      messages: [{ role: 'user', content: 'q' }, turn],
    });
    const sent = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string).messages[1];
    expect(sent.content).toEqual([
      { type: 'text', text: 'One.' },
      { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: { city: 'Berlin' } },
    ]);
  });

  it('still replays verbatim when the arguments differ only in key order or spacing', async () => {
    const turn = thinkingTurn('toolu_1', 'One.', { city: 'SF', units: { temp: 'C' } });
    turn.toolCalls[0]!.function.arguments = '{ "units": {"temp":"C"}, "city": "SF" }';
    expect(await send([{ role: 'user', content: 'q' }, turn])).toEqual([['thinking', 'text', 'tool_use']]);
  });
});

describe('AnthropicConnector review-5 stream details', () => {
  const streamOf = (events: unknown[]) =>
    new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => new Response(sse(events), { status: 200, headers: { 'content-type': 'text/event-stream' } }))),
    });

  it('keeps a message_start usage count when message_delta restates it as null', async () => {
    const deltas = await drain(
      streamOf([
        { type: 'message_start', message: { usage: { input_tokens: 5, cache_read_input_tokens: 90, cache_creation_input_tokens: 10, output_tokens: 1 } } },
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hi' } },
        { type: 'content_block_stop', index: 0 },
        { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: null, cache_read_input_tokens: null, output_tokens: 9 } },
      ]),
      { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] },
    );
    expect(deltas.find((d) => d.usage)?.usage).toEqual({ inputTokens: 105, outputTokens: 9, totalTokens: 114 });
  });

  it('keeps a tool_use input given on content_block_start when no input_json_delta follows', async () => {
    const deltas = await drain(
      streamOf([
        { type: 'message_start', message: { usage: { input_tokens: 1 } } },
        { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'toolu_1', name: 'f', input: { a: 1 } } },
        { type: 'content_block_stop', index: 0 },
        { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 3 } },
      ]),
      { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] },
    );
    expect(deltas.find((d) => d.providerMetadata)?.providerMetadata).toEqual({
      anthropic: { content: [{ type: 'tool_use', id: 'toolu_1', name: 'f', input: { a: 1 } }] },
    });
  });

  it('classifies a mid-stream error event without a recognised type as unknown', async () => {
    await expect(
      drain(streamOf([{ type: 'error', error: { type: 'brand_new_error', message: 'odd' } }]), {
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ statusCode: 200, providerCode: 'unknown', providerMessage: 'odd' });
  });
});

describe('AnthropicConnector review-5 request details', () => {
  const body = async (input: Partial<ChatInput>) => {
    const fetchMock = mockFetch(() => jsonResponse(200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: {} }));
    await new AnthropicConnector({ apiKey: 'k', fetch: asFetch(fetchMock) }).complete({
      model: 'claude-opus-5-5',
      messages: [{ role: 'user', content: 'x' }],
      ...input,
    });
    return JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
  };

  it('sends an empty effort as medium and an unrecognised one as given', async () => {
    expect((await body({ reasoning: { effort: '' as never } })).output_config).toEqual({ effort: 'medium' });
    expect((await body({ reasoning: { effort: 'max' as never } })).output_config).toEqual({ effort: 'max' });
  });

  it('defaults max_tokens to 16000', async () => {
    expect((await body({})).max_tokens).toBe(16000);
  });

  it('does not read "context window" on its own as an overflow', async () => {
    const err = (await new AnthropicConnector({
      apiKey: 'k',
      fetch: asFetch(
        mockFetch(() =>
          jsonResponse(400, { type: 'error', error: { type: 'invalid_request_error', message: 'max_tokens: 300000 > 128000, the maximum for this model\'s context window' } }),
        ),
      ),
    })
      .complete({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'x' }] })
      .then(expectedFailure, (e: unknown) => e as ConnectorError)) as ConnectorError;
    expect(err.providerCode).toBe('invalid_request');
  });
});
