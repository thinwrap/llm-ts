import { describe, it, expect, vi } from 'vitest';
import { Chat } from './chat.facade';
import { ConnectorError } from '../types';
import type { ChatInput, ChatResult, IChatConnector } from '../types';

const input: ChatInput = { model: 'm', messages: [{ role: 'user', content: 'hi' }] };

describe('Chat facade', () => {
  it('constructs a shared-compat connector by provider id and delegates complete()', async () => {
    const fetchMock = vi.fn((_url: string, _init?: RequestInit): Promise<Response> =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            model: 'm',
            choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    );
    const chat = new Chat('openai', {
      apiKey: 'k',
      fetch: fetchMock as unknown as typeof fetch,
    });
    expect(chat.id).toBe('openai');
    const res = await chat.complete(input);
    expect(res.message.content).toBe('ok');
    expect(fetchMock.mock.calls[0]![0]).toContain('api.openai.com');
  });

  it('accepts a connector instance directly', async () => {
    const fake: IChatConnector = {
      id: 'fake',
      async complete(): Promise<ChatResult> {
        return {
          message: { role: 'assistant', content: 'x' },
          finishReason: 'stop',
          usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
          model: 'm',
          raw: null,
        };
      },
      async *stream() {
        /* no deltas */
      },
    };
    const chat = new Chat(fake);
    expect(chat.id).toBe('fake');
    expect((await chat.complete(input)).message.content).toBe('x');
  });

  it('throws ConnectorError for an unknown provider id', () => {
    expect(
      () => new Chat('nope' as unknown as 'openai', { apiKey: 'k' }),
    ).toThrow(ConnectorError);
  });
});
