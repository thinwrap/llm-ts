import { describe, it, expect, vi } from 'vitest';
import { OpenAICompatEmbeddingsConnector } from './openai-compat.embeddings.connector';
import { SPECS } from './spec';
import { Embeddings } from '../../facades/embeddings.facade';
import { ConnectorError } from '../../types';

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

describe('OpenAICompatEmbeddingsConnector', () => {
  it('POSTs to /embeddings and normalizes data[].embedding into number[][] (input order)', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        model: 'text-embedding-3-small',
        // deliberately out of order to prove we sort by index
        data: [
          { index: 1, embedding: [0.3, 0.4] },
          { index: 0, embedding: [0.1, 0.2] },
        ],
        usage: { prompt_tokens: 5, total_tokens: 5 },
      }),
    );
    const c = new OpenAICompatEmbeddingsConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(fetchMock) });
    const res = await c.create({ model: 'text-embedding-3-small', input: ['a', 'b'], dimensions: 2 });

    expect(res.embeddings).toEqual([[0.1, 0.2], [0.3, 0.4]]);
    expect(res.usage).toEqual({ inputTokens: 5, totalTokens: 5 });
    expect(res.model).toBe('text-embedding-3-small');

    const call = fetchMock.mock.calls[0]!;
    expect(call[0]).toBe('https://api.openai.com/v1/embeddings');
    const body = JSON.parse(call[1]!.body as string);
    expect(body).toMatchObject({ model: 'text-embedding-3-small', input: ['a', 'b'], encoding_format: 'float', dimensions: 2 });
  });

  it('maps 401 → auth_failed', async () => {
    const fetchMock = mockFetch(() => jsonResponse(401, { error: { message: 'bad key' } }));
    await expect(
      new OpenAICompatEmbeddingsConnector(SPECS.openai, { apiKey: 'k', fetch: asFetch(fetchMock) }).create({
        model: 'm',
        input: 'x',
      }),
    ).rejects.toMatchObject({ name: 'ConnectorError', providerCode: 'auth_failed' });
  });
});

describe('Embeddings facade', () => {
  it('dispatches an embeddings-capable provider', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { model: 'm', data: [{ index: 0, embedding: [1, 2, 3] }], usage: { prompt_tokens: 1, total_tokens: 1 } }),
    );
    const emb = new Embeddings('together', { apiKey: 'k', fetch: asFetch(fetchMock) });
    expect(emb.id).toBe('together');
    const res = await emb.create({ model: 'BAAI/bge-large-en-v1.5', input: 'hello' });
    expect(res.embeddings).toEqual([[1, 2, 3]]);
    expect(fetchMock.mock.calls[0]![0]).toContain('api.together.ai');
  });

  it('throws for a provider with no OpenAI-compatible embeddings surface', () => {
    // deepseek has no float embeddings — not an EmbeddingsProviderId; cast to reach the runtime guard.
    expect(
      () => new Embeddings('deepseek' as unknown as 'openai', { apiKey: 'k' }),
    ).toThrow(ConnectorError);
  });
});
