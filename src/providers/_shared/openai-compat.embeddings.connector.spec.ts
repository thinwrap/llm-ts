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

describe('OpenAICompatEmbeddingsConnector 2xx validation', () => {
  const create = (body: unknown, inputs: string | string[] = ['a', 'b']) =>
    new OpenAICompatEmbeddingsConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, body))),
    })
      .create({ model: 'm', input: inputs })
      .catch((e: unknown) => e) as Promise<ConnectorError>;

  it('raises a 200 carrying an error object, classified by its code but keeping the 200', async () => {
    const err = await create({ error: { code: 503, message: 'upstream down' } });
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('provider_unavailable');
    expect(err.message).toBe('upstream down');
  });

  it.each([[{}], [{ data: null }]])('raises the empty-body error for %j', async (body) => {
    const err = await create(body);
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.providerCode).toBe('provider_unavailable');
  });

  it('raises when the vector count does not match the input count', async () => {
    const err = await create({ data: [{ index: 0, embedding: [0.1, 0.2] }] });
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.providerCode).toBe('provider_unavailable');
    expect(err.message).toContain('1 embeddings for 2 inputs');
  });

  it('counts a single string input as one', async () => {
    const err = await create({ data: [] }, 'a');
    expect(err.providerCode).toBe('provider_unavailable');
  });

  it.each([[null], ['0.2']])('raises on a non-numeric vector cell (%j) instead of coercing it', async (cell) => {
    const err = await create({
      data: [
        { index: 0, embedding: [0.1, cell] },
        { index: 1, embedding: [0.3, 0.4] },
      ],
    });
    expect(err).toBeInstanceOf(ConnectorError);
    expect(err.providerCode).toBe('provider_unavailable');
  });

  it('raises when a row has no embedding array', async () => {
    const err = await create({ data: [{ index: 0 }, { index: 1, embedding: [0.3] }] });
    expect(err.providerCode).toBe('provider_unavailable');
  });
});

describe('OpenAICompatEmbeddingsConnector dimensions field', () => {
  const body = { data: [{ index: 0, embedding: [0.1, 0.2] }], usage: { prompt_tokens: 1, total_tokens: 1 } };

  it.each([
    ['mistral', 'output_dimension'],
    ['openai', 'dimensions'],
  ] as const)('sends %s dimensions as %s', async (id, field) => {
    const fetchMock = mockFetch(() => jsonResponse(200, body));
    await new OpenAICompatEmbeddingsConnector(SPECS[id], { apiKey: 'k', fetch: asFetch(fetchMock) }).create({
      model: 'm',
      input: 'hi',
      dimensions: 256,
    });
    const sent = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(sent[field]).toBe(256);
    expect(Object.keys(sent).filter((k) => k === 'dimensions' || k === 'output_dimension')).toEqual([field]);
  });

  it('ignores a blank Retry-After instead of reading it as 0', async () => {
    const err = (await new OpenAICompatEmbeddingsConnector(SPECS.openai, {
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(429, { error: { message: 'slow' } }, { 'retry-after': '' }))),
    })
      .create({ model: 'm', input: 'hi' })
      .catch((e: unknown) => e)) as ConnectorError;
    expect(err.providerCode).toBe('rate_limited');
    expect((err.cause as Record<string, unknown>).retryAfterSeconds).toBeUndefined();
  });
});

describe('OpenAICompatEmbeddingsConnector credit exhaustion', () => {
  it('maps a 402 to rate_limited', async () => {
    const err = (await new OpenAICompatEmbeddingsConnector(SPECS.openrouter, {
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(402, { error: { code: 402, message: 'Insufficient credits' } }))),
    })
      .create({ model: 'm', input: 'hi' })
      .catch((e: unknown) => e)) as ConnectorError;
    expect(err.providerCode).toBe('rate_limited');
  });
});

describe('OpenAICompatEmbeddingsConnector in-body error names', () => {
  it('classifies a 2xx error with a string code the way chat does', async () => {
    const err = (await new OpenAICompatEmbeddingsConnector(SPECS.openrouter, {
      apiKey: 'k',
      fetch: asFetch(mockFetch(() => jsonResponse(200, { error: { code: 'insufficient_quota', message: 'quota' } }))),
    })
      .create({ model: 'm', input: 'hi' })
      .catch((e: unknown) => e)) as ConnectorError;
    expect(err.providerCode).toBe('rate_limited');
  });
});

describe('OpenAICompatEmbeddingsConnector in-body error status', () => {
  const create = (body: unknown) =>
    new OpenAICompatEmbeddingsConnector(SPECS.openrouter, { apiKey: 'k', fetch: asFetch(mockFetch(() => jsonResponse(200, body))) })
      .create({ model: 'm', input: 'hi' })
      .catch((e: unknown) => e) as Promise<ConnectorError>;

  it('keeps the 200 it arrived with when the code is a name', async () => {
    const err = await create({ error: { code: 'insufficient_quota', message: 'quota' } });
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('rate_limited');
  });

  it('classifies an error with no usable code as unknown', async () => {
    const err = await create({ error: { message: 'odd' } });
    expect(err.statusCode).toBe(200);
    expect(err.providerCode).toBe('unknown');
  });
});
