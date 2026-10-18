import { describe, it, expect, vi } from 'vitest';
import { BedrockConnector, signAwsRequest } from './bedrock.connector';
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

const creds = { region: 'us-east-1', accessKeyId: 'AKIAEXAMPLE', secretAccessKey: 'secret' };

describe('BedrockConnector.complete', () => {
  it('translates the Converse request/response, hoists system, and signs with SigV4', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        output: { message: { content: [{ text: 'hello' }] } },
        stopReason: 'end_turn',
        usage: { inputTokens: 5, outputTokens: 2, totalTokens: 7 },
      }),
    );
    const c = new BedrockConnector({ ...creds, fetch: asFetch(fetchMock) });
    const input: ChatInput = {
      model: 'amazon.nova-pro-v1:0',
      maxOutputTokens: 50,
      messages: [
        { role: 'system', content: 'be brief' },
        { role: 'user', content: 'hi' },
      ],
    };
    const res = await c.complete(input);

    expect(res.message.content).toBe('hello');
    expect(res.finishReason).toBe('stop');
    expect(res.usage).toEqual({ inputTokens: 5, outputTokens: 2, totalTokens: 7 });

    const call = fetchMock.mock.calls[0]!;
    // WIRE path is single-encoded (colon → %3A); the SigV4 canonical path is
    // DOUBLE-encoded — proven in the dedicated regression test below.
    expect(call[0]).toBe(
      'https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-pro-v1%3A0/converse',
    );
    const init = call[1]!;
    const headers = init.headers as Record<string, string>;
    expect(headers.Authorization).toMatch(/^AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE\//);
    expect(headers.Authorization).toContain('/us-east-1/bedrock/aws4_request');
    expect(headers.Authorization).toMatch(/SignedHeaders=[^,]*host[^,]*, Signature=[0-9a-f]{64}$/);
    expect(headers['X-Amz-Date']).toMatch(/^\d{8}T\d{6}Z$/);
    expect(headers['X-Amz-Content-Sha256']).toMatch(/^[0-9a-f]{64}$/);

    const body = JSON.parse(init.body as string);
    expect(body.system).toEqual([{ text: 'be brief' }]);
    expect(body.inferenceConfig.maxTokens).toBe(50);
    expect(body.messages).toEqual([{ role: 'user', content: [{ text: 'hi' }] }]);
  });

  it('maps toolUse blocks → normalized toolCalls and tools → toolConfig', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        output: { message: { content: [{ toolUse: { toolUseId: 't1', name: 'get_weather', input: { city: 'SF' } } }] } },
        stopReason: 'tool_use',
        usage: { inputTokens: 8, outputTokens: 4, totalTokens: 12 },
      }),
    );
    const c = new BedrockConnector({ ...creds, fetch: asFetch(fetchMock) });
    const res = await c.complete({
      model: 'anthropic.claude-x',
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
    expect(body.toolConfig.tools).toEqual([
      { toolSpec: { name: 'get_weather', inputSchema: { json: { type: 'object' } } } },
    ]);
    expect(body.toolConfig.toolChoice).toEqual({ any: {} });
    expect(body.inferenceConfig.maxTokens).toBe(4096); // default when maxOutputTokens omitted
  });

  it('includes X-Amz-Security-Token when a session token is supplied', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { output: { message: { content: [{ text: 'ok' }] } }, stopReason: 'end_turn', usage: {} }),
    );
    await new BedrockConnector({ ...creds, sessionToken: 'sess-123', fetch: asFetch(fetchMock) }).complete({
      model: 'm',
      messages: [{ role: 'user', content: 'x' }],
    });
    const headers = fetchMock.mock.calls[0]![1]!.headers as Record<string, string>;
    expect(headers['X-Amz-Security-Token']).toBe('sess-123');
    expect(headers.Authorization).toContain('x-amz-security-token');
  });

  it('maps 403 → auth_failed and a 400 ValidationException → invalid_request', async () => {
    const authFetch = mockFetch(() => jsonResponse(403, { message: 'not authorized', __type: 'AccessDeniedException' }));
    await expect(
      new BedrockConnector({ ...creds, fetch: asFetch(authFetch) }).complete({
        model: 'm',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ name: 'ConnectorError', providerCode: 'auth_failed' });

    const valFetch = mockFetch(() => jsonResponse(400, { message: 'bad input', __type: 'ValidationException' }));
    await expect(
      new BedrockConnector({ ...creds, fetch: asFetch(valFetch) }).complete({
        model: 'm',
        messages: [{ role: 'user', content: 'x' }],
      }),
    ).rejects.toMatchObject({ providerCode: 'invalid_request' });
  });

  it('SigV4 signs the DOUBLE-encoded canonical path for a colon-bearing model id (wire stays single-encoded)', async () => {
    const model = 'amazon.nova-pro-v1:0'; // on-demand id → contains `:`
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { output: { message: { content: [{ text: 'ok' }] } }, stopReason: 'end_turn', usage: {} }),
    );
    await new BedrockConnector({ ...creds, fetch: asFetch(fetchMock) }).complete({
      model,
      messages: [{ role: 'user', content: 'hi' }],
    });

    const call = fetchMock.mock.calls[0]!;
    const url = call[0] as string;
    const init = call[1]!;
    const headers = init.headers as Record<string, string>;
    const serializedBody = init.body as string;

    // The WIRE URL is single-encoded (`%3A`), never double-encoded.
    expect(url).toContain('/model/amazon.nova-pro-v1%3A0/converse');
    expect(url).not.toContain('%253A');

    // Re-sign with the same captured inputs (real signer) for BOTH encodings and
    // compare against the signature the connector actually emitted. This proves
    // the connector's canonical path is the DOUBLE-encoded one, not the wire path.
    const sigOf = (auth: string | undefined) => /Signature=([0-9a-f]{64})/.exec(auth ?? '')![1]!;
    const signerArgs = {
      method: 'POST',
      service: 'bedrock',
      region: creds.region,
      host: 'bedrock-runtime.us-east-1.amazonaws.com',
      accessKeyId: creds.accessKeyId,
      secretAccessKey: creds.secretAccessKey,
      serializedBody,
      additionalSignedHeaders: { 'Content-Type': 'application/json' },
      isoTimestamp: headers['X-Amz-Date']!,
    };
    const singlePath = `/model/${encodeURIComponent(model)}/converse`; // %3A
    const doublePath = `/model/${encodeURIComponent(encodeURIComponent(model))}/converse`; // %253A
    expect(singlePath).toContain('%3A');
    expect(singlePath).not.toContain('%253A');
    expect(doublePath).toContain('%253A');
    expect(singlePath).not.toBe(doublePath);

    const connectorSig = sigOf(headers.Authorization);
    const singleSig = sigOf(signAwsRequest({ ...signerArgs, path: singlePath }).Authorization);
    const doubleSig = sigOf(signAwsRequest({ ...signerArgs, path: doublePath }).Authorization);

    expect(connectorSig).toBe(doubleSig); // signed the double-encoded canonical path
    expect(connectorSig).not.toBe(singleSig); // NOT the single-encoded wire path
  });

  it('signs a colon-free model id identically whether single- or double-encoded (no regression)', async () => {
    const model = 'llama-3-70b'; // no `:` → single- and double-encode are identical
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { output: { message: { content: [{ text: 'ok' }] } }, stopReason: 'end_turn', usage: {} }),
    );
    await new BedrockConnector({ ...creds, fetch: asFetch(fetchMock) }).complete({
      model,
      messages: [{ role: 'user', content: 'hi' }],
    });
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain('/model/llama-3-70b/converse');
    expect(encodeURIComponent(model)).toBe(encodeURIComponent(encodeURIComponent(model)));
  });
});

describe('BedrockConnector construction', () => {
  it('throws a ConnectorError (not a raw TypeError) on a malformed baseUrl', () => {
    expect(() => new BedrockConnector({ ...creds, baseUrl: 'not a url' })).toThrow(ConnectorError);
    expect(() => new BedrockConnector({ ...creds, baseUrl: 'not a url' })).toThrow(/could not derive a host/);
  });
});

describe('BedrockConnector.stream (non-incremental)', () => {
  it('yields the Converse result as content + finish + usage deltas', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, {
        output: { message: { content: [{ text: 'hello world' }] } },
        stopReason: 'end_turn',
        usage: { inputTokens: 3, outputTokens: 5, totalTokens: 8 },
      }),
    );
    const c = new BedrockConnector({ ...creds, fetch: asFetch(fetchMock) });
    const deltas: ChatStreamDelta[] = [];
    for await (const d of c.stream({ model: 'm', messages: [{ role: 'user', content: 'hi' }] })) {
      deltas.push(d);
    }
    expect(deltas.map((d) => d.contentDelta ?? '').join('')).toBe('hello world');
    expect(deltas.some((d) => d.finishReason === 'stop')).toBe(true);
    expect(deltas.find((d) => d.usage)?.usage).toEqual({ inputTokens: 3, outputTokens: 5, totalTokens: 8 });
    // hit the non-stream Converse endpoint (fallback), not converse-stream
    expect(fetchMock.mock.calls[0]![0]).toContain('/converse');
    expect(fetchMock.mock.calls[0]![0]).not.toContain('converse-stream');
  });
});

describe('Chat facade → bedrock dispatch', () => {
  it('constructs the native Bedrock adapter by provider id', async () => {
    const fetchMock = mockFetch(() =>
      jsonResponse(200, { output: { message: { content: [{ text: 'ok' }] } }, stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
    );
    const chat = new Chat('bedrock', { ...creds, fetch: asFetch(fetchMock) });
    expect(chat.id).toBe('bedrock');
    const res = await chat.complete({ model: 'm', messages: [{ role: 'user', content: 'hi' }] });
    expect(res.message.content).toBe('ok');
    expect(fetchMock.mock.calls[0]![0]).toContain('bedrock-runtime.us-east-1.amazonaws.com');
  });
});

describe('BedrockConnector 2xx without output', () => {
  it.each([[{}], [{ stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 0, totalTokens: 1 } }]])(
    'raises the empty-body error for %j',
    async (body) => {
      const c = new BedrockConnector({ ...creds, fetch: asFetch(mockFetch(() => jsonResponse(200, body))) });
      const err = (await c
        .complete({ model: 'amazon.nova-lite-v1:0', messages: [{ role: 'user', content: 'x' }] })
        .catch((e: unknown) => e)) as ConnectorError;
      expect(err).toBeInstanceOf(ConnectorError);
      expect(err.statusCode).toBe(200);
      expect(err.providerCode).toBe('provider_unavailable');
    },
  );
});

const okConverse = () =>
  jsonResponse(200, { output: { message: { content: [{ text: 'ok' }] } }, stopReason: 'end_turn', usage: {} });

describe('BedrockConnector toolChoice none', () => {
  it('omits toolConfig entirely so the model is offered no tools', async () => {
    const fetchMock = mockFetch(okConverse);
    await new BedrockConnector({ ...creds, fetch: asFetch(fetchMock) }).complete({
      model: 'amazon.nova-pro-v1:0',
      messages: [{ role: 'user', content: 'hi' }],
      tools: [{ type: 'function', function: { name: 'f', parameters: { type: 'object' } } }],
      toolChoice: 'none',
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body).not.toHaveProperty('toolConfig');
  });
});

describe('BedrockConnector SigV4 header merge', () => {
  it('keeps one entry per header name whatever the casing, and signs the value it sends', async () => {
    const fetchMock = mockFetch(okConverse);
    await new BedrockConnector({
      ...creds,
      headers: { 'content-type': 'application/json; charset=utf-8' },
      fetch: asFetch(fetchMock),
    }).complete({
      model: 'amazon.nova-pro-v1:0',
      messages: [{ role: 'user', content: 'hi' }],
      _passthrough: { headers: { 'CONTENT-TYPE': 'application/json', host: 'ignored.example' } },
    });
    const init = fetchMock.mock.calls[0]![1]!;
    const headers = init.headers as Record<string, string>;
    const names = Object.keys(headers).map((n) => n.toLowerCase());
    expect(names.filter((n) => n === 'content-type')).toHaveLength(1);
    expect(names.filter((n) => n === 'host')).toHaveLength(1);
    const contentType = Object.entries(headers).find(([n]) => n.toLowerCase() === 'content-type')![1];
    expect(contentType).toBe('application/json');
    expect(Object.entries(headers).find(([n]) => n.toLowerCase() === 'host')![1]).toBe('bedrock-runtime.us-east-1.amazonaws.com');

    // Re-signing with exactly what went on the wire reproduces the signature.
    const resigned = signAwsRequest({
      method: 'POST',
      path: `/model/${encodeURIComponent(encodeURIComponent('amazon.nova-pro-v1:0'))}/converse`,
      service: 'bedrock',
      region: creds.region,
      host: 'bedrock-runtime.us-east-1.amazonaws.com',
      accessKeyId: creds.accessKeyId,
      secretAccessKey: creds.secretAccessKey,
      serializedBody: init.body as string,
      additionalSignedHeaders: { 'CONTENT-TYPE': contentType },
      isoTimestamp: headers['X-Amz-Date']!,
    });
    expect(headers.Authorization).toBe(resigned.Authorization);
  });
});

describe('BedrockConnector region validation', () => {
  it.each(['us-east-1', 'eu-central-2', 'us-gov-west-1'])('accepts %s', (region) => {
    expect(() => new BedrockConnector({ ...creds, region })).not.toThrow();
  });

  it.each(['attacker.example#', 'us-east-1.evil.example/x?', 'US-EAST-1', '', 'us east 1'])(
    'rejects %j before any request can be signed to another host',
    (region) => {
      expect(() => new BedrockConnector({ ...creds, region })).toThrow(ConnectorError);
    },
  );
});

describe('BedrockConnector usage and finish reasons', () => {
  it('folds cache reads and writes into inputTokens', async () => {
    const c = new BedrockConnector({
      ...creds,
      fetch: asFetch(
        mockFetch(() =>
          jsonResponse(200, {
            output: { message: { content: [{ text: 'ok' }] } },
            stopReason: 'end_turn',
            usage: { inputTokens: 4, outputTokens: 50, totalTokens: 1554, cacheReadInputTokens: 1500, cacheWriteInputTokens: 0 },
          }),
        ),
      ),
    });
    const res = await c.complete({ model: 'm', messages: [{ role: 'user', content: 'x' }] });
    expect(res.usage).toEqual({ inputTokens: 1504, outputTokens: 50, totalTokens: 1554 });
  });

  it.each([
    ['model_context_window_exceeded', 'length'],
    ['malformed_tool_use', 'unknown'],
    ['malformed_model_output', 'unknown'],
  ])('maps stopReason %s → %s', async (stopReason, expected) => {
    const c = new BedrockConnector({
      ...creds,
      fetch: asFetch(mockFetch(() => jsonResponse(200, { output: { message: { content: [{ text: 'x' }] } }, stopReason, usage: {} }))),
    });
    expect((await c.complete({ model: 'm', messages: [{ role: 'user', content: 'x' }] })).finishReason).toBe(expected);
  });
});

describe('BedrockConnector error classification', () => {
  const fail = (status: number, body: unknown, headers?: Record<string, string>) =>
    new BedrockConnector({ ...creds, fetch: asFetch(mockFetch(() => jsonResponse(status, body, headers))) })
      .complete({ model: 'm', messages: [{ role: 'user', content: 'x' }] })
      .then(expectedFailure, (e: unknown) => e as ConnectorError);

  it.each([
    [400, 'Input is too long for requested model.', 'context_length_exceeded'],
    [400, 'The model returned the following errors: prompt is too long: 210000 tokens > 200000 maximum', 'context_length_exceeded'],
    [400, 'The maximum tokens you requested exceeds the model limit of 8192.', 'invalid_request'],
    [400, 'Malformed input request: #: extraneous key [max_tokens] is not permitted', 'invalid_request'],
    [400, 'A conversation must start with a user message.', 'invalid_request'],
    [408, 'Model has timed out in processing the request.', 'provider_unavailable'],
    [424, 'The model returned an error.', 'provider_unavailable'],
  ])('maps %i "%s" → %s', async (status, message, expected) => {
    expect((await fail(status, { message })).providerCode).toBe(expected);
  });

  it('ignores a negative Retry-After', async () => {
    const err = await fail(429, { message: 'Too many requests' }, { 'retry-after': '-1' });
    expect(err.providerCode).toBe('rate_limited');
    expect((err.cause as Record<string, unknown>).retryAfterSeconds).toBeUndefined();
  });
});

describe('BedrockConnector review-5', () => {
  it('maps a 402 to rate_limited', async () => {
    const err = (await new BedrockConnector({ ...creds, fetch: asFetch(mockFetch(() => jsonResponse(402, { message: 'Payment required' }))) })
      .complete({ model: 'm', messages: [{ role: 'user', content: 'x' }] })
      .then(expectedFailure, (e: unknown) => e as ConnectorError)) as ConnectorError;
    expect(err.providerCode).toBe('rate_limited');
  });

  it.each([
    ['The model returned the following errors: input length and `max_tokens` exceed context limit: 190000 + 20000 > 200000', 'context_length_exceeded'],
    ['The context window setting is invalid', 'invalid_request'],
  ])('classifies a 400 "%s" as %s', async (message, expected) => {
    const err = (await new BedrockConnector({ ...creds, fetch: asFetch(mockFetch(() => jsonResponse(400, { message }))) })
      .complete({ model: 'm', messages: [{ role: 'user', content: 'x' }] })
      .then(expectedFailure, (e: unknown) => e as ConnectorError)) as ConnectorError;
    expect(err.providerCode).toBe(expected);
  });

  it('signs the full request path when baseUrl carries a path prefix', async () => {
    const fetchMock = mockFetch(okConverse);
    const model = 'amazon.nova-pro-v1:0';
    await new BedrockConnector({ ...creds, baseUrl: 'https://gw.example.com/bedrock/v2/', fetch: asFetch(fetchMock) }).complete({
      model,
      messages: [{ role: 'user', content: 'hi' }],
    });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe('https://gw.example.com/bedrock/v2/model/amazon.nova-pro-v1%3A0/converse');
    const headers = init!.headers as Record<string, string>;
    const expected = signAwsRequest({
      method: 'POST',
      path: '/bedrock/v2/model/amazon.nova-pro-v1%253A0/converse',
      service: 'bedrock',
      region: creds.region,
      host: 'gw.example.com',
      accessKeyId: creds.accessKeyId,
      secretAccessKey: creds.secretAccessKey,
      serializedBody: init!.body as string,
      additionalSignedHeaders: { 'Content-Type': 'application/json' },
      isoTimestamp: headers['X-Amz-Date']!,
    });
    expect(headers.Authorization).toBe(expected.Authorization);
  });

  it('drops empty text parts, which Converse rejects', async () => {
    const fetchMock = mockFetch(okConverse);
    await new BedrockConnector({ ...creds, fetch: asFetch(fetchMock) }).complete({
      model: 'm',
      messages: [{ role: 'user', content: [{ type: 'text', text: '' }, { type: 'text', text: 'hi' }] }],
    });
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.messages[0].content).toEqual([{ text: 'hi' }]);
  });
});
