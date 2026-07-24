import type { Passthrough } from './passthrough.type';

export type ChatRole = 'system' | 'user' | 'assistant' | 'tool';

export interface TextPart {
  type: 'text';
  text: string;
}

export interface ImagePart {
  type: 'image';
  /** Raw base64 payload (no `data:` prefix). Canonical image input; the connector encodes for the vendor wire. */
  base64: string;
  /** e.g. `image/png`, `image/jpeg`. */
  mediaType: string;
}

export type ContentPart = TextPart | ImagePart;

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    /** JSON-encoded arguments string. */
    arguments: string;
  };
}

export interface ChatMessage {
  role: ChatRole;
  content: string | ContentPart[] | null;
  /** Present on assistant turns that invoked tools. */
  toolCalls?: ToolCall[];
  /** Required on `role: 'tool'` messages — the id of the tool call being answered. */
  toolCallId?: string;
  /** Optional author / tool name. */
  name?: string;
}

export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    /** JSON Schema object describing the arguments. */
    parameters?: Record<string, unknown>;
  };
}

export type ToolChoice =
  | 'auto'
  | 'none'
  | 'required'
  | { type: 'function'; function: { name: string } };

export type ResponseFormat =
  | { type: 'text' }
  | { type: 'json_object' }
  | {
      type: 'json_schema';
      jsonSchema: { name: string; schema: Record<string, unknown> };
    };

/** Minimal normalized reasoning control (D2). CoT output is NOT normalized — read it from `ChatResult.raw`. */
export interface ReasoningOptions {
  effort: 'low' | 'medium' | 'high';
}

export interface ChatInput {
  model: string;
  messages: ChatMessage[];
  tools?: ToolDefinition[];
  toolChoice?: ToolChoice;
  responseFormat?: ResponseFormat;
  temperature?: number;
  topP?: number;
  maxOutputTokens?: number;
  stop?: string | string[];
  reasoning?: ReasoningOptions;
  /** Optional cancellation signal, forwarded to the underlying `fetch` for both `complete()` and `stream()`. */
  signal?: AbortSignal;
  /** Escape hatch for sub-baseline / provider-specific request fields. Never emulated. */
  _passthrough?: Passthrough;
}

export type FinishReason =
  | 'stop'
  | 'length'
  | 'tool_calls'
  | 'content_filter'
  | 'unknown';

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface ChatResult {
  message: { role: 'assistant'; content: string | null; toolCalls?: ToolCall[] };
  finishReason: FinishReason;
  usage: Usage;
  /** The model id the provider reported serving (may differ from the requested id). */
  model: string;
  /** Verbatim vendor response body — reasoning CoT, cached-token counts, cost, etc. live here. */
  raw: unknown;
}

export interface ChatStreamDelta {
  /** Incremental assistant text. */
  contentDelta?: string;
  /** Incremental tool-call fragment (concatenate `argumentsDelta` per `index`). */
  toolCallDelta?: {
    index: number;
    id?: string;
    functionName?: string;
    argumentsDelta?: string;
  };
  /** Present on the terminal chunk. */
  finishReason?: FinishReason;
  /** Present on the final usage chunk, when the provider reports it. */
  usage?: Usage;
  /** Verbatim vendor chunk. */
  raw?: unknown;
}
