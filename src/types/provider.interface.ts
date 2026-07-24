import type { ChatInput, ChatResult, ChatStreamDelta } from './chat.types';

/** The contract every chat connector satisfies — shared-compat or native adapter. */
export interface IChatConnector {
  readonly id: string;
  complete(input: ChatInput): Promise<ChatResult>;
  stream(input: ChatInput): AsyncIterable<ChatStreamDelta>;
}
