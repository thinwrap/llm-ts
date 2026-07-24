export interface BedrockConfig {
  /** AWS region, e.g. `us-east-1`. No environment inference. */
  region: string;
  /** AWS access key id. */
  accessKeyId: string;
  /** AWS secret access key. */
  secretAccessKey: string;
  /** Optional STS session token (temporary credentials). */
  sessionToken?: string;
  /** Override origin. Default `https://bedrock-runtime.<region>.amazonaws.com`. */
  baseUrl?: string;
  /** Converse `inferenceConfig.maxTokens` used when `ChatInput.maxOutputTokens` is omitted. Default 4096. */
  defaultMaxTokens?: number;
  /** Bring-your-own fetch. Defaults to `globalThis.fetch`. */
  fetch?: typeof fetch;
  /** Extra headers merged onto every request (participate in the SigV4 signature). */
  headers?: Record<string, string>;
}
