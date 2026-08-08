# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.1] — 2026-08-08

### Fixed

- A non-2xx is classified even when the host's `fetch` rejects instead of resolving, and a
  transport rejection never carries the key-bearing request URL into the error.

## [1.0.0] — 2026-07-24

### Added

- Initial release of `@thinwrap/llm`: `Chat` (`complete` + `stream`) and `Embeddings` facades
  over 18 providers behind one normalized surface, with a typed `ConnectorError`.
- Fifteen OpenAI-compatible providers share one connector driven by a spec registry; Anthropic,
  Bedrock and Gemini have bespoke wire adapters emitting identical shapes.
- Zero runtime dependencies: bring-your-own `fetch`; Bedrock SigV4 hand-rolled on `node:crypto`.
