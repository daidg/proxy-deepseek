# proxy-deepseek

Local proxy that fixes DeepSeek V4 thinking-mode issues when using
Claude Code via OpenRouter:

- **Protocol translation** — converts Anthropic `/v1/messages` ⇄ OpenAI `/v1/chat/completions`
  (Claude Code speaks Anthropic; DeepSeek expects OpenAI)
- **`reasoning_content` passback** — stores and re-injects `reasoning_content` across
  multi-turn tool calls, preventing 400 errors

Also works as a lightweight reasoning_content passback layer for OpenAI-native clients
(Codex CLI, Cursor, etc.) via `/v1/chat/completions`.

## The Problem

**For Claude Code specifically:** Two layers of incompatibility with DeepSeek:
1. Claude Code speaks Anthropic protocol — DeepSeek expects OpenAI format
2. DeepSeek V4 (and V3.2+) thinking mode requires `reasoning_content` from an
   assistant turn with tool calls to be passed back verbatim in every subsequent
   request. No client does this automatically → **400 error** on the 2nd+ turn.

**For OpenAI-native clients:** Only issue #2 applies — they already speak the
right protocol, but still need a layer that preserves `reasoning_content`.

## What It Does

| Transformation | Description |
|---|---|
| `reasoning_content` preservation | Stores reasoning from responses and re-injects on subsequent turns |
| `content: null` fix | Converts `null` content to `""` on assistant messages with tool_calls |
| Anthropic → OpenAI conversion | Translates `/v1/messages` (Anthropic format) to `/v1/chat/completions` (OpenAI format) |
| `[TOOL_CALLS_START]` → `role:tool` | Converts Claude Code's user-format tool results to OpenAI `role:tool` messages |
| Tools format normalization | Adds missing `type: "function"` wrapper to Claude Code's tool definitions |
| SSE streaming conversion | Translates OpenAI streaming chunks to Anthropic SSE event format |

## Setup

```bash
npm install
```

Edit `.env`:

```
UPSTREAM_URL=https://openrouter.ai/api/v1/chat/completions
UPSTREAM_API_KEY=***
UPSTREAM_MODEL=deepseek/deepseek-v4-flash
DEBUG=0
```

## Run

```bash
# Default port 3000
node src/server.js

# Custom port
node src/server.js --port=8080
```

## Claude Code config

Claude Code speaks Anthropic protocol — point it to `/v1/messages`:

```
Base URL: http://localhost:3000/v1/messages
API Key:  (your OpenRouter key — passed through to upstream)
Model:    deepseek/deepseek-v4-flash
```

## OpenAI-compatible clients

Clients that speak OpenAI format (Codex CLI, Cursor, etc.) use `/v1/chat/completions`:

```
Base URL: http://localhost:3000/v1/chat/completions
API Key:  (your OpenRouter key)
Model:    deepseek/deepseek-v4-flash
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/messages` | Anthropic format (Claude Code) |
| POST | `/v1/chat/completions` | OpenAI format (Codex CLI, Cursor, etc.) |
| DELETE | `/session/:sessionId` | Clear a session's reasoning store |
| GET | `/sessions` | List active sessions |

## Architecture

```
Client → proxy-deepseek → OpenRouter → DeepSeek
           ↑
     intercept response
     store reasoning_content (keyed by tool_call.id)
           ↓
     next request: inject stored reasoning_content
```
