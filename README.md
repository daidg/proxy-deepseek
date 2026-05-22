# proxy-deepseek

Local proxy that fixes DeepSeek V4 thinking-mode `reasoning_content` passback errors
when using Claude Code, Codex CLI, or any OpenAI-compatible client via OpenRouter.

## The Problem

DeepSeek V4 (and V3.2+) thinking mode requires that `reasoning_content` from
an assistant turn with tool calls is passed back verbatim in every subsequent
request. OpenAI-compatible clients don't do this automatically → **400 error**
on the 2nd+ turn.

Additionally, DeepSeek's API has stricter message format requirements than
OpenAI's — `content: null` with `tool_calls` is rejected, and Anthropic-format
messages (used by Claude Code) must be converted to OpenAI format.

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
