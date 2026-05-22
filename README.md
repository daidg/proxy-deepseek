# proxy-deepseek

Local proxy that fixes DeepSeek V4 thinking-mode issues when using
Claude Code via OpenRouter:

- **Protocol translation** — converts Anthropic `/v1/messages` ⇄ OpenAI `/v1/chat/completions`
  (OpenRouter serves DeepSeek via OpenAI format; Claude Code speaks Anthropic)
- **`reasoning_content` passback** — stores and re-injects `reasoning_content` across
  multi-turn tool calls, preventing 400 errors

## The Problem

Claude Code + OpenRouter + DeepSeek V4 thinking mode hits two incompatibilities:

1. Claude Code speaks Anthropic protocol, but OpenRouter serves DeepSeek via OpenAI format
2. DeepSeek V4 (and V3.2+) thinking mode requires `reasoning_content` from an
   assistant turn with tool calls to be passed back verbatim in every subsequent
   request. No client does this automatically → **400 error** on the 2nd+ turn.

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

`.env` is optional. Defaults work for OpenRouter — the proxy passes through the
client's API key and model. Only configure overrides if needed:

```
# Override upstream (default: OpenRouter)
# UPSTREAM_URL=https://api.deepseek.com/v1/chat/completions

# Fallback API key (only used when client doesn't send one)
# UPSTREAM_API_KEY=***

# Force a specific model (otherwise uses the client's model)
# UPSTREAM_MODEL=deepseek/deepseek-v4-flash

# Enable debug logging
# DEBUG=1
```

## Run

```bash
# Default port 3000
node src/server.js

# Custom port
node src/server.js --port=8080
```

## Claude Code config

Point Claude Code to the proxy's `/v1/messages` endpoint:

```
Base URL: http://localhost:3000/v1/messages
API Key:  (your OpenRouter key — passed through to upstream)
Model:    deepseek/deepseek-v4-flash
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/messages` | Anthropic format — Claude Code's endpoint |
| DELETE | `/session/:sessionId` | Clear a session's reasoning store |
| GET | `/sessions` | List active sessions |

## Architecture

```
Claude Code → proxy-deepseek → OpenRouter → DeepSeek
                ↑
          intercept response
          store reasoning_content (keyed by tool_call.id)
                ↓
          next request: inject stored reasoning_content
```
