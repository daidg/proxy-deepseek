# proxy-deepseek

Local proxy that fixes DeepSeek V4 thinking-mode `reasoning_content` passback errors
when using Claude Code, Codex CLI, or any OpenAI-compatible client via OpenRouter.

## The Problem

DeepSeek V4 (and V3.2+) thinking mode requires that `reasoning_content` from
an assistant turn with tool calls is passed back verbatim in every subsequent
request. OpenAI-compatible clients (Claude Code, Codex CLI, etc.) don't do this
automatically → **400 error** on the 2nd+ turn.

## The Fix

This proxy intercepts responses, stores `reasoning_content` by tool_call ID,
and re-injects it on the next turn — exactly like CCR PR #1376.

## Setup

```bash
npm install
```

Edit `.env`:

```
UPSTREAM_URL=https://openrouter.ai/api/v1/chat/completions
UPSTREAM_API_KEY=sk-or-v1-xxxxx
UPSTREAM_MODEL=deepseek/deepseek-v4-flash
PORT=3000
```

## Run

```bash
node src/server.js
```

## Claude Code config

In your OpenAI provider settings, point to the proxy:

```
Base URL: http://localhost:3000/v1/chat/completions
API Key:  (your OpenRouter key)
Model:    deepseek/deepseek-v4-flash
```

Or if using CCR (Claude Code Router), update its upstream to `http://localhost:3000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/chat/completions` | OpenAI format proxy |
| POST | `/v1/messages` | Anthropic format (Claude Code) |
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
