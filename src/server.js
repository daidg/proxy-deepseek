/**
 * proxy-deepseek
 * 
 * Local proxy server that:
 * 1. Accepts requests in Anthropic /v1/messages format (Claude Code)
 *    and OpenAI /v1/chat/completions format.
 * 2. Transforms them to OpenAI /v1/chat/completions for OpenRouter.
 * 3. Handles DeepSeek V4 thinking-mode reasoning_content preservation
 *    across multi-turn conversations.
 * 
 * Usage:
 *   node src/server.js [--port 3000]
 */

require('dotenv').config();

// Bootstrap global-agent to make Node.js fetch() respect HTTP_PROXY
require('global-agent').bootstrap();

const express = require('express');
const cors = require('cors');
const { ReasoningStore } = require('./reasoningStore');
const {
  DeepSeekTransformer,
  StreamingConverter,
  requiresReasoningContent,
  fixToolFormat,
  anthropicMessagesToOpenAI,
} = require('./deepseekTransformer');

const DEFAULT_PORT = 3000;
const PORT = parseInt(process.argv.find(a => a.startsWith('--port='))?.split('=')[1] ?? String(DEFAULT_PORT));

// ─── Upstream configuration ────────────────────────────────────────────────
const UPSTREAM_URL = process.env.UPSTREAM_URL || 'https://openrouter.ai/api/v1/chat/completions';
const UPSTREAM_API_KEY = process.env.UPSTREAM_API_KEY || '';
const UPSTREAM_MODEL = process.env.UPSTREAM_MODEL || '';

// ─── Session store: sessionId → ReasonStore ────────────────────────────────
const sessions = new Map();

function getOrCreateSession(sessionId) {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, new ReasoningStore());
  }
  return sessions.get(sessionId);
}

function getTransformer(store) {
  return new DeepSeekTransformer(store);
}

// ─── Auto-prune stale sessions every 5 minutes ─────────────────────────────
const PRUNE_INTERVAL_MS = 5 * 60 * 1000;
const PRUNE_MAX_AGE_MS = 30 * 60 * 1000;
const pruneTimer = setInterval(() => {
  let totalPruned = 0;
  for (const [sessionId, store] of sessions) {
    if (store.isStale(PRUNE_MAX_AGE_MS)) {
      sessions.delete(sessionId);
      totalPruned++;
    }
  }
  if (totalPruned > 0) {
    console.log(`[prune] Removed ${totalPruned} stale sessions (max age: ${PRUNE_MAX_AGE_MS / 1000}s)`);
  }
}, PRUNE_INTERVAL_MS);
// Don't keep the process alive just for pruning
pruneTimer.unref();

// ─── Utility ───────────────────────────────────────────────────────────────

/** Extract session ID from request headers or generate a default */
function resolveSessionId(req) {
  return (
    req.headers['x-session-id'] ||
    req.headers['x-claude-session-id'] ||
    'default'
  );
}

// ─── Message format conversion ─────────────────────────────────────────────

/**
 * Convert Anthropic /v1/messages format to OpenAI /v1/chat/completions format.
 * Claude Code sends body like:
 * {
 *   model: "sonnet",
 *   max_tokens: 4096,
 *   messages: [{ role: "user", content: "..." }],
 *   tools: [...],
 *   system: "..."  OR  system: [{ type: "text", text: "..." }, ...]
 * }
 */
function anthropicToOpenAI(anthropicBody) {
  const { messages, system, model, max_tokens, ...rest } = anthropicBody;

  // Normalize system to a string (Anthropic API sends either a string or an array of text blocks)
  let systemStr = '';
  if (typeof system === 'string') {
    systemStr = system;
  } else if (Array.isArray(system)) {
    systemStr = system
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('\n');
  }

  const systemMsg = systemStr ? [{ role: 'system', content: systemStr }] : [];

  // Convert messages from Anthropic format to OpenAI format
  // (this is the critical missing piece — tool_use/tool_result → tool_calls/role:tool)
  const converted = anthropicMessagesToOpenAI(messages);
  const openAIMessages = [
    ...systemMsg,
    ...converted,
  ];
  console.log(`[DEBUG anthropicToOpenAI] messages: ${messages.length} → ${converted.length} after Anthropic→OpenAI conversion`);

  // Route to DeepSeek model — prefer UPSTREAM_MODEL env, keep original model as fallback
  const targetModel = UPSTREAM_MODEL || model;
  console.log(`[DEBUG anthropicToOpenAI] input model=${model}, targetModel=${targetModel}`);

  return {
    ...rest,
    model: targetModel,
    messages: openAIMessages,
    max_tokens: max_tokens || 4096,
  };
}

/**
 * Convert OpenAI chat completion response to Anthropic Messages response format.
 */
function openAIResponseToAnthropic(openAIResp, model) {
  const msg = openAIResp.choices?.[0]?.message;
  const finishReason = openAIResp.choices?.[0]?.finish_reason;
  if (!msg) return { type: 'message', role: 'assistant', content: [] };

  const content = [];

  if (msg.content) {
    content.push({ type: 'text', text: msg.content });
  }

  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      let input;
      if (tc.function?.arguments) {
        try {
          input = JSON.parse(tc.function.arguments);
        } catch (e) {
          console.error('[openAIResponseToAnthropic] Failed to parse tool arguments JSON:', e.message);
          input = tc.function.arguments;
        }
      } else {
        input = tc.input;
      }

      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function?.name || tc.name,
        input,
      });
    }
  }

  // Map OpenAI finish_reason to Anthropic stop_reason
  const stopReason = finishReason === 'tool_calls' ? 'tool_use'
                   : finishReason === 'length' ? 'max_tokens'
                   : finishReason === 'stop' ? 'end_turn'
                   : null;

  return {
    id: openAIResp.id || `msg_${Date.now()}`,
    type: 'message',
    role: 'assistant',
    model: openAIResp.model || model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: openAIResp.usage?.prompt_tokens || 0,
      output_tokens: openAIResp.usage?.completion_tokens || 0,
    },
  };
}

// ─── Core: unified proxy-to-upstream ───────────────────────────────────────

/**
 * Unified proxy function shared by both /v1/chat/completions and /v1/messages.
 *
 * @param {Request}  req    - Express request
 * @param {Response} res    - Express response
 * @param {Object}   opts
 * @param {string}   opts.targetUrl             - upstream URL
 * @param {string}   opts.apiKey                - authorization key
 * @param {Object}   opts.requestBody           - request body (already in OpenAI format)
 * @param {string}   opts.model                 - model name
 * @param {boolean}  opts.isStreaming           - whether this is a streaming request
 * @param {string}   opts.sessionId             - session identifier
 * @param {Function} [opts.streamConverterFactory] - (model) => converter with .convert(chunk) -> string[]
 *                                                   Pass null for SSE passthrough (OpenAI paths)
 * @param {Function} [opts.responseTransformer] - (openAIJson, model) => response for non-streaming
 *                                                 Pass null for passthrough (OpenAI paths)
 */
async function proxyToUpstream(req, res, {
  targetUrl, apiKey, requestBody, model,
  isStreaming, sessionId,
  streamConverterFactory,
  responseTransformer,
}) {
  const store = getOrCreateSession(sessionId);
  const transformer = getTransformer(store);

  // Fix tools format: add type:"function" if missing (Claude Code omits it)
  if (requestBody.tools) {
    requestBody = { ...requestBody, tools: fixToolFormat(requestBody.tools) };
    console.log('[DEBUG] Fixed tools format, count:', requestBody.tools.length);
  }

  // Phase 1: Outbound transformation (reasoning_content injection)
  const needsTransform = requiresReasoningContent('openrouter', model);
  if (needsTransform) {
    requestBody = {
      ...requestBody,
      messages: transformer.transformOutbound(requestBody.messages),
    };
  }

  // ── Execute upstream request ───────────────────────────────────────────
  const supportsHalfDuplex = typeof ReadableStream !== 'undefined';

  try {
    const fetchOptions = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify(requestBody),
    };

    if (isStreaming && supportsHalfDuplex) {
      fetchOptions.duplex = 'half';
    }

    const response = await fetch(targetUrl, fetchOptions);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[${new Date().toISOString()}] Upstream error ${response.status}:`, errorText.slice(0, 500));
      console.error(`[DEBUG] Failed requestBody:`, JSON.stringify(requestBody).slice(0, 500));
      res.status(response.status).json({
        error: { message: `Upstream error: ${response.status}`, details: errorText },
      });
      return;
    }

    if (isStreaming) {
      // ── Streaming response ───────────────────────────────────────────
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Request-Session', sessionId);

      const converter = streamConverterFactory ? streamConverterFactory(model) : null;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let chunkCount = 0;
      let errorCount = 0;

      /** Process a single parsed SSE chunk — handles both passthrough and conversion */
      function processChunk(chunk) {
        chunkCount++;
        try {
          // Phase 2: Store reasoning_content from streaming chunks
          if (needsTransform) {
            transformer.transformStreamingChunk(chunk);
          }

          if (converter) {
            // Convert OpenAI SSE → Anthropic SSE (with event: lines)
            const events = converter.convert(chunk);
            for (const event of events) {
              const parsed = JSON.parse(event);
              res.write(`event: ${parsed.type}\ndata: ${event}\n\n`);
            }
          } else {
            // Passthrough: relay the raw OpenAI SSE chunk
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }
        } catch (e) {
          errorCount++;
          console.error(`[DEBUG SSE chunk error]`, e.message);
        }
      }

      /** Promise-chain pump: reads stream, buffers lines, dispatches to processChunk */
      function startPump() {
        reader.read().then(({ done, value }) => {
          if (done) {
            console.log(`[DEBUG SSE] stream DONE, totalChunks=${chunkCount}, errors=${errorCount}`);
            if (buffer.trim()) {
              const line = buffer.trim();
              if (line.startsWith('data: ')) {
                const data = line.slice(6).trim();
                if (data !== '[DONE]') {
                  try { processChunk(JSON.parse(data)); } catch (e) {
                    console.error('[DEBUG SSE final parse error]', e.message);
                  }
                }
              }
            }
            // Ensure stop sequence for Anthropic conversion (all blocks)
            if (converter && !converter.sentMessageStop) {
              for (const block of converter._blocks) {
                if (!converter._stopped.has(block.index)) {
                  converter._stopped.add(block.index);
                  res.write(`event: content_block_stop\ndata: {"type":"content_block_stop","index":${block.index}}\n\n`);
                }
              }
              res.write('event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}\n\n');
              res.write('event: message_stop\ndata: {"type":"message_stop"}\n\n');
            }
            res.write('data: [DONE]\n\n');
            res.end();
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep last partial line
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6).trim();
            if (data === '[DONE]') {
              if (converter && !converter.sentMessageStop) {
                for (const block of converter._blocks) {
                  if (!converter._stopped.has(block.index)) {
                    converter._stopped.add(block.index);
                    res.write(`event: content_block_stop\ndata: {"type":"content_block_stop","index":${block.index}}\n\n`);
                  }
                }
                res.write('event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}\n\n');
                res.write('event: message_stop\ndata: {"type":"message_stop"}\n\n');
              }
              res.write('data: [DONE]\n\n');
              res.end();
              reader.cancel().catch(() => {});
              return;
            }
            try {
              processChunk(JSON.parse(data));
            } catch (e) {
              console.error('[DEBUG SSE parse error]', e.message);
            }
          }
          if (converter && converter.sentMessageStop) {
            reader.cancel().catch(() => {});
            res.end();
            return;
          }
          startPump(); // continue the chain
        }).catch((err) => {
          console.error('[DEBUG SSE reader error]', err.message);
          res.end();
        });
      }

      startPump();

    } else {
      // ── Non-streaming response ────────────────────────────────────────
      const json = await response.json();

      // Phase 2: Store reasoning_content from complete response
      if (needsTransform) {
        transformer.transformInbound(json);
      }

      res.setHeader('X-Request-Session', sessionId);
      if (responseTransformer) {
        res.json(responseTransformer(json, model));
      } else {
        res.json(json);
      }
    }

  } catch (err) {
    console.error(`[${new Date().toISOString()}] Proxy error:`, err.message);
    if (!res.headersSent) {
      res.status(502).json({ error: { message: `Proxy error: ${err.message}` } });
    }
  }
}

// ─── Express app ────────────────────────────────────────────────────────────

const app = express();
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: '10mb' }));

// ── Request logging (catch-all for debugging) ──────────────────────────────
app.use((req, res, next) => {
  console.log(`[REQ] ${req.method} ${req.originalUrl} from ${req.ip} headers=${JSON.stringify({
    'content-type': req.headers['content-type'],
    'authorization': req.headers['authorization'] ? 'SET' : 'EMPTY',
    'anthropic-version': req.headers['anthropic-version'],
    'x-api-key': req.headers['x-api-key'] ? 'SET' : 'EMPTY',
  })}`);
  next();
});

// ── Health check ────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({ status: 'ok', sessions: sessions.size, uptime: process.uptime() });
});

// ── OpenAI /v1/chat/completions ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  // API key: pass through from client (Authorization header), .env as fallback
  const apiKey = req.headers['authorization']?.replace('Bearer ', '')
              || UPSTREAM_API_KEY
              || '';
  const requestBody = req.body;
  const isStreaming = requestBody.stream === true;

  console.log(`[DEBUG] /v1/chat/completions model=${requestBody.model}, stream=${isStreaming}`);

  await proxyToUpstream(req, res, {
    targetUrl: UPSTREAM_URL,
    apiKey,
    requestBody,
    model: requestBody.model || '',
    isStreaming,
    sessionId: resolveSessionId(req),
    // OpenAI path: passthrough (no conversion needed)
    streamConverterFactory: null,
    responseTransformer: null,
  });
});

// ── Anthropic /v1/messages (Claude Code primary endpoint) ──────────────────
app.post('/v1/messages', async (req, res) => {
  // API key: pass through from client (x-api-key or Authorization), .env as fallback
  const apiKey = req.headers['x-api-key']
              || req.headers['authorization']?.replace('Bearer ', '')
              || UPSTREAM_API_KEY
              || '';

  const isStreaming = req.body.streaming === true || req.body.stream === true;
  console.log(`[DEBUG] /v1/messages stream=${isStreaming}, hasSystem=${!!req.body.system}, msgCount=${req.body.messages?.length}, tools=${req.body.tools?.length}`);

  // Convert Anthropic → OpenAI format
  const openAIBody = anthropicToOpenAI(req.body);
  const model = openAIBody.model || '';

  await proxyToUpstream(req, res, {
    targetUrl: UPSTREAM_URL,
    apiKey,
    requestBody: openAIBody,
    model,
    isStreaming,
    sessionId: resolveSessionId(req),
    streamConverterFactory: (m) => new StreamingConverter(m),
    responseTransformer: openAIResponseToAnthropic,
  });
});

// ─── Admin endpoints ────────────────────────────────────────────────────────

app.delete('/session/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  if (sessions.has(sessionId)) {
    sessions.delete(sessionId);
    res.json({ ok: true, message: `Session ${sessionId} cleared` });
  } else {
    res.status(404).json({ error: 'Session not found' });
  }
});

app.get('/sessions', (req, res) => {
  res.json({ count: sessions.size, sessionIds: [...sessions.keys()] });
});

// ─── Start ──────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[${new Date().toISOString()}] proxy-deepseek listening on http://localhost:${PORT}`);
  console.log(`[${new Date().toISOString()}] upstream: ${UPSTREAM_URL}`);
  console.log(`[${new Date().toISOString()}] model: ${UPSTREAM_MODEL || '(from request)'}`);
  console.log(`[${new Date().toISOString()}] session pruning: every ${PRUNE_INTERVAL_MS / 1000}s, max age ${PRUNE_MAX_AGE_MS / 1000}s`);
});

// ─── Graceful shutdown ──────────────────────────────────────────────────────

function shutdown(signal) {
  console.log(`[${new Date().toISOString()}] ${signal} received — shutting down`);
  clearInterval(pruneTimer);
  sessions.clear();
  process.exit(0);
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

module.exports = { app };
