/**
 * DeepSeek Transformer
 * 
 * Two-phase transformer for DeepSeek V4 thinking mode:
 * 
 * Phase 1 — OUTBOUND (request → DeepSeek):
 *   Inject stored reasoning_content into assistant messages that have tool_calls.
 *   The presence of reasoning_content tells DeepSeek this turn is a "continuation"
 *   of a thinking-mode conversation that involved tools.
 * 
 * Phase 2 — INBOUND (response ← DeepSeek):
 *   Extract reasoning_content from the response and store it by tool_call ID.
 *   This will be injected on the NEXT request in the same session.
 */

const { ReasoningStore } = require('./reasoningStore');

const DEBUG = process.env.DEBUG === '1' || process.env.DEBUG === 'true';

/**
 * DeepSeek transformer state per session.
 * @param {string} sessionId
 * @param {ReasoningStore} store
 */
class DeepSeekTransformer {
  constructor(store) {
    this.store = store;
  }

  /**
   * Phase 1: Transform outbound messages.
   *
   * Handles two DeepSeek V4 API requirements:
   * 1. reasoning_content injection: when an assistant turn with tool_calls is
   *    the LAST message, inject stored reasoning_content so the model continues
   *    thinking across turns.
   * 2. content: null fix: OpenAI API rejects assistant messages where
   *    content is null and tool_calls are present. Convert to "".
   *
   * @param {Array} messages - OpenAI-format messages array
   * @returns {Array} transformed messages
   */
  transformOutbound(messages) {
    // Guard: messages must be an array
    if (!Array.isArray(messages)) {
      return messages;
    }

    // Pass 1: Fix content: null → "" for ALL assistant messages with tool_calls
    // (DeepSeek rejects null content with tool_calls regardless of position)
    if (DEBUG) {
      const before = JSON.stringify(messages);
      messages = this._fixNullContent(messages);
      const after = JSON.stringify(messages);
      if (before !== after) {
        console.log('[DEBUG transformOutbound] Fixed null content in assistant message');
      }
    } else {
      messages = this._fixNullContent(messages);
    }

    // Pass 2: Convert Claude Code's [TOOL_CALLS_START] user messages to OpenAI role:tool messages.
    messages = convertToolResults(messages);

    // Inject reasoning_content into ALL assistant messages with tool_calls
    // that have stored reasoning. DeepSeek requires reasoning_content on every
    // assistant(tool_calls) message in the history, not just the last one.
    const stored = this.store.getAll();
    if (DEBUG) console.log(`[DEBUG transformOutbound] storedReasoning size=${stored.size} keys=[${[...stored.keys()].join(',')}]`);
    if (stored.size === 0) return messages;

    messages = messages.map(msg => {
      if (msg.role !== 'assistant' || !msg.tool_calls?.length) return msg;

      // Collect reasoning_content for this message's tool_calls
      const reasoningContents = [];
      for (const tc of msg.tool_calls) {
        if (tc.id) {
          const rc = stored.get(tc.id);
          if (rc) reasoningContents.push(rc);
        }
      }

      if (DEBUG) console.log(`[DEBUG transformOutbound] assistant[${msg.tool_calls?.map(t=>t.id)?.join(',')}] matched ${reasoningContents.length} reasoning(s)`);

      if (reasoningContents.length === 0) return msg;

      // Merge all reasoning into one (take longest)
      const reasoning = reasoningContents.sort((a, b) => b.length - a.length)[0];

      if (DEBUG) console.log(`[DEBUG transformOutbound] INJECTED reasoning_content (${reasoning.length} chars)`);
      return { ...msg, reasoning_content: reasoning };
    });

    return messages;
  }

  /**
   * Phase 2: Transform inbound response.
   * Extract reasoning_content from assistant message and store by tool_call ID.
   *
   * @param {Object} response - OpenAI chat completion chunk/response
   * @returns {Object} unchanged response
   */
  transformInbound(response) {
    // Handle both streaming (chunk) and non-streaming responses
    const assistantMessage = response.choices?.[0]?.message;
    if (!assistantMessage) return response;

    const rc = assistantMessage.reasoning_content || assistantMessage.reasoning;
    const { tool_calls } = assistantMessage;
    
    if (tool_calls?.length && rc) {
      for (const tc of tool_calls) {
        if (tc.id) {
          this.store.save(tc.id, rc);
        }
      }
    }

    return response;
  }

  /**
   * Streaming Phase 2: Extract from streaming chunks.
   * DeepSeek streams reasoning_content and tool_calls in SEPARATE delta chunks.
   * We accumulate reasoning_content and associate it with tool_calls by their ID.
   *
   * @param {Object} chunk - SSE chunk from streaming response
   * @returns {Object} chunk unchanged
   */
  transformStreamingChunk(chunk) {
    const delta = chunk.choices?.[0]?.delta;
    if (!delta) return chunk;

    if (DEBUG) {
      const deltaKeys = Object.keys(delta);
      const hasRC = !!delta.reasoning_content;
      const hasTC = delta.tool_calls?.length > 0;
      if (hasRC || hasTC) {
        console.log(`[DEBUG streamChunk] delta keys=${deltaKeys.join(',')} hasRC=${hasRC} hasTC=${hasTC} pendingRC=${(this._pendingReasoning || '').length} toolIds=${delta.tool_calls?.map(t=>t.id)?.join(',')||'none'}`);
      }
    }

    // Accumulate reasoning_content — arrives as delta.reasoning in DeepSeek V4 chunks
    if (delta.reasoning) {
      this._pendingReasoning = (this._pendingReasoning || '') + delta.reasoning;
    }

    // When tool_calls arrive, associate accumulated reasoning_content
    if (delta.tool_calls?.length && this._pendingReasoning) {
      for (const tc of delta.tool_calls) {
        if (tc.id) {
          this.store.save(tc.id, this._pendingReasoning);
          if (DEBUG) console.log(`[DEBUG streamChunk] SAVED reasoning_content (${this._pendingReasoning.length} chars) for tool_call ${tc.id}`);
        }
      }
    }

    return chunk;
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _fixNullContent(messages) {
    return messages.map(msg => {
      if (msg.role === 'assistant' && msg.tool_calls?.length) {
        if (msg.content === null || msg.content === undefined) {
          return { ...msg, content: '' };
        }
      }
      return msg;
    });
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Streaming Converter (OpenAI SSE → Anthropic SSE)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Streaming converter that properly handles the full Anthropic SSE event sequence.
 * 
 * OpenAI streaming yields chunks like:
 *   { choices: [{ delta: { role: 'assistant', content: '...' } }] }
 *   { choices: [{ delta: { tool_calls: [...] } }] }
 *   { choices: [{ delta: { reasoning_content: '...' } }] }
 *   { choices: [{ finish_reason: 'tool_calls' }] }
 * 
 * Anthropic streaming expects:
 *   message_start        → { type: 'message_start', message: { id, role, content, ... } }
 *   content_block_start → { type: 'content_block_start', index, content_block: { type: 'tool_use', id, name } }
 *   content_block_delta → { type: 'content_block_delta', index, delta: { content: '...' } or { partial_json: '...' } }
 *   content_block_stop  → { type: 'content_block_stop', index }
 *   message_delta       → { type: 'message_delta', delta: { ... }, usage: { ... } }
 *   message_stop        → { type: 'message_stop' }
 */
class StreamingConverter {
  constructor(model) {
    this.model = model;
    this.messageId = `msg_${Date.now()}`;
    this.sentMessageStart = false;
    this.sentMessageDelta = false;
    this.sentMessageStop = false;

    // Track content blocks with proper indices.
    this._blocks = [];           // [{index, type, id?, name?}] — order matters
    this._blockIndices = {};     // type → block lookup for fast access
    this._started = new Set();   // blocks whose content_block_start has been sent
    this._stopped = new Set();   // blocks whose content_block_stop has been sent

    // Accumulate usage from final chunk
    this._usage = null;
  }

  /**
   * Convert an OpenAI SSE chunk to one or more Anthropic SSE events.
   * Returns an array of strings (JSON-serialized events).
   */
  convert(chunk) {
    const events = [];
    const delta = chunk.choices?.[0]?.delta;
    const finishReason = chunk.choices?.[0]?.finish_reason;

    // Capture usage from final chunk for message_delta
    if (chunk.usage) {
      this._usage = chunk.usage;
    }

    // ── 1. message_start (send once at the beginning) ──────────────────────
    if (!this.sentMessageStart) {
      this.sentMessageStart = true;
      events.push(JSON.stringify({
        type: 'message_start',
        message: {
          id: this.messageId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        }
      }));
    }

    // ── 2. Tool call block start ──────────────────────────────────────────
    if (delta?.tool_calls?.length) {
      const tc = delta.tool_calls[0];
      const block = this._ensureBlock('tool_use', tc.id || `toolu_${Date.now()}`, tc.function?.name || tc.name || 'unknown');
      if (!this._started.has(block.index)) {
        this._started.add(block.index);
        events.push(JSON.stringify({
          type: 'content_block_start',
          index: block.index,
          content_block: {
            type: 'tool_use',
            id: block.id,
            name: block.name,
            input: {},
          }
        }));
      }
    }

    // ── 3a. Text content delta ────────────────────────────────────────────
    const textContent = (delta?.content !== undefined && delta.content !== '')
      ? delta.content
      : (delta?.reasoning || '');
    if (textContent) {
      const block = this._ensureBlock('text');
      if (!this._started.has(block.index)) {
        this._started.add(block.index);
        events.push(JSON.stringify({
          type: 'content_block_start',
          index: block.index,
          content_block: { type: 'text', text: '' }
        }));
      }
      events.push(JSON.stringify({
        type: 'content_block_delta',
        index: block.index,
        delta: { type: 'text_delta', text: textContent }
      }));
    }

    // ── 3b. Tool use delta ────────────────────────────────────────────────
    if (delta?.tool_calls?.length) {
      const tc = delta.tool_calls[0];
      if (tc.function?.arguments) {
        const block = this._blockIndices['tool_use'];
        const idx = block ? block.index : 0;
        events.push(JSON.stringify({
          type: 'content_block_delta',
          index: idx,
          delta: { type: 'input_json_delta', partial_json: tc.function.arguments }
        }));
      }
    }

    // ── 4. reasoning (DeepSeek V4 thinking — arrives as delta.reasoning via OpenRouter)
    if (delta?.reasoning) {
      const block = this._ensureBlock('text');
      events.push(JSON.stringify({
        type: 'content_block_delta',
        index: block.index,
        delta: { type: 'thinking_delta', thinking: delta.reasoning }
      }));
    }

    // ── 5. Finish: stop ALL active content blocks, then message_delta + message_stop
    if ((finishReason === 'stop' || finishReason === 'tool_calls' || finishReason === 'length') && !this.sentMessageStop) {
      // Stop all blocks that haven't been stopped yet
      for (const block of this._blocks) {
        if (!this._stopped.has(block.index)) {
          this._stopped.add(block.index);
          events.push(JSON.stringify({ type: 'content_block_stop', index: block.index }));
        }
      }

      if (!this.sentMessageDelta) {
        this.sentMessageDelta = true;
        // Map OpenAI finish_reason to Anthropic stop_reason
        const stopReason = finishReason === 'tool_calls' ? 'tool_use'
                         : finishReason === 'length' ? 'max_tokens'
                         : 'end_turn';
        const outputTokens = this._usage?.completion_tokens
                          || this._usage?.output_tokens
                          || 0;

        events.push(JSON.stringify({
          type: 'message_delta',
          delta: { stop_reason: stopReason, stop_sequence: null },
          usage: { output_tokens: outputTokens },
        }));
      }

      this.sentMessageStop = true;
      events.push(JSON.stringify({ type: 'message_stop' }));
    }

    return events;
  }

  // ── Internal ────────────────────────────────────────────────────────────

  /** Get or create a content block of the given type. Returns {index, type, id?, name?}. */
  _ensureBlock(type, id, name) {
    if (this._blockIndices[type]) return this._blockIndices[type];
    const block = { index: this._blocks.length, type, ...(id ? { id } : {}), ...(name ? { name } : {}) };
    this._blocks.push(block);
    this._blockIndices[type] = block;
    return block;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Determine if a request is heading to a DeepSeek thinking model.
 * @param {string} model 
 * @returns {boolean}
 */
function isDeepSeekThinkingModel(model) {
  if (!model) return false;
  const thinkingModels = [
    'deepseek-reasoner',
    'deepseek-v4',
    'deepseek-v3.2',
    'deepseek-v3.1',
  ];
  const lower = model.toLowerCase();
  return thinkingModels.some(m => lower.includes(m));
}

/**
 * Determine if a request is heading to a provider that requires
 * OpenAI-compatible reasoning_content handling.
 * 
 * @param {string} provider - 'deepseek', 'openrouter', 'openai', etc.
 * @param {string} model
 * @returns {boolean}
 */
function requiresReasoningContent(provider, model) {
  // DeepSeek official API with thinking models
  if (provider === 'deepseek' && isDeepSeekThinkingModel(model)) {
    return true;
  }
  // OpenRouter routing to DeepSeek
  if (model?.toLowerCase().includes('deepseek') && isDeepSeekThinkingModel(model)) {
    return true;
  }
  return false;
}

/**
 * Fix Claude Code's tools format → DeepSeek/OpenAI format.
 * Claude Code sends: { name, description, parameters } or { type, name, description, parameters }
 * DeepSeek expects: { type: "function", function: { name, description, parameters } }
 */
function fixToolFormat(tools) {
  if (!Array.isArray(tools)) return tools;
  return tools.map(tool => {
    // Already in OpenAI format: { type: "function", function: { name, description, parameters } }
    if (tool.function && typeof tool.function === 'object') return tool;
    // Has nested function object with different key
    if (tool.function?.name || tool.function?.description) return tool;
    // Claude Code format: { name, description, parameters } or { type, name, description, parameters }
    return {
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description || '',
        parameters: tool.parameters || { type: 'object', properties: {} },
      },
    };
  });
}

/**
 * Convert Claude Code's [TOOL_CALLS_START] user messages to OpenAI role:tool messages.
 *
 * Claude Code format:
 *   {role: "user", content: "[TOOL_CALLS_START]\n{\"tool\": \"fn\", \"tool_input\": {...}, \"tool_call_id\": \"abc\"}"}
 *
 * OpenAI format (DeepSeek expects):
 *   {role: "tool", tool_call_id: "abc", content: "..."}
 */
function convertToolResults(messages) {
  const TOOL_CALLS_START = '[TOOL_CALLS_START]';

  return messages.map((msg, idx) => {
    if (msg.role !== 'user') return msg;
    if (typeof msg.content !== 'string') return msg;
    if (!msg.content.startsWith(TOOL_CALLS_START)) return msg;

    // Extract the JSON payload after "[TOOL_CALLS_START]\n"
    const jsonStr = msg.content.slice(TOOL_CALLS_START.length).trim();
    let payload;
    try {
      payload = JSON.parse(jsonStr);
    } catch {
      console.log('[DEBUG convertToolResults] Failed to parse JSON:', jsonStr.slice(0, 100));
      return msg;
    }

    const { tool, tool_input, tool_call_id } = payload;
    if (!tool_call_id) {
      console.log('[DEBUG convertToolResults] No tool_call_id in payload');
      return msg;
    }

    // Serialize tool_input back to a string content
    const content = typeof tool_input === 'string' ? tool_input : JSON.stringify(tool_input);

    console.log(`[DEBUG convertToolResults] Converting user msg ${idx} → role:tool tool_call_id=${tool_call_id}, tool=${tool}`);

    return {
      role: 'tool',
      tool_call_id,
      content,
    };
  });
}

/**
 * Convert messages from Anthropic format to OpenAI format.
 * Handles the critical conversion that was missing:
 *
 * Anthropic assistant:  {role:"assistant", content:[{type:"tool_use",id,name,input}, {type:"text",text}...]}
 *   → OpenAI:           {role:"assistant", content:"text", tool_calls:[{id,type:"function",function:{name,arguments}}]}
 *
 * Anthropic user:      {role:"user", content:[{type:"tool_result",tool_use_id,content}, {type:"text",text}...]}
 *   → OpenAI:           [{role:"tool", tool_call_id, content}, {role:"user", content:"text"}]
 *
 * String content messages pass through unchanged.
 *
 * @param {Array} messages - Anthropic-format messages array
 * @returns {Array} OpenAI-format messages array (may have different length)
 */
function anthropicMessagesToOpenAI(messages) {
  if (!Array.isArray(messages)) return messages;

  const result = [];
  for (const msg of messages) {
    if (msg.role === 'assistant') {
      result.push(_convertAssistantMessage(msg));
    } else if (msg.role === 'user' && Array.isArray(msg.content)) {
      // User messages with content blocks: split into tool + text messages.
      // CRITICAL: tool messages must come BEFORE text messages so that
      // assistant(tool_calls) is immediately followed by role:tool messages.
      // DeepSeek rejects: assistant(tool_calls) → user(text) → role:tool
      const textBlocks = msg.content.filter(c => c.type === 'text');
      const toolBlocks = msg.content.filter(c => c.type === 'tool_result');

      // Emit tool messages FIRST (before any text), preserving order
      for (const block of toolBlocks) {
        const content = typeof block.content === 'string'
          ? block.content
          : (block.content ? JSON.stringify(block.content) : '');
        const callId = block.tool_use_id || block.tool_call_id;
        if (callId) {
          result.push({ role: 'tool', tool_call_id: callId, content });
        }
      }
      // Emit text blocks AFTER tool messages
      if (textBlocks.length) {
        result.push({ role: 'user', content: textBlocks.map(c => c.text).join('') });
      }
    } else {
      // String content or unknown format — pass through
      result.push(msg);
    }
  }
  return result;
}

/**
 * Convert a single Anthropic-format assistant message to OpenAI format.
 * @param {Object} msg - {role:"assistant", content:[...]}
 * @returns {Object} - {role:"assistant", content:"...", tool_calls?: [...]}
 */
function _convertAssistantMessage(msg) {
  // Already string content (OpenAI format) — pass through
  if (typeof msg.content === 'string') return msg;

  // Handle content blocks array
  const content = Array.isArray(msg.content) ? msg.content : [];
  const textParts = content.filter(c => c.type === 'text');
  const toolParts = content.filter(c => c.type === 'tool_use');

  const textContent = textParts.map(c => c.text).join('') || '';

  const result = { role: 'assistant', content: textContent };

  if (toolParts.length) {
    result.tool_calls = toolParts.map(tc => ({
      id: tc.id,
      type: 'function',
      function: {
        name: tc.name,
        arguments: JSON.stringify(tc.input),
      },
    }));
  }

  return result;
}

module.exports = { DeepSeekTransformer, StreamingConverter, isDeepSeekThinkingModel, requiresReasoningContent, fixToolFormat, anthropicMessagesToOpenAI };
