/**
 * ReasoningStore
 * 
 * Stores reasoning_content from DeepSeek V4 thinking-mode responses,
 * keyed by tool_call IDs, so it can be re-injected on subsequent turns.
 * 
 * DeepSeek V4 thinking mode rule: when an assistant turn performed tool calls,
 * the reasoning_content from that turn MUST be sent back verbatim in the
 * assistant message on every subsequent request. Claude Code / OpenAI clients
 * don't do this automatically, so we handle it ourselves.
 * 
 * DESIGN NOTE: This store is per-session (one instance per conversation).
 * Session lifecycle (creation, pruning) is managed by server.js.
 */

const DEBUG = process.env.DEBUG === '1' || process.env.DEBUG === 'true';

class ReasoningStore {
  constructor() {
    /** @type {Map<string, string>} toolCallId → reasoning_content */
    this._entries = new Map();
    /** @type {number} last access timestamp (ms) */
    this.lastAccess = Date.now();
  }

  // ── Public interface ──────────────────────────────────────────────────────

  /** Number of stored reasoning entries. */
  get size() {
    return this._entries.size;
  }

  /**
   * Save reasoning_content for a given tool_call ID.
   * Keeps the first (most complete) reasoning_content — subsequent calls
   * for the same toolCallId are no-ops.
   * 
   * @param {string} toolCallId - the id field of the tool_call
   * @param {string} reasoningContent - the reasoning_content string to preserve
   */
  save(toolCallId, reasoningContent) {
    if (!this._entries.has(toolCallId) && reasoningContent) {
      this._entries.set(toolCallId, reasoningContent);
    }
    this.lastAccess = Date.now();
  }

  /**
   * Retrieve stored reasoning_content for a given tool_call ID.
   * @param {string} toolCallId
   * @returns {string|null}
   */
  get(toolCallId) {
    this.lastAccess = Date.now();
    return this._entries.get(toolCallId) ?? null;
  }

  /**
   * Get all stored reasoning entries as a shallow copy.
   * @returns {Map<string, string>}
   */
  getAll() {
    this.lastAccess = Date.now();
    return new Map(this._entries);
  }

  /**
   * Clear all stored reasoning.
   */
  clear() {
    this._entries.clear();
  }

  /**
   * Check if this store has been idle longer than maxAgeMs.
   * @param {number} maxAgeMs
   * @returns {boolean}
   */
  isStale(maxAgeMs) {
    return (Date.now() - this.lastAccess) > maxAgeMs;
  }
}

module.exports = { ReasoningStore };
