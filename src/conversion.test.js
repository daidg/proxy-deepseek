/**
 * Tests for proxy-deepseek core conversion functions.
 * Run: node --test src/conversion.test.js
 */
const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  anthropicMessagesToOpenAI,
  fixToolFormat,
} = require('./deepseekTransformer');

// ═══════════════════════════════════════════════════════════════════════════════
// anthropicMessagesToOpenAI
// ═══════════════════════════════════════════════════════════════════════════════

describe('anthropicMessagesToOpenAI', () => {

  it('passes through string-content user messages unchanged', () => {
    const input = [
      { role: 'user', content: 'hello' },
    ];
    const output = anthropicMessagesToOpenAI(input);
    assert.equal(output.length, 1);
    assert.equal(output[0].role, 'user');
    assert.equal(output[0].content, 'hello');
  });

  it('converts assistant with text-only content blocks', () => {
    const input = [
      { role: 'assistant', content: [{ type: 'text', text: 'Hello world' }] },
    ];
    const output = anthropicMessagesToOpenAI(input);
    assert.equal(output.length, 1);
    assert.equal(output[0].role, 'assistant');
    assert.equal(output[0].content, 'Hello world');
    assert.equal(output[0].tool_calls, undefined);
  });

  it('converts assistant with tool_use blocks → tool_calls', () => {
    const input = [{
      role: 'assistant',
      content: [
        { type: 'tool_use', id: 'toolu_abc123', name: 'bash', input: { cmd: 'ls' } },
      ],
    }];
    const output = anthropicMessagesToOpenAI(input);
    assert.equal(output.length, 1);
    assert.equal(output[0].role, 'assistant');
    assert.equal(output[0].content, ''); // no text, only tool use
    assert.ok(output[0].tool_calls);
    assert.equal(output[0].tool_calls.length, 1);
    assert.equal(output[0].tool_calls[0].id, 'toolu_abc123');
    assert.equal(output[0].tool_calls[0].type, 'function');
    assert.equal(output[0].tool_calls[0].function.name, 'bash');
    assert.equal(output[0].tool_calls[0].function.arguments, '{"cmd":"ls"}');
  });

  it('converts assistant with both text and tool_use blocks', () => {
    const input = [{
      role: 'assistant',
      content: [
        { type: 'text', text: 'Let me run a command.' },
        { type: 'tool_use', id: 'toolu_xyz', name: 'bash', input: { cmd: 'pwd' } },
      ],
    }];
    const output = anthropicMessagesToOpenAI(input);
    assert.equal(output[0].content, 'Let me run a command.');
    assert.equal(output[0].tool_calls.length, 1);
    assert.equal(output[0].tool_calls[0].function.name, 'bash');
  });

  it('splits user content blocks: tool results BEFORE text', () => {
    const input = [{
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 'toolu_abc123', content: 'file1.txt\nfile2.txt' },
        { type: 'tool_result', tool_use_id: 'toolu_xyz', content: '/home/user' },
        { type: 'text', text: 'Here are the results.' },
      ],
    }];
    const output = anthropicMessagesToOpenAI(input);
    assert.equal(output.length, 3);
    // tool messages come FIRST (order preserved)
    assert.equal(output[0].role, 'tool');
    assert.equal(output[0].tool_call_id, 'toolu_abc123');
    assert.equal(output[0].content, 'file1.txt\nfile2.txt');
    assert.equal(output[1].role, 'tool');
    assert.equal(output[1].tool_call_id, 'toolu_xyz');
    assert.equal(output[1].content, '/home/user');
    // text comes AFTER tool messages
    assert.equal(output[2].role, 'user');
    assert.equal(output[2].content, 'Here are the results.');
  });

  it('handles tool_result with object content → JSON stringifies', () => {
    const input = [{
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 'toolu_obj', content: { files: ['a.txt'], status: 'ok' } },
      ],
    }];
    const output = anthropicMessagesToOpenAI(input);
    assert.equal(output[0].role, 'tool');
    assert.equal(output[0].content, '{"files":["a.txt"],"status":"ok"}');
  });

  it('handles empty content array on assistant → content: ""', () => {
    const input = [
      { role: 'assistant', content: [] },
    ];
    const output = anthropicMessagesToOpenAI(input);
    assert.equal(output[0].role, 'assistant');
    assert.equal(output[0].content, '');
  });

  it('handles non-array input gracefully', () => {
    const output = anthropicMessagesToOpenAI(null);
    assert.equal(output, null);
  });

  it('multi-turn: assistant(tool_use) → user(tool_results+text) → assistant(text)', () => {
    const input = [
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'tc1', name: 'read', input: { path: '/f' } }],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'tc1', content: 'file contents' },
          { type: 'text', text: 'Continue.' },
        ],
      },
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'The file says...' }],
      },
    ];
    const output = anthropicMessagesToOpenAI(input);
    // Turn 1: assistant with tool_calls
    assert.equal(output[0].role, 'assistant');
    assert.equal(output[0].tool_calls[0].function.name, 'read');
    // Turn 2: tool message THEN user text
    assert.equal(output[1].role, 'tool');
    assert.equal(output[1].tool_call_id, 'tc1');
    assert.equal(output[2].role, 'user');
    assert.equal(output[2].content, 'Continue.');
    // Turn 3: assistant text
    assert.equal(output[3].role, 'assistant');
    assert.equal(output[3].content, 'The file says...');
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// fixToolFormat
// ═══════════════════════════════════════════════════════════════════════════════

describe('fixToolFormat', () => {

  it('wraps Claude Code flat format {name,description,parameters}', () => {
    const input = [
      { name: 'bash', description: 'Run commands', parameters: { type: 'object', properties: {} } },
    ];
    const output = fixToolFormat(input);
    assert.equal(output.length, 1);
    assert.deepEqual(output[0], {
      type: 'function',
      function: {
        name: 'bash',
        description: 'Run commands',
        parameters: { type: 'object', properties: {} },
      },
    });
  });

  it('passes through already-correct OpenAI format', () => {
    const input = [{
      type: 'function',
      function: { name: 'bash', description: 'Run', parameters: {} },
    }];
    const output = fixToolFormat(input);
    assert.deepEqual(output[0], input[0]);
  });

  it('handles tool with missing description → empty string', () => {
    const input = [
      { name: 'read', parameters: { type: 'object' } },
    ];
    const output = fixToolFormat(input);
    assert.equal(output[0].function.description, '');
  });

  it('handles tool with missing parameters → default object', () => {
    const input = [
      { name: 'read', description: 'Read files' },
    ];
    const output = fixToolFormat(input);
    assert.deepEqual(output[0].function.parameters, { type: 'object', properties: {} });
  });

  it('passes through non-array input', () => {
    assert.equal(fixToolFormat(null), null);
    assert.equal(fixToolFormat(undefined), undefined);
  });

  it('handles mixed format: some OpenAI some Claude Code', () => {
    const input = [
      { type: 'function', function: { name: 'existing', description: 'ok', parameters: {} } },
      { name: 'new_tool', description: 'A new tool', parameters: {} },
    ];
    const output = fixToolFormat(input);
    // First passes through
    assert.equal(output[0].function.name, 'existing');
    // Second gets wrapped
    assert.equal(output[1].type, 'function');
    assert.equal(output[1].function.name, 'new_tool');
  });

  it('idempotent: running twice produces same result', () => {
    const input = [
      { name: 'bash', description: 'Run', parameters: { type: 'object' } },
    ];
    const first = fixToolFormat(input);
    const second = fixToolFormat(first);
    assert.deepEqual(second[0], first[0]);
  });
});
