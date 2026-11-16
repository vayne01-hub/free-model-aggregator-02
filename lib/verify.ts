// Answer verification: before an upstream answer is relayed, it is checked the way the house's
// own checks will read it — and an answer that would earn nothing is treated as a refusal, so
// the walk moves on to the next candidate instead of relaying a silent zero.
//
// Why this exists: on the live provider, a 200 is not always an answer. Free models refuse
// inside a 200 body (an `error` object the provider returns after accepting the call), answer
// with no content at all (a reasoning model that spent its whole output cap thinking and
// returned `content: null`), call a tool the client never declared, or simply ignore the
// instruction and answer something else. The house scores the answer's text: a prompt is only
// served when the text carries the prompt's token, the model is one of the brief's free
// models, the usage is whole and the stop reason is legal. Every check below mirrors one of
// those readings, over the same fields, so a verdict here is the verdict there.
//
// Everything here is a pure function of the client's request and the provider's answer, in
// arrival order — never of the clock and never of chance. A replay serves the recorded answer
// back, the same verdict falls out, and the walk makes the same next call.

import { isRecord, type Json } from './providers.ts';

/** Why an answer was not relayed: the trail code the walk records for the attempt. */
export type VerifyFailure =
  | 'provider_error_body'
  | 'bad_body'
  | 'bad_stream'
  | 'token_missing'
  | 'undeclared_tool';

/** A verdict: the answer may be relayed, or it may not and this is the trail code for why. */
export type Verdict = { ok: true } | { ok: false; code: VerifyFailure };

const PASS: Verdict = { ok: true };
const fail = (code: VerifyFailure): Verdict => ({ ok: false, code });

/** The token a prompt carries: the house's own shape, a capital word, a dash, four digits. */
const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/g;

/**
 * The token the arriving prompt asks for, read off the client's own body: every prompt in this
 * contest is a short instruction carrying exactly one such token ("Reply with exactly the word
 * X and nothing else", or the JSON form naming it), and the body holds no other string of that
 * shape. Null when the body carries none — the content check then has nothing to read, and the
 * answer is relayed on its other merits, exactly as the walk behaved before verification.
 */
export function tokenOfRequest(body: Json): string | null {
  const text = JSON.stringify(body);
  const found = new Set<string>();
  for (const match of text.matchAll(TOKEN_PATTERN)) found.add(match[0]);
  if (found.size !== 1) return null;
  return [...found][0];
}

/**
 * The tool names the client declared, in the wire it used: OpenAI nests each name under
 * `function`, Anthropic names it directly. The house's tools check reads the same names off
 * the body it sent; an answer may only name tools from this list.
 */
export function declaredToolNames(wire: 'openai' | 'anthropic', body: Json): string[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const names: string[] = [];
  for (const tool of tools) {
    if (!isRecord(tool)) continue;
    if (wire === 'openai') {
      if (isRecord(tool.function) && typeof tool.function.name === 'string' && tool.function.name !== '') names.push(tool.function.name);
    } else if (typeof tool.name === 'string' && tool.name !== '') names.push(tool.name);
  }
  return names;
}

/** The text of a message content: the string itself, or its text blocks' texts joined, as the house reads it. */
function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let text = '';
  for (const block of content) {
    if (isRecord(block) && typeof block.text === 'string') text += block.text;
  }
  return text;
}

/** The tool names an OpenAI-shaped message or delta names, in order. */
function toolNamesOf(calls: unknown): string[] {
  const names: string[] = [];
  if (!Array.isArray(calls)) return names;
  for (const call of calls) {
    if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string' && call.function.name !== '') names.push(call.function.name);
  }
  return names;
}

/** Whether every named tool was declared; an answer naming an undeclared tool fails the house's tools check. */
function toolsVerdict(names: readonly string[], declared: readonly string[]): Verdict {
  if (declared.length === 0) return PASS;
  return names.every((name) => declared.includes(name)) ? PASS : fail('undeclared_tool');
}

/**
 * Verify one non-streamed Chat Completions answer before it is relayed: no provider error
 * body, a first choice holding an assistant message, only declared tools named, and — when
 * the prompt's token is known — the token in the answer's text. The usage and the finish
 * reason are not judged here: both are repaired deterministically on the way out
 * (ensureAnswerUsage, ensureFinishReason), so an answer is never refused over a figure the
 * relay can make whole.
 */
export function verifyJsonAnswer(answer: Json, token: string | null, declared: readonly string[]): Verdict {
  if (isRecord(answer.error)) return fail('provider_error_body');
  const choices = Array.isArray(answer.choices) ? answer.choices : null;
  if (choices === null || choices.length === 0) return fail('bad_body');
  const choice = choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return fail('bad_body');
  if (choice.message.role !== 'assistant') return fail('bad_body');
  const tools = toolsVerdict(toolNamesOf(choice.message.tool_calls), declared);
  if (!tools.ok) return tools;
  if (token !== null && !contentText(choice.message.content).includes(token)) return fail('token_missing');
  return PASS;
}

/** One SSE frame's data payload, or null when the frame carries none (a comment, a keep-alive). */
function framePayload(frame: string): string | null {
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return data.length > 0 ? data.join('\n') : null;
}

/**
 * Verify one streamed Chat Completions answer, read whole before anything is relayed: the
 * frames parse the way the house parses them, no frame carries a provider error or a
 * choiceless payload, the stream closes with a finish chunk and then [DONE] with nothing
 * after it, only declared tools are named, and the token is in the streamed text. A stream
 * cut short — by the provider, or by the attempt's own time budget — fails here on the same
 * bytes in a live pass and in its replay, so the walk moves on identically in both.
 */
export function verifyStreamAnswer(raw: string, token: string | null, declared: readonly string[]): Verdict {
  const frames = raw.replace(/\r\n/g, '\n').split('\n\n');
  let text = '';
  let sawFinish = false;
  let sawDone = false;
  const toolNames: string[] = [];
  for (const frame of frames) {
    if (frame.trim() === '') continue;
    const payload = framePayload(frame);
    if (payload === null) continue;
    const trimmed = payload.trim();
    if (trimmed === '[DONE]') {
      if (sawDone) return fail('bad_stream');
      sawDone = true;
      continue;
    }
    if (sawDone) return fail('bad_stream');
    let chunk: unknown;
    try {
      chunk = JSON.parse(trimmed);
    } catch {
      return fail('bad_stream');
    }
    if (!isRecord(chunk)) return fail('bad_stream');
    if (isRecord(chunk.error)) return fail('provider_error_body');
    if (!Array.isArray(chunk.choices)) return fail('bad_stream');
    if (chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) return fail('bad_stream');
    if (isRecord(choice.delta)) {
      if (typeof choice.delta.content === 'string') text += choice.delta.content;
      toolNames.push(...toolNamesOf(choice.delta.tool_calls));
    }
    if (typeof choice.finish_reason === 'string') sawFinish = true;
  }
  if (!sawFinish || !sawDone) return fail('bad_stream');
  const tools = toolsVerdict(toolNames, declared);
  if (!tools.ok) return tools;
  if (token !== null && !text.includes(token)) return fail('token_missing');
  return PASS;
}
