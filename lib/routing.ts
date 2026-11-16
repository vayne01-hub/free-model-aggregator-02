// Routing state for one trial: which models have refused, in what order to walk the ranked
// candidates, and the refusal trail a recovered prompt reports on its response.
//
// Everything here is a function of the brief and of the gateway's answers, in the order they
// arrived — never of the clock and never of chance. A replay serves the same answers back, so
// the same walk happens again and the export a replay compares is the live pass's own.
//
// The entry runs on a free-tier key (about 50 free-model requests a day), so the walk is
// deliberately frugal: candidates are dialled one at a time, a call is spent only on an actual
// refusal — never speculatively, never in parallel — and a model that has been rate-limited
// once is walked past for the rest of the trial unless nothing else can serve the prompt.

import { isRecord, type ModelRow } from './providers.ts';

/** The most gateway calls one prompt may cost, refusals included. */
export const MAX_ATTEMPTS_PER_PROMPT = 4;
/** The trail's own cap: the rules read at most 8 rows, and a longer trail is read as none. */
export const MAX_TRAIL_ROWS = 8;

/**
 * The time budget of one attempt, by its place in the walk, in milliseconds. The house gives
 * a prompt 30 seconds to be answered; a free model that is going to answer usually does so in
 * a few seconds, so an attempt still silent at its budget is one the prompt cannot afford to
 * keep waiting on. The budget only ever cuts an attempt short: the gateway tears the call down
 * when this server lets go, the house's record keeps the partial answer that arrived, and the
 * walk's verdict on a partial answer (lib/verify.ts) is the same in a live pass and in its
 * replay — so the attempt the budget ended is walked past identically in both, and no choice
 * the walk makes ever depends on the clock: the budgets are fixed constants, and which attempt
 * a prompt is on is a function of the answers alone.
 */
export const ATTEMPT_TIMEOUT_MS: readonly number[] = [15_000, 9_000, 6_000, 6_000];

/** The budget of the attempt with this 0-based index in its prompt's walk. */
export function attemptTimeoutMs(attemptIndex: number): number {
  return ATTEMPT_TIMEOUT_MS[Math.min(attemptIndex, ATTEMPT_TIMEOUT_MS.length - 1)];
}

/** One row of the `x-makefun-upstream-failures` trail, in the rules' own shape. */
export interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

/**
 * The code a refusal carries, read off the provider's own error body: its `error.code` when it
 * has one (a string, or a number spelled out), else its `error.type`, else the HTTP status
 * spelled as `http_<status>`. The gateway relays a provider's refusal as the provider sent it,
 * so the body is the provider's own; a body that is not that shape still leaves a usable code.
 */
export function refusalCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = parsed.error.code;
      if (typeof code === 'string' && code !== '') return code;
      if (typeof code === 'number' && Number.isFinite(code)) return String(code);
      const type = parsed.error.type;
      if (typeof type === 'string' && type !== '') return type;
    }
  } catch {
    // Not JSON at all: the status alone names it.
  }
  return `http_${status}`;
}

/**
 * The trial's memory of refusals, by model id. One rule sidelines a model for later prompts:
 *
 *   - any refusal sidelines it at once. A trial lasts a few minutes at most, and a free model
 *     that has just refused — rate-limited (20 requests a minute, and a small daily allowance
 *     behind this entry's key), down at its provider, or answering unusably — is in the same
 *     state when the next prompt arrives seconds later, so a later prompt that led with it
 *     would pay the same refusal again, in the coin the trial is shortest of: the 30 seconds
 *     the house gives each prompt. Fast demotion costs little when the refusal was a moment's
 *     fault: the sidelined model keeps its ranked place among the sidelined, and the walk
 *     reaches it again inside the same prompt's own attempt budget only when it is that
 *     prompt's own candidate — for later prompts it stands behind every model that has not
 *     refused, which is where a recovered prompt wants its first call spent anyway.
 *
 * A sidelined model is never struck off entirely: when it is the only candidate left that meets
 * a prompt, the walk still reaches it.
 */
export class RefusalBook {
  private readonly counts = new Map<string, number>();
  private readonly rateLimited = new Set<string>();

  /** Record one refusal by `modelId`, with the HTTP status the gateway relayed. */
  note(modelId: string, status: number): void {
    this.counts.set(modelId, (this.counts.get(modelId) ?? 0) + 1);
    if (status === 429) this.rateLimited.add(modelId);
  }

  /** How often `modelId` has refused in this trial so far. */
  refusalsOf(modelId: string): number {
    return this.counts.get(modelId) ?? 0;
  }

  /** Whether `modelId` has been rate-limited (429) in this trial so far. */
  isRateLimited(modelId: string): boolean {
    return this.rateLimited.has(modelId);
  }

  /** Whether later prompts walk past `modelId` first: it has refused once in this trial. */
  sidelined(modelId: string): boolean {
    return this.isRateLimited(modelId) || this.refusalsOf(modelId) >= 1;
  }

  /**
   * The walk for one prompt: the ranked candidates with every sidelined model moved behind
   * every model that is not, the ranked order kept inside each half. Pure in the book's state,
   * so a replay walks the same list.
   */
  order(candidates: readonly ModelRow[]): ModelRow[] {
    const clean: ModelRow[] = [];
    const sidelined: ModelRow[] = [];
    for (const model of candidates) {
      if (this.sidelined(model.id)) sidelined.push(model);
      else clean.push(model);
    }
    return [...clean, ...sidelined];
  }
}

/**
 * The trail header's value for one prompt: the refusal rows in the order the calls refused, at
 * most MAX_TRAIL_ROWS of them, as compact JSON. Null when nothing refused, so a clean prompt's
 * response carries no header at all.
 */
export function trailHeader(rows: readonly TrailRow[]): string | null {
  if (rows.length === 0) return null;
  return JSON.stringify(rows.slice(0, MAX_TRAIL_ROWS));
}
