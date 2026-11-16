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

/**
 * The most gateway calls one prompt may cost, refusals included. Eight is deep enough to walk
 * past a dead ladder head, a rate limit, a silent zero, a hang and a second refusal and still
 * reach the reliable middle of the table; a prompt that has refused eight models is refused by
 * the tier, not by a model, and further calls would spend the key's daily allowance on it.
 * Eight is also the trail's own cap (MAX_TRAIL_ROWS): a walk that fails every attempt trails
 * exactly eight rows, which the rules still read whole. Failed attempts are not counted by
 * the house's frugality tie-break (only answered calls are), so the depth is bought with
 * time — bounded by the budgets below — not with standing.
 */
export const MAX_ATTEMPTS_PER_PROMPT = 8;
/** The trail's own cap: the rules read at most 8 rows, and a longer trail is read as none. */
export const MAX_TRAIL_ROWS = 8;

/**
 * The time budget of one attempt, by its place in the walk, in milliseconds. The house gives
 * a prompt 30 seconds to be answered; a free model that is going to answer usually does so in
 * a few seconds, so an attempt still silent at its budget is one the prompt cannot afford to
 * keep waiting on. All eight budgets together stay under the house's 30 seconds (29 in all),
 * so even a walk that burns every budget gives up before the house hangs up. The budget only
 * ever cuts an attempt short, and a cut attempt replays identically: cut before its answer
 * arrived, the gateway's record of it is a 499 (client closed the request), which a replay
 * hands back as the attempt's response and the walk reads as the same attempt_timeout the
 * live cut produced; cut mid-answer, the record is the partial body itself, and verification's
 * verdict on those bytes (lib/verify.ts) is the same in both passes. No choice the walk makes
 * ever depends on the clock: the budgets are fixed constants, and which attempt a prompt is on
 * is a function of the answers alone.
 */
export const ATTEMPT_TIMEOUT_MS: readonly number[] = [8_000, 5_000, 4_000, 3_000, 3_000, 2_000, 2_000, 2_000];

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
 * Models this entry walks last on every prompt, whatever their published rating: on this
 * entry's key they are not weather but policy. Inkling, the table's top-rated model, answers
 * every call through this key with a 403 — OpenRouter serves it only to named agentic
 * harnesses, which this key is not — and the refusal is slow (seconds, not milliseconds), so
 * a walk that led with it would pay most of its first attempt's budget for a certainty. It
 * stays in the catalog and in the candidate list, at the tail: if every other candidate has
 * failed, the walk still reaches it, and on some future key it may answer there. The set is a
 * constant of the brief's ids, so a replay partitions the same list the live pass did.
 */
export const DEPRIORITIZED_MODELS: ReadonlySet<string> = new Set(['openrouter/thinkingmachines/inkling-2-small:free']);

/**
 * One prompt's candidates with every deprioritized model moved behind all the rest, the
 * incoming order kept inside each half. Pure in the list, so a replay walks the same list.
 */
export function deprioritize(candidates: readonly ModelRow[]): ModelRow[] {
  const head: ModelRow[] = [];
  const tail: ModelRow[] = [];
  for (const model of candidates) {
    if (DEPRIORITIZED_MODELS.has(model.id)) tail.push(model);
    else head.push(model);
  }
  return [...head, ...tail];
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
 * Within the sidelined, a second rule orders the walk: a refusal that names the key or the
 * model itself — a 401, a 403, a 404 — is a hard refusal, a state of the key or the catalogue
 * that the trial's remaining seconds will not change, so hard-refused models walk behind the
 * softly refused (a 429, a 5xx, a timeout, an unusable answer), whose fault may already be
 * over. The ranked order is kept inside each tier.
 *
 * A sidelined model is never struck off entirely: when it is the only candidate left that meets
 * a prompt, the walk still reaches it.
 */
export class RefusalBook {
  private readonly counts = new Map<string, number>();
  private readonly rateLimited = new Set<string>();
  private readonly hardRefused = new Set<string>();

  /** Record one refusal by `modelId`, with the HTTP status the gateway relayed. */
  note(modelId: string, status: number): void {
    this.counts.set(modelId, (this.counts.get(modelId) ?? 0) + 1);
    if (status === 429) this.rateLimited.add(modelId);
    if (status === 401 || status === 403 || status === 404) this.hardRefused.add(modelId);
  }

  /** How often `modelId` has refused in this trial so far. */
  refusalsOf(modelId: string): number {
    return this.counts.get(modelId) ?? 0;
  }

  /** Whether `modelId` has been rate-limited (429) in this trial so far. */
  isRateLimited(modelId: string): boolean {
    return this.rateLimited.has(modelId);
  }

  /** Whether `modelId` has hard-refused (401, 403 or 404) in this trial so far. */
  isHardRefused(modelId: string): boolean {
    return this.hardRefused.has(modelId);
  }

  /** Whether later prompts walk past `modelId` first: it has refused once in this trial. */
  sidelined(modelId: string): boolean {
    return this.isRateLimited(modelId) || this.refusalsOf(modelId) >= 1;
  }

  /**
   * The walk for one prompt: the ranked candidates in three tiers — every model that has not
   * refused, then the softly refused, then the hard-refused — the ranked order kept inside
   * each tier. Pure in the book's state, so a replay walks the same list.
   */
  order(candidates: readonly ModelRow[]): ModelRow[] {
    const clean: ModelRow[] = [];
    const soft: ModelRow[] = [];
    const hard: ModelRow[] = [];
    for (const model of candidates) {
      if (this.isHardRefused(model.id)) hard.push(model);
      else if (this.sidelined(model.id)) soft.push(model);
      else clean.push(model);
    }
    return [...clean, ...soft, ...hard];
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
