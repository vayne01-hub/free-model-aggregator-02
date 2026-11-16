// free-model-aggregator-02: one server, two client wires, and a real free model behind every
// answer — routed by the brief's own ratings, failed over on refusals, and frugal with calls.
//
// THE SHAPE OF A TRIAL
//   The harness runs this file as `node /bot/main.ts` from the repository root, Node 22 with its
//   standard library alone (TypeScript runs directly; a relative import needs its .ts extension).
//   The env is at $MAKEFUN_ENV_URL, the brief at $MAKEFUN_PROMPT_FILE, and model calls go through
//   the grader's gateway at $OPENAI_BASE_URL on the trial's dummy key in $OPENAI_API_KEY (the
//   gateway swaps it for the entrant's pushed OpenRouter key; no real key ever enters here).
//   The server:
//     1. listens on 0.0.0.0 at a free port the OS picks (never 8080: the env owns it on one
//        layout, and the listen step below refuses it);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": ...};
//     3. serves GET /v1/models (Anthropic's shape when the request carries an x-api-key header,
//        OpenAI's otherwise), POST /v1/chat/completions and POST /v1/messages, each as JSON or
//        as server-sent events, as the client asked;
//     4. answers every prompt with one real completion from a free model in the brief, dialled on
//        the gateway's Chat Completions route — an Anthropic-shaped request is translated into
//        that body first and its answer translated back (lib/wire.ts);
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 the moment it says done: a
//        process still running when the trial's wall is spent scores zero, so nothing here ever
//        waits on anything else.
//   `node main.ts --list` prints the catalog document (lib/catalog.ts) and exits 0 before any
//   server starts; the register body carries that same document, byte for byte.
//
// ROUTING (quality is 60% of the score)
//   Each prompt is served by the best-rated free model that meets its requirements: the brief
//   publishes a rating per model, and quality is the served model's rating over the best rating
//   a meeting model carries. Candidates are ranked once per prompt (lib/providers.ts
//   rankCandidates) from the brief's manifest row for the arriving request — the manifest is in
//   the client's arrival order — with the request body's own declarations as the fallback for a
//   brief that holds no row.
//
// FAILOVER (10%) ON A FREE-TIER KEY
//   The key behind this entry is a free-tier one (about 50 free-model requests a day), so calls
//   are spent like they matter: one at a time, never in parallel, never speculatively. When a
//   call is refused (a 4xx or 5xx relayed from the provider), the walk moves to the next ranked
//   candidate that meets the prompt — never back to the model that just refused: on a free
//   tier a refusal is a state, not a moment, and a same-model retry pays the refusal's price
//   twice. Any refusal at all sidelines its model for the trial's later prompts
//   (lib/routing.ts RefusalBook): a free model's rate limit, outage or bad-answer habit
//   outlives this trial's seconds, so later prompts lead with a model that has not refused —
//   unless it is the only candidate left; a refusal that names the key or the model itself
//   (a 401, 403 or 404) walks behind even the softly refused, and one model the key cannot
//   use at all walks last from the start (lib/routing.ts DEPRIORITIZED_MODELS). One prompt
//   costs at most MAX_ATTEMPTS_PER_PROMPT calls; the whole trial stays far under the
//   gateway's 60-call cap, and the common trial costs exactly one call per prompt, which is
//   also what the score's tie-break rewards.
//   A prompt recovered after refusals reports them on its response header
//   `x-makefun-upstream-failures` (lib/routing.ts trailHeader): that header is the only way the
//   failover column can see a refusal the walk recovered from, because the env sees this
//   server's answers and never the gateway's traffic.
//
// VERIFICATION (served 15%, quality 60%) AND THE 30-SECOND BUDGET
//   The house gives each prompt 30 seconds to be answered, and on the live provider a 200 is
//   not always an answer: a model can refuse inside a 200 body, answer with no content (a
//   reasoning model that spent its whole cap thinking), name a tool the client never
//   declared, or ignore the instruction entirely — and every one of those relays as a silent
//   zero, because the house scores the answer's text against the prompt's token. So every
//   answer is read whole and verified before anything is relayed (lib/verify.ts), against the
//   prompt's own token read off the client's body; an answer that would earn nothing is
//   treated as a refusal and the walk moves on. Streams are buffered whole for the same
//   reason: a stream is only relayed once its recorded bytes are known to close properly and
//   to carry the token.
//   Each attempt also carries a fixed time budget (lib/routing.ts ATTEMPT_TIMEOUT_MS): the
//   first attempt may take 8 of the prompt's 30 seconds, later ones less, and all eight
//   together stay under the 30, so one hanging model cannot spend the whole prompt. The
//   budget only ever cuts an attempt short, and a cut attempt replays identically: cut
//   before its answer arrived, the house's record of it is a 499 (client closed the
//   request), which the walk reads as the same attempt_timeout the live cut produced; cut
//   mid-answer, the record is the partial body itself, which verification rejects (or, if
//   the token already arrived whole, accepts) on the same bytes in both passes — so the
//   walk's choices stay a function of the answers alone.
//
// THE MODEL RULE
//   A response's `model` field names the model this server really dialled, on every answer and
//   every streamed chunk: the provider's own answer spells a free model without the gateway's
//   `openrouter/` prefix, so the dialled id is put back everywhere (lib/wire.ts). A prompt whose
//   reported model has no matching successful call in the house's record earns nothing.
//
// DETERMINISM
//   Every choice here is a function of the brief, the requests and the gateway's answers, in
//   arrival order: no wall clock readings, no randomness. The one timer — an attempt's fixed
//   budget — decides nothing by itself: what it cuts is frozen by the gateway into the record
//   (a 499, or the partial body), and the walk reads that record, in a replay, to the same
//   step it took live. A replay serves the recorded answers back and the same walk happens
//   again, leaving the same export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { buildCatalog, catalogText, registrationPayload, type CatalogDocument } from './lib/catalog.ts';
import {
  freeModels,
  isRecord,
  rankCandidates,
  readBrief,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import { MAX_ATTEMPTS_PER_PROMPT, RefusalBook, attemptTimeoutMs, deprioritize, refusalCode, trailHeader, type TrailRow } from './lib/routing.ts';
import { declaredToolNames, tokenOfRequest, verifyJsonAnswer, verifyStreamAnswer } from './lib/verify.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  ensureAnswerUsage,
  ensureFinishReason,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway's own ceiling on one call's output cap; a request over it is refused outright. */
const GATEWAY_MAX_OUTPUT_TOKENS = 8192;
/** The output cap named when a client body carries none: the gateway refuses a call with no cap. */
const DEFAULT_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogDocument = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** The trial's refusal memory: which models have refused so far, and how often. */
const book = new RefusalBook();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on
 * the trial network); a base that does not already end in `/v1` gets one, so both spellings
 * reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  headers: Record<string, string> = {},
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), headers);
  else sendJson(res, status, openAIError(message, type, null), headers);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * A Chat Completions body with an output cap the gateway accepts: the client's own cap when it
 * named one (clamped to the gateway's ceiling), else the default. The gateway refuses a call
 * that names no cap at all, so no body leaves here without one.
 */
function withOutputCap(body: Json): Json {
  const out: Json = { ...body };
  let capped = false;
  for (const key of ['max_tokens', 'max_completion_tokens'] as const) {
    const value = out[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      capped = true;
      if (value > GATEWAY_MAX_OUTPUT_TOKENS) out[key] = GATEWAY_MAX_OUTPUT_TOKENS;
    }
  }
  if (!capped) out.max_tokens = DEFAULT_MAX_TOKENS;
  return out;
}

/**
 * The gateway body for one attempt on `model`: an OpenAI-shaped client body passes through with
 * the dialled model's id in place of `auto`; an Anthropic-shaped one is translated into Chat
 * Completions first. Either way the output cap is made gateway-safe.
 */
function buildUpstream(wire: 'openai' | 'anthropic', body: Json, model: ModelRow): Json {
  if (wire === 'anthropic') return withOutputCap(anthropicToOpenAI(body, model.id));
  return withOutputCap({ ...body, model: model.id });
}

/**
 * POST one attempt to the gateway's Chat Completions route, naming the model this entry
 * dialled, under the attempt's time budget: the fetch — and the read of its body, which the
 * same signal governs — is abandoned when the budget runs out, and the caller treats the
 * attempt as refused (see isTimeoutError). The gateway tears the call down when this server
 * lets go, so an abandoned attempt costs the budget and nothing else.
 */
async function dial(model: string, body: Json, timeoutMs: number): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

/** Whether a dial failed because the attempt's time budget ran out. */
function isTimeoutError(err: unknown): boolean {
  const name = (err as { name?: unknown } | null)?.name;
  return name === 'TimeoutError' || name === 'AbortError';
}

/**
 * Read an attempt's whole answer, keeping whatever arrived if the read is cut short. A cut
 * read is not an error to the walk: the gateway froze the same partial bytes as the attempt's
 * record when the attempt was cut, a replay serves exactly those bytes, and verification
 * judges them — so the bytes that arrived are the answer of record, whole or partial, and
 * the verdict on them is the same in a live pass and in its replay.
 */
async function readAnswerBody(upstream: Response): Promise<string> {
  if (upstream.body === null) return '';
  const reader = upstream.body.getReader();
  const parts: Uint8Array[] = [];
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value !== undefined) parts.push(value);
    }
  } catch {
    // Cut short (the attempt's budget, or the connection): what arrived is the record.
  }
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const all = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    all.set(part, at);
    at += part.length;
  }
  return new TextDecoder().decode(all);
}

/** The response headers for a relayed answer: the refusal trail, when this prompt was recovered. */
function relayHeaders(trail: readonly TrailRow[]): Record<string, string> {
  const headers: Record<string, string> = {};
  const trailValue = trailHeader(trail);
  if (trailValue !== null) headers['x-makefun-upstream-failures'] = trailValue;
  return headers;
}

/**
 * A verified JSON answer, relayed to the client in its own wire: the dialled model's id on the
 * answer, and its usage and finish reason first made whole (lib/wire.ts ensureAnswerUsage and
 * ensureFinishReason): some free models answer with no usage object at all, or name no finish
 * reason, and the wire checks read both.
 */
function relayJson(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  answer: Json,
  upstreamBody: Json,
  model: ModelRow,
  trail: readonly TrailRow[],
): void {
  const record = ensureAnswerUsage(ensureFinishReason(answer), JSON.stringify(upstreamBody.messages ?? []));
  const headers = relayHeaders(trail);
  if (wire === 'anthropic') sendJson(res, 200, openAIToAnthropic(record, model.id), headers);
  else sendJson(res, 200, openAIJsonWithModel(record, model.id), headers);
}

/** A buffered stream's bytes as the readable the wire relays consume. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/**
 * A verified stream, relayed to the client in its own wire from the bytes that were verified:
 * the dialled model's id on every chunk, and the refusal trail on the response header when
 * this prompt was recovered after one or more refusals. The bytes are the attempt's own, read
 * whole during verification, so what is relayed is exactly what was checked.
 */
async function relayStream(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  raw: string,
  upstreamBody: Json,
  model: ModelRow,
  trail: readonly TrailRow[],
): Promise<void> {
  const headers = relayHeaders(trail);
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
  try {
    if (wire === 'anthropic') {
      for await (const event of anthropicEventsFromOpenAI(streamOf(raw), model.id)) res.write(event);
    } else {
      for await (const frame of openAIStreamWithModel(streamOf(raw), model.id, JSON.stringify(upstreamBody.messages ?? []))) res.write(frame);
    }
  } finally {
    res.end();
  }
}

/** A prompt every attempt failed: a 502 in the client's wire, with the trail of what refused it. */
function giveUp(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  last: { status: number; text: string } | null,
  trail: readonly TrailRow[],
): void {
  const headers: Record<string, string> = {};
  const trailValue = trailHeader(trail);
  if (trailValue !== null) headers['x-makefun-upstream-failures'] = trailValue;
  const detail = last === null ? '' : `the gateway last answered HTTP ${last.status}${last.text === '' ? '' : `: ${last.text.slice(0, 500)}`}`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail === '' ? 'no gateway call answered' : detail, headers);
  else sendError(res, wire, 502, 'upstream_error', detail === '' ? 'no gateway call answered' : detail, headers);
}

/**
 * Serve one client request on either wire: rank the candidates that meet the prompt, walk them
 * in order, and relay the first answer that verifies. The walk is the design at the top of this
 * file — sequential, refusal-driven, one attempt per model, capped per prompt, each attempt
 * under its time budget, and an answer relayed only once it is verified against the prompt's
 * token — and every step of it is decided by the responses alone, so a replay walks it
 * identically.
 */
async function serve(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = deprioritize(book.order(rankCandidates(brief, requirementsFor(body))));
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const token = tokenOfRequest(body);
  const declared = declaredToolNames(wire, body);
  const streamed = body.stream === true;
  const trail: TrailRow[] = [];
  let last: { status: number; text: string } | null = null;
  // The house hangs up on a prompt it has waited 30 seconds for; from then on no attempt can
  // earn anything for this prompt, and the calls further attempts would cost are the key's to
  // keep for the prompts still to come. A replay is never hung up on — the recorded answers
  // arrive at once — so this only ever stops a walk the live pass had already lost.
  let abandoned = false;
  res.on('close', () => {
    abandoned = true;
  });
  let attempts = 0;
  let index = 0;
  while (index < candidates.length && attempts < MAX_ATTEMPTS_PER_PROMPT) {
    if (abandoned) return;
    const model = candidates[index];
    index += 1;
    const timeoutMs = attemptTimeoutMs(attempts);
    attempts += 1;
    const upstreamBody = buildUpstream(wire, body, model);
    let upstream: Response;
    try {
      upstream = await dial(model.id, upstreamBody, timeoutMs);
    } catch (err) {
      if (isTimeoutError(err)) {
        // The attempt produced no answer within its budget: trailed and noted like a refusal,
        // because a model too slow to answer inside the budget is, for this trial, down.
        trail.push({ status: 504, code: 'attempt_timeout', model: model.id });
        book.note(model.id, 504);
        last = { status: 504, text: `no answer within ${timeoutMs} ms` };
      } else {
        // The gateway itself did not answer: no response exists to trail, and the walk moves on.
        last = { status: 502, text: (err as Error).message };
      }
      continue;
    }
    if (upstream.status === 499) {
      // The gateway's record of an attempt the live pass cut at its budget: client closed the
      // request. It is read as the attempt_timeout the live cut produced — same trail row,
      // same note — so the replay walks this step exactly as the live pass did.
      await upstream.text().catch(() => '');
      trail.push({ status: 504, code: 'attempt_timeout', model: model.id });
      book.note(model.id, 504);
      last = { status: 504, text: `no answer within ${timeoutMs} ms` };
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      trail.push({ status: upstream.status, code: refusalCode(upstream.status, text), model: model.id });
      book.note(model.id, upstream.status);
      last = { status: upstream.status, text };
      continue;
    }
    // A 2xx: read the whole answer before anything is relayed — the attempt's budget still
    // governs the read, and a read it cuts leaves the partial bytes as the answer of record —
    // and verify it; an answer that would earn nothing is a refusal.
    const raw = await readAnswerBody(upstream);
    if (streamed) {
      const verdict = verifyStreamAnswer(raw, token, declared);
      if (!verdict.ok) {
        trail.push({ status: upstream.status, code: verdict.code, model: model.id });
        book.note(model.id, upstream.status);
        last = { status: upstream.status, text: raw };
        continue;
      }
      await relayStream(res, wire, raw, upstreamBody, model, trail);
      return;
    }
    let answer: unknown = null;
    try {
      answer = JSON.parse(raw);
    } catch {
      // Not JSON at all: the verdict below names it bad_body.
    }
    if (!isRecord(answer)) {
      trail.push({ status: upstream.status, code: 'bad_body', model: model.id });
      book.note(model.id, upstream.status);
      last = { status: upstream.status, text: raw };
      continue;
    }
    const verdict = verifyJsonAnswer(answer, token, declared);
    if (!verdict.ok) {
      trail.push({ status: upstream.status, code: verdict.code, model: model.id });
      book.note(model.id, upstream.status);
      last = { status: upstream.status, text: raw };
      continue;
    }
    relayJson(res, wire, answer, upstreamBody, model, trail);
    return;
  }
  giveUp(res, wire, last, trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(registrationPayload(port, catalog)),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so
    // wait for the write to land.
    const text = catalogText(catalog);
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
