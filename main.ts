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
//   `node main.ts --list` prints the catalog document and exits 0 before any server starts; the
//   register body carries that same document, byte for byte.
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
//   are spent like they matter: one at a time, never in parallel, never speculatively, and a
//   retry only ever follows an actual refusal. When a call is refused (a 4xx or 5xx relayed
//   from the provider), the walk moves to the next ranked candidate that meets the prompt; a
//   5xx earns its model exactly one immediate retry first, since a server error is often a
//   moment's fault and the retry keeps the prompt on its best model. A model that answered 429
//   is remembered for the rest of the trial (lib/routing.ts RefusalBook) and walked past for
//   later prompts — a free model's rate limit outlives this trial's seconds — unless it is the
//   only candidate left. One prompt costs at most MAX_ATTEMPTS_PER_PROMPT calls; the
//   whole trial stays far under the gateway's 60-call cap, and the common trial costs exactly
//   one call per prompt, which is also what the score's tie-break rewards.
//   A prompt recovered after refusals reports them on its response header
//   `x-makefun-upstream-failures` (lib/routing.ts trailHeader): that header is the only way the
//   failover column can see a refusal the walk recovered from, because the env sees this
//   server's answers and never the gateway's traffic.
//
// THE MODEL RULE
//   A response's `model` field names the model this server really dialled, on every answer and
//   every streamed chunk: the provider's own answer spells a free model without the gateway's
//   `openrouter/` prefix, so the dialled id is put back everywhere (lib/wire.ts). A prompt whose
//   reported model has no matching successful call in the house's record earns nothing.
//
// DETERMINISM
//   Every choice here is a function of the brief, the requests and the gateway's answers, in
//   arrival order: no wall clock, no randomness, no timers in any decision. A replay serves the
//   recorded answers back and the same walk happens again, leaving the same export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  rankCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import { MAX_ATTEMPTS_PER_PROMPT, RefusalBook, refusalCode, trailHeader, type TrailRow } from './lib/routing.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
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
const catalog: CatalogReport = buildCatalog(brief);
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

/** POST one attempt to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/**
 * A successful gateway answer, relayed to the client in its own wire: the dialled model's id on
 * every answer and every chunk, the usage and the stop reason whole, and the refusal trail on
 * the response header when this prompt was recovered after one or more refusals.
 */
async function relay(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  streamed: boolean,
  upstream: Response,
  model: ModelRow,
  trail: readonly TrailRow[],
): Promise<void> {
  const headers: Record<string, string> = {};
  const trailValue = trailHeader(trail);
  if (trailValue !== null) headers['x-makefun-upstream-failures'] = trailValue;
  if (streamed) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      if (wire === 'anthropic') {
        for await (const event of anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model.id)) res.write(event);
      } else {
        for await (const frame of openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model.id)) res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  const record = isRecord(answer) ? answer : {};
  if (wire === 'anthropic') sendJson(res, 200, openAIToAnthropic(record, model.id), headers);
  else sendJson(res, 200, openAIJsonWithModel(record, model.id), headers);
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
 * in order, and relay the first answer the gateway gives. The walk is the failover design at the
 * top of this file — sequential, refusal-driven, one 5xx retry per model, capped per prompt —
 * and every step of it is decided by the responses alone, so a replay walks it identically.
 */
async function serve(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = book.order(rankCandidates(brief, requirementsFor(body)));
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const trail: TrailRow[] = [];
  const retried = new Set<string>();
  let last: { status: number; text: string } | null = null;
  let attempts = 0;
  let index = 0;
  while (index < candidates.length && attempts < MAX_ATTEMPTS_PER_PROMPT) {
    const model = candidates[index];
    attempts += 1;
    let upstream: Response;
    try {
      upstream = await dial(model.id, buildUpstream(wire, body, model));
    } catch (err) {
      // The gateway itself did not answer: no response exists to trail, and the walk moves on.
      last = { status: 502, text: (err as Error).message };
      index += 1;
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      await relay(res, wire, body.stream === true, upstream, model, trail);
      return;
    }
    if (upstream.ok) {
      // A 2xx with no body is not an answer a client can be served; nothing was refused, so the
      // trail stays empty, and the prompt is given up rather than walked further on a guess.
      last = { status: 502, text: 'the gateway answered with no body' };
      break;
    }
    const text = await upstream.text().catch(() => '');
    trail.push({ status: upstream.status, code: refusalCode(upstream.status, text), model: model.id });
    book.note(model.id, upstream.status);
    last = { status: upstream.status, text };
    if (upstream.status >= 500 && !retried.has(model.id) && attempts < MAX_ATTEMPTS_PER_PROMPT) {
      // One immediate retry for a server error: it keeps the prompt on its best model when the
      // fault was a moment's, and it costs exactly the one call a failover would have cost.
      retried.add(model.id);
      continue;
    }
    index += 1;
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
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
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
    const text = `${JSON.stringify(catalog)}\n`;
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
  await register(port, catalog);
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
