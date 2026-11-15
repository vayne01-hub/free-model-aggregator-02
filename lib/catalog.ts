// The catalog document: the entry's own statement of what it can serve.
//
// Two things leave this process carrying the catalog, and both must spell it identically:
//
//   - `node main.ts --list` prints it, as catalogText renders it below, and
//   - the body the entry POSTs to the env's /register route at startup carries it
//     (registrationPayload below).
//
// The document is derived from the brief alone (readBrief in lib/providers.ts), and this module
// owns its shape end to end: which models appear, in which order, and the provider block they
// sit under. Nothing here reads the clock, the network or the environment, so a live pass and
// its replay produce the same bytes.

import type { Brief } from './providers.ts';

/** The one provider every model in the brief belongs to. */
const PROVIDER_ID = 'openrouter';
/**
 * The provider's public endpoint. Never the gateway's address: that origin is trial-local and
 * differs between a live pass and its replay, while the catalog is part of the state a replay
 * must reproduce byte for byte.
 */
const PROVIDER_BASE_URL = 'https://openrouter.ai/api/v1';
/** The name of the entrant secret that holds the provider key; the key itself never enters the sandbox. */
const PROVIDER_KEY_ENV = 'OPENROUTER_API_KEY';
/** The protocol version the env's register route expects in the body. */
const REGISTER_PROTOCOL_VERSION = 1;

/** One model row of the catalog: its routed id, that it is free, and its context length as the brief states it. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** The catalog's provider block: who serves the models, at which public endpoint, on which client wire. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** The whole catalog document — the object `--list` prints and /register carries. */
export interface CatalogDocument {
  providers: CatalogProvider[];
}

/** The body POSTed to /register once the server is listening. */
export interface RegistrationPayload {
  protocol_version: number;
  port: number;
  catalog: CatalogDocument;
}

/** The provider block the catalog's models sit under. */
function providerBlock(models: CatalogModel[]): CatalogProvider {
  return { id: PROVIDER_ID, base_url: PROVIDER_BASE_URL, key_env: PROVIDER_KEY_ENV, wire: 'openai', models };
}

/**
 * Build the catalog from a brief: every free model it lists, once each, in the brief's order,
 * under the one provider.
 *
 * The set matters exactly: the catalog column scores the registered set against the brief's
 * dialable free ids, so a free model dropped here — or a paid one let in — loses it. A repeated
 * id collapses to its first row. A brief holding no free models yields an empty provider list
 * rather than a provider block with no models in it.
 */
export function buildCatalog(brief: Brief): CatalogDocument {
  const modelsById = new Map<string, CatalogModel>();
  for (const row of brief.models) {
    if (!row.free || modelsById.has(row.id)) continue;
    modelsById.set(row.id, { id: row.id, free: true, context_length: row.context_length });
  }
  if (modelsById.size === 0) return { providers: [] };
  return { providers: [providerBlock([...modelsById.values()])] };
}

/** The catalog as `--list` prints it: the document as one JSON text, newline-terminated. */
export function catalogText(catalog: CatalogDocument): string {
  return `${JSON.stringify(catalog)}\n`;
}

/** The /register body for a listening port: the protocol version, the port, and the catalog itself. */
export function registrationPayload(port: number, catalog: CatalogDocument): RegistrationPayload {
  return { protocol_version: REGISTER_PROTOCOL_VERSION, port, catalog };
}
