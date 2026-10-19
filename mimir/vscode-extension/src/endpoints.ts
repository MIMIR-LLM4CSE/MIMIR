/**
 * The endpoint the user asked this extension to remember, so a later window can
 * reconnect to it unattended. No secret is ever part of it. Free of `vscode`, so
 * unit-testable.
 */

/** Global-state key holding the endpoint the user asked us to remember. */
export const REMEMBERED_KEY = "mimir.rememberedEndpoint";

/**
 * Backends reached at an address the user types: they can be remembered, reconnected
 * to unattended, and asked for their model list. Anthropic is absent: it needs a key
 * that is never persisted.
 */
export const ADDRESSED_BACKENDS: readonly string[] = ["vllm", "ray", "ollama"];

/** Endpoint we can reconnect to unattended. */
export interface RememberedEndpoint {
  backend: string;
  baseUrl: string;
  model: string;
}

/** *saved* when it names an addressed backend at some address, else undefined. */
export function validRemembered(saved: RememberedEndpoint | undefined): RememberedEndpoint | undefined {
  if (!saved || !saved.baseUrl) return undefined;
  if (!ADDRESSED_BACKENDS.includes(saved.backend)) return undefined;
  return saved;
}
