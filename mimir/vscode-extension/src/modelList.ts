/**
 * Model discovery — asks the backend endpoint what it serves.
 *
 * The connect form used to read its model list from `mimir.vllmAvailableModels` /
 * `mimir.availableModels` in a hand-written `.vscode/settings.json`. The endpoint
 * already knows the answer, so we ask it instead and the user only ever types a URL.
 *
 * The fetch lives in the extension host, not the webview: the webview's CSP only
 * allows `connect-src ws://localhost:*`, so React cannot reach an HTTP endpoint.
 */
import { directGet } from "./directHttp";

/** Endpoints that can enumerate their own models. Anthropic keeps a static list. */
export type DiscoverableBackend = "vllm" | "ollama" | "ray";

/** Strip trailing slashes so URL joins never double up. */
function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/**
 * URL that lists the models served at *baseUrl*.
 *
 * vLLM and Ray Serve both speak the OpenAI API, so the list is under `/v1/models` —
 * tolerate a base URL the user already suffixed with `/v1` (or, for Ray, ended with
 * the app's route prefix) rather than producing `/v1/v1`.
 */
export function modelsUrl(backend: DiscoverableBackend, baseUrl: string): string {
  const base = trimSlash(baseUrl);
  if (backend === "ollama") {
    return `${base}/api/tags`;
  }
  return base.endsWith("/v1") ? `${base}/models` : `${base}/v1/models`;
}

/**
 * Extract model names from a parsed models response.
 *
 * Anything that isn't a non-empty string is dropped rather than surfaced as a
 * blank entry in the dropdown; a shape we don't recognise yields `[]`, which the
 * form reports as "no models found" instead of throwing.
 */
export function parseModels(backend: DiscoverableBackend, body: unknown): string[] {
  const rec = body as Record<string, unknown> | null;
  const raw = backend === "ollama" ? rec?.models : rec?.data;
  if (!Array.isArray(raw)) {
    return [];
  }
  const key = backend === "ollama" ? "name" : "id";
  const names = raw
    .map((entry) => (entry as Record<string, unknown> | null)?.[key])
    .filter((n): n is string => typeof n === "string" && n.trim() !== "");
  return [...new Set(names)];
}

/**
 * GET the model list from *baseUrl*.
 *
 * The request goes through `directGet`, not `http.get`: see `directHttp.ts` for why
 * the extension host must reach a cluster address without the proxy VS Code would
 * otherwise impose. `verifySsl` mirrors the `mimir.vllmVerifySsl` setting — one
 * switch for both OpenAI-compatible endpoints — for HTTPS routes behind a private CA.
 *
 * The default deadline is generous (30 s) because the endpoints this asks about are
 * often slow to *answer*, not absent: a cluster route behind a VPN, a gateway that
 * queues the first request, a vLLM server still loading weights. A short deadline
 * turns "it is coming" into "it is broken" — the user reads a failure under the
 * address field and retypes an address that was right all along.
 */
export async function fetchModels(
  backend: DiscoverableBackend,
  baseUrl: string,
  verifySsl = true,
  timeoutMs = 30000,
): Promise<string[]> {
  let url: URL;
  try {
    url = new URL(modelsUrl(backend, baseUrl));
  } catch {
    throw new Error(`invalid URL: ${baseUrl}`);
  }
  const res = await directGet(url, { verifySsl, timeoutMs });
  if (res.status < 200 || res.status >= 300) {
    throw new Error(`HTTP ${res.status} from ${url.href}`);
  }
  try {
    return parseModels(backend, JSON.parse(res.body));
  } catch {
    throw new Error(`unreadable response from ${url.href}`);
  }
}

/** Node's codes for a certificate the client would not trust. */
const CERT_CODES = new Set([
  "SELF_SIGNED_CERT_IN_CHAIN",
  "DEPTH_ZERO_SELF_SIGNED_CERT",
  "UNABLE_TO_VERIFY_LEAF_SIGNATURE",
  "UNABLE_TO_GET_ISSUER_CERT",
  "UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
  "CERT_UNTRUSTED",
  "CERT_HAS_EXPIRED",
  "ERR_TLS_CERT_ALTNAME_INVALID",
]);

/**
 * Turn a `fetchModels` failure into a sentence the user can act on.
 *
 * The raw error ("connect ECONNREFUSED 10.0.0.1:8000") says what the socket saw,
 * not what it means or what to try. Each case here names what happened in plain
 * words, then the likely fix. The raw text still goes to the output channel, for
 * whoever needs the code.
 */
export function explainFetchError(err: unknown, baseUrl: string): string {
  const e = err as { code?: string; message?: string } | null;
  const code = e?.code ?? "";
  const msg = e?.message ?? String(err);
  let host = baseUrl;
  try {
    host = new URL(baseUrl).host;
  } catch {
    return `“${baseUrl}” is not a valid address. It should look like http://host:port.`;
  }

  if (CERT_CODES.has(code) || /certificate|self.signed/i.test(msg)) {
    return `The server at ${host} answered, but its security certificate is not trusted `
      + `(common for internal servers). If you trust this server, untick `
      + `“Vllm Verify Ssl” in the MIMIR settings, then retry.`;
  }
  if (code === "EPROTO" || /wrong version number/i.test(msg)) {
    return `The server at ${host} does not speak HTTPS. Try the address with http:// instead.`;
  }
  switch (code) {
    case "ECONNREFUSED":
      return `No server is running at ${host}. Check that the model server is started `
        + `and that the port is right.`;
    case "ENOTFOUND":
    case "EAI_AGAIN":
      return `The name “${host}” could not be found. Check the spelling, or your VPN.`;
    case "EHOSTUNREACH":
    case "ENETUNREACH":
      return `${host} cannot be reached from this machine. Check your VPN or network.`;
    case "ECONNRESET":
      return `The server at ${host} closed the connection. If the address starts with `
        + `http://, try https:// (or the reverse).`;
  }
  if (/timed out/.test(msg)) {
    return `${host} did not answer in time. The server may still be starting, `
      + `or the network (VPN, firewall) blocks the way.`;
  }
  const status = /^HTTP (\d+)/.exec(msg)?.[1];
  if (status === "401" || status === "403") {
    return `The server at ${host} refused access (HTTP ${status}). It needs an API key.`;
  }
  if (status === "404") {
    return `A server answers at ${host}, but it has no model list at this address. `
      + `Check the path (for example, with or without /v1).`;
  }
  if (status) {
    return `The server at ${host} answered with an error (HTTP ${status}). `
      + `It may still be starting.`;
  }
  if (/unreadable response/.test(msg)) {
    return `Something answers at ${host}, but it is not a model server. `
      + `Check the port and the backend type.`;
  }
  return `Could not get the model list from ${host} (${msg}).`;
}
