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
import * as http from "http";
import * as https from "https";

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
 * Uses Node's http/https with an agent of our own. The env vars are ignored either
 * way, but `http.proxySupport` (VS Code's default) patches these modules in the
 * extension host to inject a proxy agent — which black-holes the on-prem endpoint
 * this asks about. Passing an explicit agent leaves nothing for that patch to fill
 * in, so the request goes straight to the address the user typed. `verifySsl`
 * mirrors the `mimir.vllmVerifySsl` setting — one switch for both OpenAI-compatible
 * endpoints — for internal HTTPS routes served behind a private CA.
 *
 * The deadline is enforced by a timer of our own, not by the `timeout` request
 * option. That option only arms `socket.setTimeout`, and only once a socket has been
 * assigned: a name that never resolves, a connection that never completes, or a
 * proxy CONNECT that never answers leaves the request with no socket, no timer and
 * no error — a promise that never settles. The caller then waits forever on a
 * question that will never be answered, which is the one outcome a discovery call
 * must not produce. The rejection names the last phase reached, because "it hung"
 * and "it hung before DNS even answered" call for different fixes.
 */
export function fetchModels(
  backend: DiscoverableBackend,
  baseUrl: string,
  verifySsl = true,
  timeoutMs = 5000,
): Promise<string[]> {
  return new Promise((resolve, reject) => {
    let url: URL;
    try {
      url = new URL(modelsUrl(backend, baseUrl));
    } catch {
      reject(new Error(`invalid URL: ${baseUrl}`));
      return;
    }
    const secure = url.protocol === "https:";
    const mod = secure ? https : http;
    // Explicit, so VS Code's proxy patching has no default agent to substitute.
    const agent = secure
      ? new https.Agent({ rejectUnauthorized: verifySsl })
      : new http.Agent();

    // How far the request got, for the message if it gets no further.
    let phase = "no socket assigned (name resolution, connection or proxy CONNECT)";
    let settled = false;
    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(deadline);
      fn();
    };
    const deadline = setTimeout(() => {
      finish(() => {
        req.destroy();
        reject(new Error(`timed out after ${timeoutMs} ms — ${phase}`));
      });
    }, timeoutMs);

    const req = mod.get(
      url,
      { agent, rejectUnauthorized: verifySsl, timeout: timeoutMs },
      (res) => {
        phase = `response started (HTTP ${res.statusCode ?? 0})`;
        const status = res.statusCode ?? 0;
        if (status < 200 || status >= 300) {
          res.resume();
          finish(() => reject(new Error(`HTTP ${status} from ${url.href}`)));
          return;
        }
        let raw = "";
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => { raw += chunk; });
        res.on("end", () => {
          finish(() => {
            try {
              resolve(parseModels(backend, JSON.parse(raw)));
            } catch {
              reject(new Error(`unreadable response from ${url.href}`));
            }
          });
        });
      },
    );

    // Each step the request clears, so a hang can say where it stopped rather than
    // only that it stopped. A socket assigned but never connected is a blocked route
    // or a proxy holding the CONNECT; a connection made but no response is the
    // endpoint itself going quiet.
    req.on("socket", (socket) => {
      phase = "socket assigned, connecting";
      socket.on("lookup", () => { phase = "name resolved, connecting"; });
      socket.on("connect", () => { phase = "connected, waiting for a response"; });
      socket.on("secureConnect", () => { phase = "TLS established, waiting for a response"; });
    });
    req.on("timeout", () => req.destroy(new Error(`timed out after ${timeoutMs} ms — ${phase}`)));
    req.on("error", (err) => finish(() => reject(err)));
  });
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
