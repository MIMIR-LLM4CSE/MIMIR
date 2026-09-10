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
    const req = mod.get(
      url,
      { agent, rejectUnauthorized: verifySsl, timeout: timeoutMs },
      (res) => {
        const status = res.statusCode ?? 0;
        if (status < 200 || status >= 300) {
          res.resume();
          reject(new Error(`HTTP ${status} from ${url.href}`));
          return;
        }
        let raw = "";
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => { raw += chunk; });
        res.on("end", () => {
          try {
            resolve(parseModels(backend, JSON.parse(raw)));
          } catch {
            reject(new Error(`unreadable response from ${url.href}`));
          }
        });
      },
    );
    req.on("timeout", () => req.destroy(new Error(`timed out after ${timeoutMs} ms`)));
    req.on("error", (err) => reject(err));
  });
}
