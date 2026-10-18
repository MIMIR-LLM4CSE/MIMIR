/**
 * HTTP from the extension host, straight to the address asked for.
 *
 * VS Code patches `http`/`https` to route through the user's proxy, even with an
 * explicit agent, and a corporate proxy holds a cluster address until the deadline.
 * `net`/`tls` are not patched, so this speaks HTTP/1.1 over its own socket, as the
 * Python side does with `trust_env=False`. No redirects, keep-alive or proxy support.
 */
import * as net from "net";
import * as tls from "tls";

/** One HTTP response, reduced to what the callers here read. */
export interface DirectResponse {
  status: number;
  body: string;
}

export interface DirectGetOptions {
  /** Verify the server's certificate (HTTPS only). Mirrors `mimir.vllmVerifySsl`. */
  verifySsl?: boolean;
  /** Whole-request deadline, covering the phases no socket timeout reaches. */
  timeoutMs?: number;
  /** Extra request headers, e.g. Authorization. */
  headers?: Record<string, string>;
}

/** Split a header block into lowercased name -> value. */
function parseHeaders(head: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const line of head.split("\r\n").slice(1)) {
    const i = line.indexOf(":");
    if (i > 0) out.set(line.slice(0, i).trim().toLowerCase(), line.slice(i + 1).trim());
  }
  return out;
}

/**
 * Decode a `Transfer-Encoding: chunked` body.
 *
 * Returns undefined while the terminating zero-length chunk has not arrived, so the
 * caller keeps reading instead of parsing half a JSON document.
 */
function decodeChunked(buf: Buffer): string | undefined {
  let at = 0;
  const parts: Buffer[] = [];
  for (;;) {
    const nl = buf.indexOf("\r\n", at, "latin1");
    if (nl < 0) return undefined;
    const size = parseInt(buf.toString("latin1", at, nl).split(";")[0], 16);
    if (Number.isNaN(size)) return undefined;
    if (size === 0) return Buffer.concat(parts).toString("utf8");
    const from = nl + 2;
    if (buf.length < from + size + 2) return undefined;
    parts.push(buf.subarray(from, from + size));
    at = from + size + 2;
  }
}

/**
 * GET *target*, reporting how far the attempt got if it does not finish.
 *
 * The deadline is our own timer: no socket timeout covers the phase before a
 * connection exists, and the rejection names the last phase reached. `Connection:
 * close` ends the body at content-length, the last chunk, or EOF.
 */
export function directGet(
  target: string | URL,
  { verifySsl = true, timeoutMs = 30000, headers = {} }: DirectGetOptions = {},
): Promise<DirectResponse> {
  return new Promise((resolve, reject) => {
    let url: URL;
    try {
      url = typeof target === "string" ? new URL(target) : target;
    } catch {
      reject(new Error(`invalid URL: ${String(target)}`));
      return;
    }

    // How far the request got, for the message if it gets no further.
    let phase = "no socket assigned (name resolution or connection)";
    let settled = false;
    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(deadline);
      socket.destroy();
      fn();
    };
    const deadline = setTimeout(
      () => finish(() => reject(new Error(`timed out after ${timeoutMs} ms — ${phase}`))),
      timeoutMs,
    );

    const secure = url.protocol === "https:";
    const port = url.port ? Number(url.port) : secure ? 443 : 80;
    const socket = secure
      ? tls.connect({
        host: url.hostname, port, servername: url.hostname, rejectUnauthorized: verifySsl,
      })
      : net.connect({ host: url.hostname, port });

    // Each step the request clears, so a hang can say where it stopped rather than
    // only that it stopped. A socket that never connects is a blocked route; a
    // connection made but no response is the endpoint itself going quiet.
    phase = "socket assigned, connecting";
    socket.on("lookup", () => { phase = "name resolved, connecting"; });
    socket.on("connect", () => { phase = "connected, waiting for a response"; });
    socket.on("secureConnect", () => { phase = "TLS established, waiting for a response"; });

    socket.on(secure ? "secureConnect" : "connect", () => {
      const lines = [
        `GET ${url.pathname}${url.search} HTTP/1.1`,
        `Host: ${url.host}`,
        "Accept: application/json",
        "User-Agent: mimir-vscode",
        "Connection: close",
        ...Object.entries(headers).map(([k, v]) => `${k}: ${v}`),
      ];
      socket.write(lines.join("\r\n") + "\r\n\r\n");
    });

    let buf = Buffer.alloc(0);
    let status = 0;
    let headEnd = -1;
    let head = new Map<string, string>();

    /** The body if it is complete, else undefined. *atEof* closes an open-ended one. */
    const body = (atEof: boolean): string | undefined => {
      const rest = buf.subarray(headEnd + 4);
      if ((head.get("transfer-encoding") ?? "").toLowerCase().includes("chunked")) {
        return decodeChunked(rest);
      }
      const len = head.get("content-length");
      if (len !== undefined) {
        const want = Number(len);
        return rest.length >= want ? rest.toString("utf8", 0, want) : undefined;
      }
      return atEof ? rest.toString("utf8") : undefined;
    };

    const take = (atEof: boolean) => {
      if (headEnd < 0) {
        headEnd = buf.indexOf("\r\n\r\n", 0, "latin1");
        if (headEnd < 0) return;
        const text = buf.toString("latin1", 0, headEnd);
        status = Number(/^HTTP\/1\.[01] (\d{3})/.exec(text)?.[1] ?? 0);
        head = parseHeaders(text);
        phase = `response started (HTTP ${status})`;
      }
      const text = body(atEof);
      if (text === undefined) return;
      finish(() => resolve({ status, body: text }));
    };

    socket.on("data", (chunk: Buffer) => { buf = Buffer.concat([buf, chunk]); take(false); });
    socket.on("end", () => take(true));
    socket.on("close", () => finish(() => reject(new Error(
      headEnd < 0 ? "connection closed before any response" : "connection closed mid-response",
    ))));
    socket.on("error", (err) => finish(() => reject(err)));
  });
}
