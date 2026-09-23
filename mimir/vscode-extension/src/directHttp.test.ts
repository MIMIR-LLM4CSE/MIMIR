/**
 * The extension host's own HTTP path.
 *
 * It parses HTTP/1.1 itself, so the framings a real endpoint uses — chunked, a
 * content-length, a body that ends only at EOF — are pinned here rather than
 * discovered against a cluster. The proxy bypass that motivates the module cannot
 * be tested without VS Code's patched `http`; what is testable, and what breaks
 * silently, is the parsing and the promise always settling.
 */
import { describe, it, expect } from "vitest";
import * as http from "http";
import * as net from "net";
import type { AddressInfo } from "net";
import { directGet } from "./directHttp";

/** Start a server running *handler*, and return its base URL. */
async function serving(handler: http.RequestListener) {
  const server = http.createServer(handler);
  await new Promise<void>(done => server.listen(0, "127.0.0.1", done));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/thing`,
    close: () => new Promise<void>(done => {
      server.closeAllConnections?.();
      server.close(() => done());
    }),
  };
}

/**
 * A server that writes *raw* bytes and closes — framings http.Server won't emit.
 *
 * `raw` is one byte per code unit (latin1): a chunk header counts bytes, so writing
 * it as UTF-8 would re-encode every byte above 0x7F and desync the framing.
 */
async function rawServing(raw: string) {
  const bytes = Buffer.from(raw, "latin1");
  const server = net.createServer(sock => {
    sock.once("data", () => { sock.end(bytes); });
  });
  await new Promise<void>(done => server.listen(0, "127.0.0.1", done));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/thing`,
    close: () => new Promise<void>(done => { server.close(() => done()); }),
  };
}

describe("directGet", () => {
  it("reads a chunked body", async () => {
    // Node streams a response with no content-length as chunked, which is what a
    // vLLM model list actually arrives as.
    const { url, close } = await serving((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.write('{"data":[{"id":"a"}');
      res.end(',{"id":"b"}]}');
    });
    try {
      const res = await directGet(url, { timeoutMs: 2000 });
      expect(res.status).toBe(200);
      expect(JSON.parse(res.body).data).toHaveLength(2);
    } finally {
      await close();
    }
  });

  it("reads a body delimited by content-length", async () => {
    const body = '{"data":[]}';
    const { url, close } = await rawServing(
      `HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${body.length}\r\n\r\n${body}`,
    );
    try {
      await expect(directGet(url, { timeoutMs: 2000 })).resolves.toEqual({ status: 200, body });
    } finally {
      await close();
    }
  });

  it("reads a body that ends only when the connection closes", async () => {
    const { url, close } = await rawServing('HTTP/1.1 200 OK\r\n\r\n{"data":[]}');
    try {
      await expect(directGet(url, { timeoutMs: 2000 })).resolves.toEqual(
        { status: 200, body: '{"data":[]}' });
    } finally {
      await close();
    }
  });

  it("decodes UTF-8 split across two chunks", async () => {
    // The body is bytes, not text: decoding per chunk would cut a multi-byte
    // character in half and yield U+FFFD in a model name.
    const name = "modèle-é";
    const bytes = Buffer.from(JSON.stringify({ id: name }), "utf8");
    const cut = 8;  // lands inside the "è"
    const { url, close } = await rawServing(
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
      + `${cut.toString(16)}\r\n${bytes.subarray(0, cut).toString("latin1")}\r\n`
      + `${(bytes.length - cut).toString(16)}\r\n${bytes.subarray(cut).toString("latin1")}\r\n`
      + "0\r\n\r\n",
    );
    try {
      const res = await directGet(url, { timeoutMs: 2000 });
      expect(JSON.parse(res.body).id).toBe(name);
    } finally {
      await close();
    }
  });

  it("reports the status of an error response rather than throwing", async () => {
    // 404 and 401 mean different fixes; the caller turns the code into words.
    const { url, close } = await serving((_req, res) => { res.statusCode = 404; res.end("nope"); });
    try {
      await expect(directGet(url, { timeoutMs: 2000 })).resolves.toMatchObject({ status: 404 });
    } finally {
      await close();
    }
  });

  it("sends the headers it is given", async () => {
    let seen: string | undefined;
    const { url, close } = await serving((req, res) => {
      seen = req.headers.authorization;
      res.end("{}");
    });
    try {
      await directGet(url, { timeoutMs: 2000, headers: { Authorization: "Bearer k" } });
      expect(seen).toBe("Bearer k");
    } finally {
      await close();
    }
  });

  it("rejects, naming the phase, when the endpoint never answers", async () => {
    // The caller shows one line under the address field. A promise that never
    // settles leaves it saying "asking…" forever, with no cause to show.
    const { url, close } = await serving(() => { /* deliberately silent */ });
    try {
      await expect(directGet(url, { timeoutMs: 150 }))
        .rejects.toThrow(/timed out after 150 ms — connected, waiting for a response/);
    } finally {
      await close();
    }
  });

  it("rejects a refusal rather than waiting out the deadline", async () => {
    const { url, close } = await serving(() => {});
    await close();  // nothing is listening on that port any more
    await expect(directGet(url, { timeoutMs: 5000 })).rejects.toThrow(/ECONNREFUSED/);
  });

  it("rejects an invalid URL without opening anything", async () => {
    await expect(directGet("not a url")).rejects.toThrow(/invalid URL/);
  });
});
