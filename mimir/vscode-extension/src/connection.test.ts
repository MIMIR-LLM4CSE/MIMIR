/** The socket presents the server's token, on every connect, against a real WebSocket server. */
import type { AddressInfo } from "net";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WebSocketServer } from "ws";

const toasts = vi.hoisted(() => [] as string[]);

vi.mock("vscode", () => ({}));
vi.mock("./notifications", () => ({
  showToast: async (_kind: string, message: string) => { toasts.push(message); },
}));

import { ServerConnection } from "./connection";

const servers: WebSocketServer[] = [];

afterEach(() => {
  toasts.length = 0;
  for (const server of servers.splice(0)) server.close();
});

/** A server taking the token "right" and answering "hi"; the headers it saw, oldest first. */
async function aServer(): Promise<{ url: string; seen: (string | undefined)[] }> {
  const seen: (string | undefined)[] = [];
  const server = new WebSocketServer({
    port: 0,
    host: "127.0.0.1",
    verifyClient: ({ req }, done) => {
      seen.push(req.headers.authorization);
      done(req.headers.authorization === "Bearer right", 401);
    },
  });
  servers.push(server);
  server.on("connection", (ws) => ws.send("hi"));
  await new Promise((resolve) => server.once("listening", resolve));
  return { url: `ws://127.0.0.1:${(server.address() as AddressInfo).port}`, seen };
}

function aConnection(starting = false) {
  const posted: Record<string, unknown>[] = [];
  const texts: string[] = [];
  const connection = new ServerConnection({
    post: (message) => posted.push(message),
    onServerText: (text) => texts.push(text),
    serverStarting: () => starting,
  });
  return { connection, posted, texts };
}

describe("ServerConnection", () => {
  it("presents the token and is let in", async () => {
    const { url, seen } = await aServer();
    const { connection, texts } = aConnection();
    connection.connect(url, "right");
    await vi.waitFor(() => expect(texts).toEqual(["hi"]));
    expect(seen).toEqual(["Bearer right"]);
    connection.retire();
  });

  it("presents it again on a reconnect", async () => {
    const { url, seen } = await aServer();
    const { connection, texts } = aConnection();
    connection.connect(url, "right");
    await vi.waitFor(() => expect(texts).toHaveLength(1));
    connection.reopen();
    await vi.waitFor(() => expect(texts).toHaveLength(2));
    expect(seen).toEqual(["Bearer right", "Bearer right"]);
    connection.retire();
  });

  it("sends no header without a token", async () => {
    const { url, seen } = await aServer();
    const { connection, posted } = aConnection();
    connection.connect(url);
    await vi.waitFor(() => expect(posted).toContainEqual({ type: "ws_closed" }));
    expect(seen).toEqual([undefined]);
  });

  it("stops at a refused token and says which setting to look at", async () => {
    const { url, seen } = await aServer();
    const { connection, posted } = aConnection(true);
    connection.connect(url, "wrong");
    await vi.waitFor(() => expect(posted).toContainEqual({ type: "ws_closed" }));
    expect(seen).toHaveLength(1);
    expect(toasts.join()).toContain("mimir.wsToken");
  });
});
