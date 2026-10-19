import * as cp from "child_process";
import { afterEach, describe, expect, it, vi } from "vitest";

const registry: { live: { pid: number; url: string } | undefined } = { live: undefined };

vi.mock("vscode", () => ({
  Uri: { joinPath: () => ({}) },
  workspace: { getConfiguration: () => ({ get: (_key: string, fallback?: unknown) => fallback }) },
  window: {
    activeTextEditor: undefined,
    state: { focused: true },
    createOutputChannel: () => ({ show() {}, append() {}, appendLine() {} }),
  },
}));

vi.mock("./serverRegistry", () => ({
  findLiveServer: async () => registry.live,
  logPath: () => "/dev/null",
  waitForServer: async () => undefined,
}));

import { MimirAgentViewProvider } from "./chatView";
import { LocalServer } from "./localServer";

const children: cp.ChildProcess[] = [];

afterEach(() => {
  registry.live = undefined;
  for (const child of children.splice(0)) child.kill("SIGKILL");
});

function aServer(): cp.ChildProcess {
  const child = cp.spawn("sleep", ["30"], { stdio: "ignore" });
  children.push(child);
  return child;
}

function ended(child: cp.ChildProcess): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), 1000);
    child.once("exit", () => { clearTimeout(timer); resolve(true); });
  });
}

/** A chat view resolved into a fake webview; returns what the webview would send. */
async function aView(server: LocalServer): Promise<(msg: unknown) => void> {
  let fromWebview: (msg: unknown) => void = () => {};
  const provider = new MimirAgentViewProvider(
    {} as never,
    { get: () => undefined, update: async () => {} } as never,
    server,
  );
  provider.resolveWebviewView({
    webview: {
      options: {},
      html: "",
      cspSource: "",
      asWebviewUri: () => "uri",
      postMessage: async () => true,
      onDidReceiveMessage: (handler: (msg: unknown) => void) => { fromWebview = handler; },
    },
    visible: true,
  } as never);
  await new Promise((resolve) => setTimeout(resolve, 20));
  return fromWebview;
}

describe("the chat view and the server it leaves", () => {
  it("ends the server on disconnect", async () => {
    const child = aServer();
    const server = new LocalServer("/ext");
    server.adopt(child.pid!);
    const send = await aView(server);
    send({ type: "disconnect" });
    expect(await ended(child)).toBe(true);
  });

  it("spares it as soon as detaching is asked, before the server answers", async () => {
    const child = aServer();
    const server = new LocalServer("/ext");
    server.adopt(child.pid!);
    const send = await aView(server);
    send({ type: "ws_send", payload: JSON.stringify({ type: "detach", enabled: true }) });
    send({ type: "disconnect" });
    expect(await ended(child)).toBe(false);
  });

  it("can end a server it attached to rather than spawned", async () => {
    const child = aServer();
    registry.live = { pid: child.pid!, url: "ws://127.0.0.1:1" };
    const send = await aView(new LocalServer("/ext"));
    send({ type: "disconnect" });
    expect(await ended(child)).toBe(true);
  });
});
