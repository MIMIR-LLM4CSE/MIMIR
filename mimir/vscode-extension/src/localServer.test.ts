import * as cp from "child_process";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("vscode", () => ({ workspace: {}, window: {} }));

import { LocalServer } from "./localServer";

const children: cp.ChildProcess[] = [];

/** A process standing in for a server this window attached to. */
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

afterEach(() => {
  for (const child of children.splice(0)) child.kill("SIGKILL");
});

describe("the server this window is responsible for", () => {
  it("is ended when the window releases it", async () => {
    const child = aServer();
    const server = new LocalServer("/ext");
    server.adopt(child.pid!);
    server.release();
    expect(await ended(child)).toBe(true);
    expect(child.signalCode).toBe("SIGTERM");
  });

  it("is left running once it has detached", async () => {
    const child = aServer();
    const server = new LocalServer("/ext");
    server.adopt(child.pid!);
    server.noteDetached(true);
    server.release();
    expect(await ended(child)).toBe(false);
  });

  it("is ended again once the detachment is taken back", async () => {
    const child = aServer();
    const server = new LocalServer("/ext");
    server.adopt(child.pid!);
    server.noteDetached(true);
    server.noteDetached(false);
    server.endUnlessKeptRunning();
    expect(await ended(child)).toBe(true);
  });

  it("is forgotten after a release, so a second one signals nothing", async () => {
    const child = aServer();
    const server = new LocalServer("/ext");
    server.adopt(child.pid!);
    server.noteDetached(true);
    server.release();
    server.noteDetached(false);
    server.release();
    expect(await ended(child)).toBe(false);
  });
});
