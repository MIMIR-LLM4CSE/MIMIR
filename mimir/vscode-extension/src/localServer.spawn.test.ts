import { EventEmitter } from "events";
import { describe, expect, it, vi } from "vitest";

const children: EventEmitter[] = [];
const spawned: { command: string; args: string[]; options: Record<string, unknown> }[] = [];
const opened: { file: string; mode: unknown }[] = [];
const channels: string[] = [];
const lines: string[] = [];
const errors: string[] = [];
const registry = { started: true };
const made: string[] = [];
const disk = { denied: false };

vi.mock("vscode", () => ({
  workspace: {
    getConfiguration: () => ({
      get: (_key: string, fallback?: unknown) => fallback,
      inspect: (key: string) => ({ key }),
    }),
  },
  window: {
    createOutputChannel: (name: string) => {
      channels.push(name);
      return { show() {}, append() {}, appendLine: (line: string) => lines.push(line) };
    },
  },
}));

vi.mock("child_process", () => ({
  spawn: (command: string, args: string[], options: Record<string, unknown>) => {
    spawned.push({ command, args, options });
    const child = Object.assign(new EventEmitter(), { pid: 4242, killed: false, unref() {}, kill() {} });
    children.push(child);
    return child;
  },
}));

vi.mock("fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("fs")>();
  return {
    ...actual,
    mkdirSync: (dir: string) => {
      if (disk.denied) throw new Error(`EACCES: permission denied, mkdir '${dir}'`);
      made.push(dir);
    },
    openSync: (file: string, _flags: string, mode: unknown) => {
      opened.push({ file, mode });
      return 99;
    },
    closeSync: () => {},
  };
});

vi.mock("./serverRegistry", () => ({
  logPath: (_cwd: string, stamp: string) => `/state/logs/server-${stamp}.log`,
  stateDir: () => "/state",
  waitForServer: () => registry.started
    ? Promise.resolve({ pid: 4242, url: "ws://127.0.0.1:5555" })
    : new Promise(() => {}),
  findLiveServer: async () => undefined,
}));

vi.mock("./notifications", () => ({
  showToast: (_kind: string, message: string) => {
    errors.push(message);
    return Promise.resolve(undefined);
  },
}));

vi.mock("./diskFull", () => ({
  diagnoseDiskFull: () => ({ reason: "quota", where: "/state/logs" }),
  describeDiskFull: () => "MIMIR could not start: your disk quota is exceeded",
}));

import { LocalServer } from "./localServer";

describe("spawning the server", () => {
  it("starts it in its own session, writing to a file, and returns its address", async () => {
    const server = new LocalServer("/ext");
    const outcome = await server.spawn(
      { model: "m", backend: "vllm", baseUrl: "http://node:8000/v1", anthropicApiKey: "", silent: true },
      () => {},
    );
    expect(outcome).toEqual({ kind: "listening", url: "ws://127.0.0.1:5555" });
    const { command, args, options } = spawned[0];
    // Its own session and no pipe: nothing about this window closing reaches it, and a
    // server meant to outlive the window does not die on its first write to a dead pipe.
    expect(options.detached).toBe(true);
    expect(options.stdio).toEqual(["ignore", 99, 99]);
    expect(opened[0].file).toMatch(/^\/state\/logs\/server-[^/]+\.log$/);
    expect(opened[0].mode).toBe(0o600);
    expect(command).toBe("python3");
    expect(args.slice(0, 2)).toEqual(["-m", "mimir.client.ui.ws.server.main"]);
    expect(args.join(" ")).toContain("--backend vllm --vllm-base-url http://node:8000/v1");
    expect(options.cwd).toBe("/ext");
    expect(args[args.indexOf("--cwd") + 2]).toBe("--port");
    expect(made).toContain("/state/logs");
    expect((options.env as NodeJS.ProcessEnv).PYTHONUNBUFFERED).toBe("1");
    expect(server.childRunning).toBe(true);
  });

  it("names a state dir it cannot create instead of starting a server that cannot log", async () => {
    const before = spawned.length;
    errors.length = 0;
    disk.denied = true;
    try {
      const outcome = await new LocalServer("/ext").spawn(
        { model: "m", backend: "vllm", baseUrl: "http://node:8000/v1", anthropicApiKey: "", silent: true },
        () => {},
      );
      expect(outcome).toEqual({ kind: "gave_up" });
    } finally {
      disk.denied = false;
    }
    expect(spawned.length).toBe(before);
    expect(errors.join("\n")).toContain("cannot create its state directory /state/logs");
    expect(errors.join("\n")).toContain("EACCES");
  });

  it("writes every start to the one server channel", async () => {
    const server = new LocalServer("/ext");
    const request = { model: "m", backend: "ollama", baseUrl: "http://node:11434", anthropicApiKey: "", silent: true };
    await server.spawn(request, () => {});
    await server.spawn(request, () => {});
    expect(channels.filter((name) => name === "MIMIR Server")).toHaveLength(1);
  });

  it("keeps the Claude key out of the command line", async () => {
    spawned.length = 0;
    await new LocalServer("/ext").spawn(
      { model: "claude", backend: "anthropic", baseUrl: "", anthropicApiKey: "sk-secret", silent: true },
      () => {},
    );
    const { args, options } = spawned[0];
    expect(args.join(" ")).not.toContain("sk-secret");
    expect((options.env as Record<string, string>).ANTHROPIC_API_KEY).toBe("sk-secret");
  });

  it("passes a model name and an endpoint as words, never as shell", async () => {
    // The model list is whatever answers on the endpoint's port, which another user of
    // a login node can serve: a name is one argument, whatever it holds.
    spawned.length = 0;
    const model = "x; touch /tmp/pwned #";
    const baseUrl = "http://127.0.0.1:8000/v1$(id)";
    await new LocalServer("/ext").spawn(
      { model, backend: "vllm", baseUrl, anthropicApiKey: "", silent: true }, () => {});
    const { command, args } = spawned[0];
    expect(command).not.toBe("bash");
    expect(args[args.indexOf("--model") + 1]).toBe(model);
    expect(args[args.indexOf("--vllm-base-url") + 1]).toBe(baseUrl);
  });

  it("names an exceeded quota in its output when the server dies before it listens", async () => {
    registry.started = false;
    lines.length = 0;
    errors.length = 0;
    try {
      let exited = false;
      void new LocalServer("/ext").spawn(
        { model: "m", backend: "vllm", baseUrl: "http://node:8000/v1", anthropicApiKey: "", silent: true },
        () => { exited = true; },
      );
      children[children.length - 1].emit("exit", 1);
      expect(exited).toBe(true);
      expect(lines).toContain("MIMIR could not start: your disk quota is exceeded");
      expect(errors).toEqual(["MIMIR could not start: your disk quota is exceeded"]);
    } finally {
      registry.started = true;
    }
  });

  it("says nothing more of an exit once the server has listened", async () => {
    await new LocalServer("/ext").spawn(
      { model: "m", backend: "vllm", baseUrl: "http://node:8000/v1", anthropicApiKey: "", silent: true },
      () => {},
    );
    lines.length = 0;
    errors.length = 0;
    children[children.length - 1].emit("exit", 1);
    expect(lines.join("\n")).not.toContain("could not start");
    expect(errors).toEqual([]);
  });
});
