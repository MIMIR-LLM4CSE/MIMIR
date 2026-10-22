import { EventEmitter } from "events";
import { describe, expect, it, vi } from "vitest";

const spawned: { command: string; options: Record<string, unknown> }[] = [];
const channels: string[] = [];

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
      return { show() {}, append() {}, appendLine() {} };
    },
  },
}));

vi.mock("child_process", () => ({
  spawn: (_shell: string, args: string[], options: Record<string, unknown>) => {
    spawned.push({ command: args[1], options });
    return Object.assign(new EventEmitter(), { pid: 4242, killed: false, unref() {}, kill() {} });
  },
}));

vi.mock("./serverRegistry", () => ({
  logPath: (_cwd: string, stamp: string) => `/state/logs/server-${stamp}.log`,
  waitForServer: async () => ({ pid: 4242, url: "ws://127.0.0.1:5555" }),
  findLiveServer: async () => undefined,
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
    const { command, options } = spawned[0];
    // Its own session and no pipe: nothing about this window closing reaches it, and a
    // server meant to outlive the window does not die on its first write to a dead pipe.
    expect(options.detached).toBe(true);
    expect(options.stdio).toBe("ignore");
    expect(command).toMatch(/>> '\/state\/logs\/server-[^']+\.log' 2>&1$/);
    expect(command).toContain("--backend vllm --vllm-base-url http://node:8000/v1");
    expect(server.childRunning).toBe(true);
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
    const { command, options } = spawned[0];
    expect(command).not.toContain("sk-secret");
    expect((options.env as Record<string, string>).ANTHROPIC_API_KEY).toBe("sk-secret");
  });
});
