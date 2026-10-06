/**
 * Finding — or correctly failing to find — this workspace's server.
 *
 * Two things are load-bearing. `workspaceId` must agree byte for byte with the Python
 * side, because that agreement *is* "one server per workspace": disagree and each end
 * looks in a different file and both think there is no server. And liveness must reject
 * a stale entry, because a window that trusts one waits on an address nothing is
 * listening at.
 */
import { createHash } from "crypto";
import * as fs from "fs";
import * as net from "net";
import * as os from "os";
import * as path from "path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  PROTOCOL, findLiveServer, portAnswers, processAlive, readEntry, registryPath,
  stateDir, workspaceId,
} from "./serverRegistry";

let tmp: string;
const savedEnv = { ...process.env };

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mimir-reg-"));
});

afterEach(() => {
  process.env = { ...savedEnv };
  fs.rmSync(tmp, { recursive: true, force: true });
});

function writeEntry(root: string, entry: Record<string, unknown>): void {
  const dir = stateDir(root);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "server.json"), JSON.stringify(entry));
}

function baseEntry(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    protocol: PROTOCOL,
    pid: process.pid,
    pid_starttime: null,
    url: "ws://127.0.0.1:4321",
    host: "127.0.0.1",
    port: 4321,
    ...over,
  };
}

/** A real listening socket, so the probe has something true to find. */
async function listening(): Promise<{ port: number; close: () => void }> {
  const server = net.createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as net.AddressInfo).port;
  return { port, close: () => server.close() };
}

/** A port nothing is listening on. */
async function freePort(): Promise<number> {
  const { port, close } = await listening();
  close();
  return port;
}

describe("workspaceId", () => {
  it("is the basename plus a short hash of the resolved path", () => {
    const real = fs.realpathSync(tmp);
    const digest = createHash("sha1").update(real, "utf8").digest("hex").slice(0, 8);
    expect(workspaceId(tmp)).toBe(`${path.basename(real)}-${digest}`);
  });

  it("matches what the Python side produces, for paths that exist everywhere", () => {
    // Fixtures, not a recomputation: recomputing the formula here would pass even if
    // both ends changed together in the same wrong way, and the agreement is the whole
    // of "one server per workspace" — disagree and each end looks in a different file
    // while both conclude there is no server.
    //
    // Captured from `state_paths.workspace_id` on these absolute paths. `/` is in the
    // list for its fallback: basename("/") is empty and both ends must say "root".
    expect(workspaceId("/tmp")).toBe("tmp-8c393341");
    expect(workspaceId("/")).toBe("root-42099b4a");
  });

  it("distinguishes two checkouts that share a basename", () => {
    const a = fs.mkdtempSync(path.join(tmp, "a-"));
    const b = fs.mkdtempSync(path.join(tmp, "b-"));
    const projA = path.join(a, "MIMIR");
    const projB = path.join(b, "MIMIR");
    fs.mkdirSync(projA);
    fs.mkdirSync(projB);
    expect(workspaceId(projA)).not.toBe(workspaceId(projB));
    expect(workspaceId(projA).startsWith("MIMIR-")).toBe(true);
  });

  it("falls back to the resolved path when the directory is gone", () => {
    const missing = path.join(tmp, "never-existed");
    expect(() => workspaceId(missing)).not.toThrow();
  });
});

describe("stateDir", () => {
  it("honours MIMIR_STATE_DIR above everything", () => {
    process.env.MIMIR_STATE_DIR = path.join(tmp, "explicit");
    expect(stateDir(tmp)).toBe(path.resolve(path.join(tmp, "explicit")));
  });

  it("puts per-workspace dirs under MIMIR_STATE_HOME", () => {
    delete process.env.MIMIR_STATE_DIR;
    process.env.MIMIR_STATE_HOME = path.join(tmp, "home");
    expect(stateDir(tmp)).toBe(
      path.resolve(path.join(tmp, "home", workspaceId(tmp))));
  });

  it("names the entry server.json inside it", () => {
    process.env.MIMIR_STATE_DIR = tmp;
    expect(registryPath(tmp)).toBe(path.join(tmp, "server.json"));
  });
});

describe("readEntry", () => {
  beforeEach(() => {
    process.env.MIMIR_STATE_DIR = path.join(tmp, "state");
  });

  it("reads back what was written", () => {
    writeEntry(tmp, baseEntry({ model: "m" }));
    expect(readEntry(tmp)?.model).toBe("m");
    expect(readEntry(tmp)?.port).toBe(4321);
  });

  it("returns nothing when there is no file", () => {
    expect(readEntry(tmp)).toBeUndefined();
  });

  it("returns nothing for a file that is not JSON", () => {
    const dir = stateDir(tmp);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, "server.json"), "{not json");
    expect(readEntry(tmp)).toBeUndefined();
  });

  it("refuses an entry written for another protocol", () => {
    // Connecting to a server whose contract has changed is worse than deciding there
    // is none.
    writeEntry(tmp, baseEntry({ protocol: PROTOCOL + 1 }));
    expect(readEntry(tmp)).toBeUndefined();
  });

  it("refuses an entry missing the address", () => {
    writeEntry(tmp, baseEntry({ url: "", port: 0 }));
    expect(readEntry(tmp)).toBeUndefined();
  });
});

describe("processAlive", () => {
  it("accepts this very process", () => {
    expect(processAlive(baseEntry() as never)).toBe(true);
  });

  it("rejects a pid that is gone", () => {
    expect(processAlive(baseEntry({ pid: 4194301 }) as never)).toBe(false);
  });

  it("rejects a recycled pid", () => {
    // Alive, but it started at a different time: a different process wears that
    // number now.
    expect(processAlive(baseEntry({ pid_starttime: 1 }) as never)).toBe(false);
  });
});

describe("portAnswers", () => {
  it("is true for a listening port", async () => {
    const { port, close } = await listening();
    try {
      expect(await portAnswers("127.0.0.1", port)).toBe(true);
    } finally {
      close();
    }
  });

  it("is false for a port nothing is on", async () => {
    expect(await portAnswers("127.0.0.1", await freePort())).toBe(false);
  });

  it("is false for port zero", async () => {
    expect(await portAnswers("127.0.0.1", 0)).toBe(false);
  });
});

describe("findLiveServer", () => {
  beforeEach(() => {
    process.env.MIMIR_STATE_DIR = path.join(tmp, "state");
  });

  it("finds a server that is actually there", async () => {
    const { port, close } = await listening();
    try {
      writeEntry(tmp, baseEntry({ port, url: `ws://127.0.0.1:${port}` }));
      const found = await findLiveServer(tmp);
      expect(found?.url).toBe(`ws://127.0.0.1:${port}`);
    } finally {
      close();
    }
  });

  it("rejects an entry whose listener has gone", async () => {
    // The case the process table cannot catch: this process is alive and nothing is
    // accepting on that port. A window that trusted it would wait on a dead address.
    const port = await freePort();
    writeEntry(tmp, baseEntry({ port, url: `ws://127.0.0.1:${port}` }));
    expect(await findLiveServer(tmp)).toBeUndefined();
  });

  it("rejects an entry whose process has gone", async () => {
    const { port, close } = await listening();
    try {
      writeEntry(tmp, baseEntry({ pid: 4194301, port }));
      expect(await findLiveServer(tmp)).toBeUndefined();
    } finally {
      close();
    }
  });

  it("finds nothing when no server was ever published", async () => {
    expect(await findLiveServer(tmp)).toBeUndefined();
  });

  it("does not look in another workspace's entry", async () => {
    const { port, close } = await listening();
    try {
      writeEntry(tmp, baseEntry({ port }));
      delete process.env.MIMIR_STATE_DIR;
      process.env.MIMIR_STATE_HOME = path.join(tmp, "home");
      const other = fs.mkdtempSync(path.join(tmp, "other-"));
      expect(await findLiveServer(other)).toBeUndefined();
    } finally {
      close();
    }
  });
});
