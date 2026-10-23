import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { afterEach, describe, expect, it, vi } from "vitest";

/** The error a write is refused with, and under which directory (everywhere if unset). */
const refusal: { code?: string; under?: string } = {};

vi.mock("fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("fs")>();
  const refusedFds = new Set<number>();
  return {
    ...actual,
    openSync: (file: string, flags: string) => {
      const fd = actual.openSync(file, flags);
      if (flags === "w" && refusal.code && (!refusal.under || file.startsWith(refusal.under))) {
        refusedFds.add(fd);
      }
      return fd;
    },
    fsyncSync: (fd: number) => {
      if (refusedFds.delete(fd)) {
        throw Object.assign(new Error(refusal.code), { code: refusal.code });
      }
      return actual.fsyncSync(fd);
    },
  };
});

import { describeDiskFull, diagnoseDiskFull, diskFullInLog, probeDiskFull } from "./diskFull";

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "mimir-diskfull-"));
}

afterEach(() => {
  refusal.code = undefined;
  refusal.under = undefined;
});

describe("reading a full disk from the log", () => {
  it("takes the reason and the path from a Python traceback", () => {
    const log = "Traceback (most recent call last):\n" +
      "OSError: [Errno 122] Disk quota exceeded: '/home/u/.mimir/ws-1234/server.json'\n";
    expect(diskFullInLog(log)).toEqual({
      reason: "quota", where: "/home/u/.mimir/ws-1234/server.json",
    });
  });

  it("knows a full disk from an exceeded quota, and a shell error has no path to give", () => {
    expect(diskFullInLog("mkdir: cannot create directory 'x': No space left on device"))
      .toEqual({ reason: "space", where: undefined });
  });

  it("finds nothing in an ordinary failure", () => {
    expect(diskFullInLog("ConnectionRefusedError: [Errno 111] Connection refused")).toBeUndefined();
  });
});

describe("probing the log's directory", () => {
  it("says nothing of a directory that takes writes, and leaves nothing behind", () => {
    const dir = tempDir();
    expect(probeDiskFull(dir)).toBeUndefined();
    expect(fs.readdirSync(dir)).toEqual([]);
  });

  it("reports a refused write at the nearest directory that exists", () => {
    const dir = tempDir();
    refusal.code = "EDQUOT";
    expect(probeDiskFull(path.join(dir, "logs"))).toEqual({ reason: "quota", where: dir });
    expect(fs.readdirSync(dir)).toEqual([]);
  });

  it("does not take another failure for a full disk", () => {
    refusal.code = "EIO";
    expect(probeDiskFull(tempDir())).toBeUndefined();
  });
});

describe("diagnosing a server's exit", () => {
  it("falls back on the probe when the log is empty", () => {
    const dir = tempDir();
    const log = path.join(dir, "server.log");
    fs.writeFileSync(log, "");
    refusal.code = "ENOSPC";
    expect(diagnoseDiskFull(log)).toEqual({ reason: "space", where: dir });
  });

  it("tries each place the server writes, past a log directory that has room", () => {
    const logs = tempDir();
    const workspace = tempDir();
    const log = path.join(logs, "server.log");
    fs.writeFileSync(log, "");
    refusal.code = "EDQUOT";
    refusal.under = workspace;
    expect(diagnoseDiskFull(log, [path.join(logs, "state"), workspace]))
      .toEqual({ reason: "quota", where: workspace });
    expect(fs.readdirSync(workspace)).toEqual([]);
  });

  it("names the cause and what to do about it", () => {
    const message = describeDiskFull({ reason: "quota", where: "/home/u/.mimir" });
    expect(message).toContain("disk quota is exceeded");
    expect(message).toContain("/home/u/.mimir");
    expect(message).toContain("quota -s");
  });
});
