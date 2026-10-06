/**
 * Finding the MIMIR server this workspace already has.
 *
 * The connect path learns a spawned server's port by regexing its stdout, which works
 * exactly as long as this extension host is the parent holding that pipe. A server that
 * outlived the window which started it cannot be found that way, so it writes its
 * address to `<state dir>/server.json` and this module reads it back.
 *
 * The state dir is the same one the Python side computes, and agreeing on it is the
 * whole of "one server per workspace": `<state home>/<basename>-<sha1(realpath)[:8]>`.
 * The hash is of the resolved path, so two checkouts that share a basename do not
 * collide; `workspaceId` here must stay byte-identical to `state_paths.workspace_id`
 * there, which is what `extension.test.ts` pins.
 *
 * Liveness is three questions, and the third is the one the other two cannot answer:
 * the pid exists; its start time matches what was recorded, because a recycled pid
 * wears the same number; and the port actually accepts a connection, because a process
 * can be alive with its listener already gone.
 *
 * The entry is advisory. A stale one costs a connection attempt that fails and is then
 * replaced — treating it as proof a server exists is how a window ends up waiting on an
 * address nothing is listening at.
 */
import * as crypto from "crypto";
import * as fs from "fs";
import * as net from "net";
import * as os from "os";
import * as path from "path";

/** Bumped when the file's shape changes incompatibly; an unknown version is refused. */
export const PROTOCOL = 1;

/** How long to wait for the port to answer — a loopback accept, not a round trip. */
const PROBE_TIMEOUT_MS = 300;

export interface ServerEntry {
  protocol: number;
  pid: number;
  pid_starttime?: number | null;
  url: string;
  host: string;
  port: number;
  workspace?: string;
  state_dir?: string;
  started_at?: number;
  model?: string;
  autonomy?: string;
  /** True once the server has made itself survivable; see `detach.py`. */
  detached?: boolean;
  /** Where its output continues, once it no longer has a pipe to write to. */
  log?: string | null;
}

/** `<basename>-<sha1(realpath)[:8]>` — must match `state_paths.workspace_id`. */
export function workspaceId(root: string): string {
  let real: string;
  try {
    real = fs.realpathSync(root);
  } catch {
    real = path.resolve(root);
  }
  const digest = crypto.createHash("sha1").update(real, "utf8").digest("hex").slice(0, 8);
  const base = path.basename(real) || "root";
  return `${base}-${digest}`;
}

/** Where this workspace's state lives, resolved the way the Python side resolves it. */
export function stateDir(workspaceRoot: string): string {
  const explicit = process.env.MIMIR_STATE_DIR;
  if (explicit) return path.resolve(explicit);
  const home = process.env.MIMIR_STATE_HOME || path.join(os.homedir(), ".mimir");
  return path.resolve(path.join(home, workspaceId(workspaceRoot)));
}

export function registryPath(workspaceRoot: string): string {
  return path.join(stateDir(workspaceRoot), "server.json");
}

/** Where a server started by this extension writes its output. */
export function logPath(workspaceRoot: string, stamp: string): string {
  return path.join(stateDir(workspaceRoot), "logs", `server-${stamp}.log`);
}

/**
 * Wait for a server to publish its address, polling the registry.
 *
 * A server spawned in its own process session has no pipe back here, so its address
 * cannot be read off its stdout: it writes `server.json` once the socket is bound and
 * this reads it. Polling rather than watching, because the file is created, renamed
 * into place and possibly replaced, and a watcher on a path that does not exist yet is
 * the harder thing to get right.
 *
 * Resolves with the entry whose `pid` is *ours* — or, when one appears for another live
 * pid, with that one: a server that loses this workspace's lock exits without serving,
 * so the right answer is the server that already holds it.
 */
export async function waitForServer(
  workspaceRoot: string,
  ourPid: number | undefined,
  { timeoutMs = 120000, intervalMs = 250 }: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<ServerEntry | undefined> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const entry = readEntry(workspaceRoot);
    if (entry) {
      if (ourPid !== undefined && entry.pid === ourPid) return entry;
      // Somebody else's, and alive: ours lost the lock and stood down.
      if (processAlive(entry) && (await portAnswers(entry.host, entry.port))) return entry;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return undefined;
}

/** The recorded entry, or undefined when there is none we can understand. */
export function readEntry(workspaceRoot: string): ServerEntry | undefined {
  try {
    const raw = fs.readFileSync(registryPath(workspaceRoot), "utf8");
    const entry = JSON.parse(raw) as ServerEntry;
    if (!entry || typeof entry !== "object") return undefined;
    // Connecting to a server whose contract has changed is worse than deciding there
    // is none.
    if (Number(entry.protocol) !== PROTOCOL) return undefined;
    if (!entry.url || !entry.port || !entry.pid) return undefined;
    return entry;
  } catch {
    return undefined;
  }
}

/** Field 22 of `/proc/<pid>/stat` — the clock tick the process started at. */
function procStarttime(pid: number): number | undefined {
  try {
    const data = fs.readFileSync(`/proc/${pid}/stat`, "utf8");
    // The comm field is parenthesized and may contain spaces: split after its close.
    const fields = data.slice(data.lastIndexOf(")") + 2).split(" ");
    const value = Number(fields[19]);
    return Number.isFinite(value) ? value : undefined;
  } catch {
    return undefined;
  }
}

/** Whether that pid is the process the entry was written by. */
export function processAlive(entry: ServerEntry): boolean {
  try {
    // Signal 0 tests for existence without delivering anything. EPERM means it exists
    // and belongs to somebody else, which for our purposes is still "there".
    process.kill(entry.pid, 0);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "EPERM") return false;
  }
  if (typeof entry.pid_starttime === "number") {
    const actual = procStarttime(entry.pid);
    // A recycled pid wears the same number; only the start time tells them apart.
    if (actual !== undefined && actual !== entry.pid_starttime) return false;
  }
  return true;
}

/** Whether something accepts a connection at that address. */
export function portAnswers(
  host: string, port: number, timeoutMs = PROBE_TIMEOUT_MS,
): Promise<boolean> {
  return new Promise((resolve) => {
    if (!port) {
      resolve(false);
      return;
    }
    const sock = new net.Socket();
    let settled = false;
    const finish = (answer: boolean) => {
      if (settled) return;
      settled = true;
      sock.destroy();
      resolve(answer);
    };
    sock.setTimeout(timeoutMs);
    sock.once("connect", () => finish(true));
    sock.once("timeout", () => finish(false));
    sock.once("error", () => finish(false));
    sock.connect(port, host || "127.0.0.1");
  });
}

/**
 * The entry if this workspace has a server that is actually there, else undefined.
 *
 * Both halves, in the cheap-first order: the process table rules out a crashed server
 * without opening a socket, and the probe rules out one whose listener has gone.
 */
export async function findLiveServer(
  workspaceRoot: string,
): Promise<ServerEntry | undefined> {
  const entry = readEntry(workspaceRoot);
  if (!entry) return undefined;
  if (!processAlive(entry)) return undefined;
  if (!(await portAnswers(entry.host, entry.port))) return undefined;
  return entry;
}
