/**
 * Finding the MIMIR server this workspace already has, from `<state dir>/server.json`.
 *
 * `workspaceId` must stay byte-identical to `state_paths.workspace_id` (pinned by
 * serverRegistry.test.ts). An entry is live when its pid exists, its start time matches, and
 * the server at its port names that pid; a stale entry costs one failed attempt.
 */
import * as crypto from "crypto";
import * as fs from "fs";
import * as http from "http";
import * as os from "os";
import * as path from "path";

/** Bumped when the file's shape changes incompatibly; an unknown version is refused. */
export const PROTOCOL = 1;

/**
 * Where the server answers its liveness probe with its pid, as plain HTTP — `ALIVE_PATH`
 * in the Python registry.
 */
export const ALIVE_PATH = "/alive";

/**
 * How long to wait for the answer: a round trip through the server's event loop, which a
 * busy turn can hold for a moment. A dead address is refused at once and does not wait.
 */
const PROBE_TIMEOUT_MS = 2000;

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
  /** What a connection presents; absent from a server that asks for none. */
  token?: string;
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

/**
 * Where this workspace's state lives, resolved the way the Python side resolves it from
 * *env* (this process's by default; a server's own when it is given one).
 */
export function stateDir(workspaceRoot: string, env: NodeJS.ProcessEnv = process.env): string {
  const explicit = env.MIMIR_STATE_DIR;
  if (explicit) return path.resolve(explicit);
  const home = env.MIMIR_STATE_HOME || path.join(os.homedir(), ".mimir");
  return path.resolve(path.join(home, workspaceId(workspaceRoot)));
}

export function registryPath(workspaceRoot: string, env: NodeJS.ProcessEnv = process.env): string {
  return path.join(stateDir(workspaceRoot, env), "server.json");
}

/** Where a server started by this extension, with *env*, writes its output. */
export function logPath(
  workspaceRoot: string, stamp: string, env: NodeJS.ProcessEnv = process.env,
): string {
  return path.join(stateDir(workspaceRoot, env), "logs", `server-${stamp}.log`);
}

/**
 * Wait for a server to publish its address, polling the registry (the file is created,
 * renamed and possibly replaced, which a watcher handles badly).
 *
 * Resolves with our pid's entry, or with another live server's: the one holding this
 * workspace's lock, which ours then yields to.
 */
export async function waitForServer(
  workspaceRoot: string,
  ourPid: number | undefined,
  env: NodeJS.ProcessEnv = process.env,
  { timeoutMs = 120000, intervalMs = 250 }: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<ServerEntry | undefined> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const entry = readEntry(workspaceRoot, env);
    if (entry) {
      if (ourPid !== undefined && entry.pid === ourPid) return entry;
      // Somebody else's, and alive: ours lost the lock and stood down.
      if (processAlive(entry) && (await serverAnswers(entry))) return entry;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return undefined;
}

/** The recorded entry, or undefined when there is none we can understand. */
export function readEntry(
  workspaceRoot: string, env: NodeJS.ProcessEnv = process.env,
): ServerEntry | undefined {
  try {
    const raw = fs.readFileSync(registryPath(workspaceRoot, env), "utf8");
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

/**
 * Whether the server at the entry's address is the entry's process, asked at `ALIVE_PATH`.
 *
 * A server from before the probe existed refuses the plain request with 426 (Upgrade
 * Required): a WebSocket server, taken as the entry's since its pid check passed.
 */
export function serverAnswers(
  entry: Pick<ServerEntry, "host" | "port" | "pid">, timeoutMs = PROBE_TIMEOUT_MS,
): Promise<boolean> {
  return new Promise((resolve) => {
    if (!entry.port) {
      resolve(false);
      return;
    }
    let settled = false;
    const finish = (answer: boolean) => {
      if (settled) return;
      settled = true;
      resolve(answer);
    };
    // agent: false — a fresh socket, not one a keep-alive pool may hand back later.
    const req = http.get(
      { host: entry.host || "127.0.0.1", port: entry.port, path: ALIVE_PATH, agent: false,
        timeout: timeoutMs },
      (res) => {
        let body = "";
        res.setEncoding("ascii");
        res.on("data", (chunk: string) => { body += chunk; });
        res.on("end", () => finish(
          res.statusCode === 426
            || (res.statusCode === 200 && body.trim() === String(entry.pid)),
        ));
        res.on("error", () => finish(false));
      },
    );
    req.on("timeout", () => { req.destroy(); finish(false); });
    req.on("error", () => finish(false));
  });
}

/**
 * The entry if this workspace has a server that is actually there, else undefined.
 *
 * Both halves, in the cheap-first order: the process table rules out a crashed server
 * without opening a socket, and the probe rules out one whose listener has gone.
 */
export async function findLiveServer(
  workspaceRoot: string, env: NodeJS.ProcessEnv = process.env,
): Promise<ServerEntry | undefined> {
  const entry = readEntry(workspaceRoot, env);
  if (!entry) return undefined;
  if (!processAlive(entry)) return undefined;
  if (!(await serverAnswers(entry))) return undefined;
  return entry;
}

/** *entry*'s token when *url* dials that server on this machine; no other host gets it. */
export function tokenFor(url: string, entry: ServerEntry | undefined): string | undefined {
  try {
    const { hostname, port } = new URL(url);
    const local = ["localhost", "127.0.0.1", "[::1]"].includes(hostname);
    return local && Number(port) === entry?.port ? entry.token : undefined;
  } catch {
    return undefined;
  }
}
