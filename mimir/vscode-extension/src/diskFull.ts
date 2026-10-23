/**
 * Telling a server that died of a full disk or an exceeded quota from any other exit.
 *
 * Such a server often leaves nothing to read: the shell's own error (creating the log
 * directory, opening the log) goes nowhere, since the child has no pipe, and the log it
 * did open takes no writes. So the log is read for the message when there is one, and
 * otherwise a small write is tried where the server writes, which fails the same way.
 */
import * as fs from "fs";
import * as path from "path";

/** What `strerror` says for EDQUOT and ENOSPC, as Python, bash and coreutils print it. */
const MESSAGE = /Disk quota exceeded|No space left on device/;
/** Python's `OSError: [Errno 122] Disk quota exceeded: '/the/path'`. */
const PYTHON_PATH = /\[Errno \d+\] (?:Disk quota exceeded|No space left on device): '([^']+)'/;
/** How much of the log's end is searched. */
const TAIL_BYTES = 16384;
/** Enough that a quota at its hard limit refuses it. */
const PROBE_BYTES = 4096;

/** Why the server could not write: the reason and, when known, where. */
export interface DiskFull {
  reason: "quota" | "space";
  /** The file or directory the failed write was to, when it is known. */
  where?: string;
}

/** The last *bytes* of *file*, or "" when it cannot be read. */
function tail(file: string, bytes: number): string {
  try {
    const size = fs.statSync(file).size;
    const length = Math.min(size, bytes);
    const fd = fs.openSync(file, "r");
    try {
      const buf = Buffer.alloc(length);
      fs.readSync(fd, buf, 0, length, size - length);
      return buf.toString();
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return "";
  }
}

/** The full-disk error *text* reports, if it reports one. */
export function diskFullInLog(text: string): DiskFull | undefined {
  const match = MESSAGE.exec(text);
  if (!match) return undefined;
  const reason = match[0].startsWith("Disk quota") ? "quota" : "space";
  return { reason, where: PYTHON_PATH.exec(text)?.[1] };
}

/** *dir*, or its nearest ancestor that exists: where a directory being created would go. */
function existingAncestor(dir: string): string {
  let current = path.resolve(dir);
  while (!fs.existsSync(current)) {
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return current;
}

/** Whether a write into *dir* is refused for want of space or quota. */
export function probeDiskFull(dir: string): DiskFull | undefined {
  const target = existingAncestor(dir);
  const probe = path.join(target, `.mimir-write-probe-${process.pid}`);
  let fd: number | undefined;
  try {
    fd = fs.openSync(probe, "w");
    fs.writeSync(fd, Buffer.alloc(PROBE_BYTES));
    // A network filesystem may accept the write and refuse it only here.
    fs.fsyncSync(fd);
    return undefined;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "EDQUOT") return { reason: "quota", where: target };
    if (code === "ENOSPC") return { reason: "space", where: target };
    return undefined;
  } finally {
    if (fd !== undefined) {
      try {
        fs.closeSync(fd);
      } catch {
        // The close may report the same refusal; the probe has already answered.
      }
    }
    try {
      fs.unlinkSync(probe);
    } catch {
      // Never created.
    }
  }
}

/**
 * Whether the server logging to *logFile* died for want of space: from its log, else
 * from a probe of the log's directory and then of each of *writesTo*.
 */
export function diagnoseDiskFull(logFile: string, writesTo: string[] = []): DiskFull | undefined {
  const fromLog = diskFullInLog(tail(logFile, TAIL_BYTES));
  if (fromLog) return fromLog;
  for (const dir of [path.dirname(logFile), ...writesTo]) {
    const full = probeDiskFull(dir);
    if (full) return full;
  }
  return undefined;
}

/** The error shown to the user. */
export function describeDiskFull(full: DiskFull): string {
  const what = full.reason === "quota" ? "your disk quota is exceeded" : "the disk is full";
  const where = full.where ? ` (writing to ${full.where})` : "";
  const check = full.reason === "quota" ? "`quota -s`" : "`df -h`";
  return (
    `MIMIR could not start: ${what}${where}. Free some space (${check} shows how much ` +
    "is left), then connect again."
  );
}
