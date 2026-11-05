import type { PanelGroup, PanelSection, RunOrigin, WatchedRun } from "../types";

/**
 * The logic behind the panel's tree, kept out of the component so it can be read and
 * tested without a browser. Nothing here knows what a group means — a scheduler, a kind
 * of machine, something a future server invents — only how to open one and how to say
 * what is running on it.
 */

/** The key a badge's own disclosure row is opened under, distinct from any group id. */
export function badgeKey(run: WatchedRun): string {
  return `badge:${run.job_key}`;
}

/** Flip one row open or closed. A new object, so React sees the change. */
export function toggle(open: Record<string, boolean>, key: string): Record<string, boolean> {
  const next = { ...open };
  if (next[key]) {
    delete next[key];
  } else {
    next[key] = true;
  }
  return next;
}

/**
 * A record kept in the webview's storage under *key*, or {} when there is none.
 *
 * Storage can be absent or throw (a webview with site data blocked), and what it holds
 * was written by an older build, so anything unreadable is an empty record.
 */
export function readStoredRecord<T>(key: string): Record<string, T> {
  try {
    const held = JSON.parse(window.localStorage.getItem(key) ?? "{}");
    return held && typeof held === "object" && !Array.isArray(held) ? held : {};
  } catch {
    return {};
  }
}

/** Keep *record* under *key*; a storage that refuses keeps it for this window only. */
export function storeRecord<T>(key: string, record: Record<string, T>): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(record));
  } catch {
    /* Held in memory still. */
  }
}

/**
 * Whether a section has anything to read: a line or a row. A detail alone is the server
 * saying there is nothing here ("no optimisation session"), which is not worth a card.
 */
export function isFilled(section: PanelSection): boolean {
  return section.lines.length > 0 || (section.groups?.length ?? 0) > 0;
}

/** Every group id in a tree, for a caller that wants to open or close all of them. */
export function groupKeys(groups: PanelGroup[] | undefined): string[] {
  const out: string[] = [];
  for (const group of groups ?? []) {
    out.push(group.id);
    out.push(...groupKeys(group.groups));
  }
  return out;
}

/**
 * What a badge says before it is opened: where the run is — its partition, then its
 * machine — else the run itself.
 *
 * No cluster name: the badge sits on a row that already names it. A job still queued has
 * no node yet, so it reads as its partition and its id.
 */
export function badgeLabel(run: WatchedRun): string {
  const nodes = run.nodes ?? [];
  const where = nodes.length === 1 ? nodes[0]
    : nodes.length > 1 ? `${nodes.length} nodes`
    : run.job_id ? run.job_id
    : run.pid ? `pid ${run.pid}`
    : run.job_key;
  const partitions = run.partitions ?? [];
  if (partitions.length === 0) return where;
  const partition = partitions.length === 1 ? partitions[0] : `${partitions.length} partitions`;
  return `${partition} · ${where}`;
}

/** Whose run this is, as the badge's colour: the conversation's own, or a sub-agent's. */
export function originKind(run: WatchedRun): "main" | "subagent" {
  return run.origin?.kind === "subagent" ? "subagent" : "main";
}

/**
 * How many runs a section holds, by whose they are: what its title says once folded.
 *
 * A run is badged on its host row and again on its node kind's, so only the rows at the
 * top of the tree are counted, with the runs no row claimed.
 */
export function sectionRunCounts(
  section: PanelSection & { unplaced?: WatchedRun[] },
): { main: number; subagent: number } {
  const counts = { main: 0, subagent: 0 };
  const runs = [
    ...(section.groups ?? []).flatMap((group) => group.badges ?? []),
    ...(section.unplaced ?? []),
  ];
  for (const run of runs) counts[originKind(run)] += 1;
  return counts;
}

/** Who started this run, in words. */
export function originLabel(origin: RunOrigin | undefined): string {
  if (!origin || origin.kind !== "subagent") return "this conversation";
  return origin.label ? `sub-agent: ${origin.label}` : "a sub-agent";
}

/** How far along, when the run's own output said so. "" when it did not. */
export function progressLabel(run: WatchedRun): string {
  const parts: string[] = [];
  if (run.phase) parts.push(run.phase);
  if (typeof run.percent === "number") parts.push(`${Math.round(run.percent)}%`);
  return parts.join(" · ");
}

/** Elapsed time as a person would say it. "" when it is not known. */
export function elapsedLabel(run: WatchedRun): string {
  const secs = run.elapsed_s;
  if (typeof secs !== "number" || secs < 0) return "";
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

/** The run's kind as a heading: "slurm-batch" reads "Slurm batch". Read, not mapped. */
export function kindLabel(run: WatchedRun): string {
  const words = (run.kind || "run").replace(/[-_]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * The rows inside an opened badge, under its heading, progress and command: where the
 * run is, how long it has gone, and whose it is.
 *
 * Built here rather than in the markup so the order and the omissions are one decision
 * in one place: a field the status op did not report is a row that does not appear,
 * never a row reading "unknown".
 */
export function badgeDetailRows(run: WatchedRun): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  if (run.job_id) rows.push({ label: "job", value: run.job_id });
  else if (run.pid) rows.push({ label: "pid", value: String(run.pid) });
  if (run.scheduler) rows.push({ label: "scheduler", value: run.scheduler });
  const partitions = run.partitions ?? [];
  if (partitions.length > 0) {
    rows.push({
      label: partitions.length > 1 ? "partitions" : "partition",
      value: partitions.join(", "),
    });
  }
  const nodes = run.nodes ?? [];
  if (nodes.length > 0) {
    rows.push({ label: nodes.length > 1 ? "nodes" : "node", value: nodes.join(", ") });
  }
  const elapsed = elapsedLabel(run);
  if (elapsed) rows.push({ label: "running for", value: elapsed });
  rows.push({ label: "launched by", value: originLabel(run.origin) });
  return rows;
}
