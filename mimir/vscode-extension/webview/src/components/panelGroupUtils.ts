import type { PanelGroup, RunOrigin, WatchedRun } from "../types";

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
 * What a badge says before it is opened: the machine, when it names one, else the run.
 *
 * The node first because the badge sits on a row that already names the cluster, and a
 * second copy of the cluster's name is not what the reader is missing.
 */
export function badgeLabel(run: WatchedRun): string {
  const nodes = run.nodes ?? [];
  if (nodes.length === 1) return nodes[0];
  if (nodes.length > 1) return `${nodes.length} nodes`;
  if (run.job_id) return run.job_id;
  if (run.pid) return `pid ${run.pid}`;
  return run.job_key;
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

/**
 * The rows inside an opened badge: what this run is, and whose.
 *
 * Built here rather than in the markup so the order and the omissions are one decision
 * in one place: a field the status op did not report is a row that does not appear,
 * never a row reading "unknown".
 */
export function badgeDetailRows(run: WatchedRun): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  if (run.job_id) rows.push({ label: "job", value: run.job_id });
  else if (run.pid) rows.push({ label: "pid", value: String(run.pid) });
  if (run.command) rows.push({ label: "command", value: run.command });
  const nodes = run.nodes ?? [];
  if (nodes.length > 1) rows.push({ label: "nodes", value: nodes.join(", ") });
  if (run.state) rows.push({ label: "state", value: run.state });
  const progress = progressLabel(run);
  if (progress) rows.push({ label: "progress", value: progress });
  const elapsed = elapsedLabel(run);
  if (elapsed) rows.push({ label: "running for", value: elapsed });
  rows.push({ label: "launched by", value: originLabel(run.origin) });
  return rows;
}
