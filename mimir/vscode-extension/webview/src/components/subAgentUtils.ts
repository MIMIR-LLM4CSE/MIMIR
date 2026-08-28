/**
 * Placing a sub-agent's tool rows among the main agent's.
 *
 * A delegated run is one tool call that lasts minutes, and several of them run at
 * once. Its steps are shown as ordinary tool rows so they read like the agent's own
 * work; what these helpers add is the little that keeps a flat list legible when the
 * work has two sources — rows grouped under the call that spawned them, and a badge
 * saying which sub-agent a row came from.
 */
import type { ToolActivity } from "../types";

/** Ids of the calls that spawned children, in the order their rows appear. */
function delegatingIds(rows: ToolActivity[]): string[] {
  const order = rows.filter((r) => rows.some((c) => c.parentId === r.id)).map((r) => r.id);
  // A parent frozen out of this list still owns its children; give it a slot too.
  for (const r of rows) {
    if (r.parentId && !order.includes(r.parentId)) order.push(r.parentId);
  }
  return order;
}

/**
 * Badge for a row produced by a sub-agent, e.g. `explore #2`.
 *
 * The role is read out of the parent's own label — the label the tool declares, so
 * nothing here has to know a tool name. Anything unreadable degrades to "sub-agent",
 * which is still true.
 */
export function originLabel(parentLabel: string | undefined, ordinal: number): string {
  const inParens = /\(([^)]{1,24})\)/.exec(parentLabel ?? "");
  const role = (inParens?.[1] ?? "").trim().toLowerCase() || "sub-agent";
  return `${role} #${Math.max(1, ordinal)}`;
}

/**
 * Re-derive every child row's badge from the current list.
 *
 * Ranks come from where the delegating rows sit, so they read in the order the user
 * sees them — and a badge assigned when a sub-agent's first step happened to arrive
 * first would otherwise contradict that order for the rest of the turn. Rebuilding
 * the lot on each insert keeps the numbering honest and costs nothing at this size.
 */
export function relabelOrigins(rows: ToolActivity[]): ToolActivity[] {
  const order = delegatingIds(rows);
  const labels = new Map(rows.map((r) => [r.id, r.label]));
  return rows.map((r) =>
    r.parentId
      ? { ...r, origin: originLabel(labels.get(r.parentId), order.indexOf(r.parentId) + 1) }
      : r
  );
}

/**
 * Insert *row* after the last row of its family (its parent, then that parent's
 * children), so siblings stay together in a list that is otherwise flat.
 * A parent that is no longer on screen puts its child at the end rather than
 * dropping it.
 */
export function insertAfterFamily(
  rows: ToolActivity[],
  parentId: string,
  row: ToolActivity,
): ToolActivity[] {
  let at = -1;
  rows.forEach((r, i) => {
    if (r.id === parentId || r.parentId === parentId) at = i;
  });
  if (at < 0) return [...rows, row];
  return [...rows.slice(0, at + 1), row, ...rows.slice(at + 1)];
}

/**
 * Stamp a delegating call's still-running children as done.
 *
 * The parent's result is the last word on its whole run: any child row still ticking
 * at that point never got a result of its own (shed event, crash, hard cap), and a
 * timer counting up forever under a finished call is worse than a silent one.
 */
export function settleFamily(
  rows: ToolActivity[],
  parentId: string,
  now: number,
): ToolActivity[] {
  return rows.map((r) =>
    r.parentId === parentId && r.status === "running"
      ? { ...r, status: "ok" as const, durationMs: r.durationMs ?? Math.max(0, now - r.startedAt) }
      : r
  );
}

/** Tail text for a delegating row: what its children add up to. */
export function subAgentTail(rows: ToolActivity[], parentId: string): string {
  const children = rows.filter((r) => r.parentId === parentId);
  if (children.length === 0) return "";
  return children.length === 1 ? "1 tool" : `${children.length} tools`;
}
