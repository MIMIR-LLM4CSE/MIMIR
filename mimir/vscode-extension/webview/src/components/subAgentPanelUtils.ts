/**
 * Reading a sub-agent's card.
 *
 * A sub-agent leaves a file behind when it ends, and the panel is the only place that
 * file is ever read. The card is written by another process, possibly mid-run, so
 * everything here treats a missing or half-written field as normal rather than as an
 * error: a child that died before finishing is exactly what the user needs to see.
 */
import type { SubAgent } from "../types";

/** What the row says the child is doing, in the user's words rather than the file's. */
export function stateLabel(sub: SubAgent): string {
  switch (sub.state) {
    case "running":
      return "running";
    case "finished":
      return sub.completed ? "done" : "incomplete";
    // Its caller ran out of patience before it ran out of work. Whatever it went on
    // to produce was never delivered, so the row must not read like a result.
    case "abandoned":
      return "abandoned";
    case "failed":
      return "failed";
    default:
      return "unknown";
  }
}

/** The line under the title: what it was given, and how long it had. */
export function grantSummary(sub: SubAgent): string {
  const tools = sub.tools ?? [];
  const scope = tools.length ? tools.join(", ") : "read-only exploration";
  const budget = sub.budget_secs ? `, ${Math.round(sub.budget_secs / 60)} min` : "";
  return `${scope}${budget}`;
}

/**
 * Where a child's work is, in one line.
 *
 * The copy is kept now, so saying that it was is no longer news — where it is, is. A
 * child that changed nothing has no branch (an empty one is dropped rather than left
 * pointing at HEAD), and a card written before copies were kept says `kept: false`.
 */
export function workspaceLine(sub: SubAgent): string {
  const ws = sub.workspace;
  if (!ws) return "";
  const parts: string[] = [];
  if (ws.branch) parts.push(`branch ${ws.branch}`);
  if (ws.path) parts.push(ws.kept === false ? "copy removed" : `copy at ${ws.path}`);
  return parts.join(" · ");
}

/** First line of the task, for the row title. A task is a paragraph; a row is a line. */
export function taskTitle(sub: SubAgent): string {
  const task = (sub.task ?? "").trim();
  if (!task) return sub.id;
  const firstLine = task.split("\n")[0];
  return firstLine.length > 90 ? `${firstLine.slice(0, 89)}…` : firstLine;
}

/**
 * The checklist a sub-agent kept, as {text, done} rows.
 *
 * The todo file is Markdown owned by the todo server; only its checkbox lines are read
 * here, so a change to the rest of that format shows up as a shorter list, never as a
 * broken panel.
 */
export function parseTodo(markdown: string | undefined): { text: string; done: boolean }[] {
  if (!markdown) return [];
  const rows: { text: string; done: boolean }[] = [];
  for (const line of markdown.split("\n")) {
    const match = /^\s*[-*]\s*\[( |x|X)\]\s*(.+?)\s*$/.exec(line);
    if (match) rows.push({ done: match[1].toLowerCase() === "x", text: match[2] });
  }
  return rows;
}

/** One step of a sub-agent, as the panel shows it. */
export interface ActivityRow {
  id: string;
  name: string;
  label: string;
  detail: string;
  status: "running" | "ok" | "error" | "note";
  ms?: number;
}

/**
 * The activity log, paired into rows.
 *
 * The log holds one line per event — a call (`tc`) and later its result (`tr`) —
 * because it is appended to while the child works and a line is never rewritten. The
 * panel wants the other shape: one row per step, still running until its result
 * lands. A result whose call was trimmed off the head of the log is dropped rather
 * than shown headless.
 *
 * A status line (`st`) is neither: it is the loop saying why it is about to ask the
 * model again without calling anything — an empty turn retried, a nudge, a refreshed
 * checklist. It stands as its own row, in order, so the minutes between two tool calls
 * have something in them instead of reading as a hang.
 */
export function activityRows(activity: unknown): ActivityRow[] {
  if (!Array.isArray(activity)) return [];
  const rows: ActivityRow[] = [];
  const byId = new Map<string, ActivityRow>();
  for (const raw of activity) {
    const ev = (raw ?? {}) as Record<string, unknown>;
    const id = String(ev.i ?? "");
    if (ev.t === "tc") {
      const row: ActivityRow = {
        id,
        name: String(ev.n ?? ""),
        label: String(ev.l ?? ""),
        detail: String(ev.d ?? ""),
        status: "running",
      };
      rows.push(row);
      byId.set(id, row);
    } else if (ev.t === "st") {
      rows.push({
        id: "", name: "", label: String(ev.s ?? ""), detail: "", status: "note",
      });
    } else if (ev.t === "tr") {
      const row = byId.get(id);
      if (!row) continue;
      row.status = ev.ok ? "ok" : "error";
      if (ev.s) row.detail = String(ev.s);
      if (typeof ev.ms === "number") row.ms = ev.ms;
    }
  }
  return rows;
}

/** Whether anything in this list is still working — what a live view polls on. */
export function anyRunning(subAgents: { state?: string }[]): boolean {
  return subAgents.some((s) => s.state === "running");
}
