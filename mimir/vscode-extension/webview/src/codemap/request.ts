// R&D requests from the code map: the form's draft, the message it puts to the agent, and
// what the map draws of a request the server recorded (proxy_request). The tab writes no
// request itself: the agent opens it from the message, through a tool and its approval.

export type Mode = "study" | "try";

export const KINDS: [string, string][] = [
  ["method", "Numerical method"], ["algorithm", "Algorithm"], ["external", "External code"],
  ["parallel", "Parallelism / HPC"], ["layout", "Data layout"], ["precision", "Precision"],
  ["refactor", "Architecture / refactor"], ["other", "Other"],
];

/** What a request is about: a file (a function in it, at its line), a whole module or
 *  component, or the whole code. */
export interface Target { file?: string; function?: string; line?: number; module?: string; component?: string; all?: boolean }

/** The form as it is being filled in, kept in the tab's state. */
export interface Draft {
  token: string; mode: Mode; kind: string; targets: Target[];
  idea: string; variants: string; source: string; constraints: string;
}

/** A run that tried a request, against the value the request started from. */
export interface RequestStep { run_id: string; variant: number | null; axis: string; verdict: string; value: number | null; delta: number | null }

/** A request as the server answers it, with the runs that tried it. */
export interface Request {
  id: string; token?: string; ts?: string; mode: Mode; kind: string; targets: Target[];
  idea: string; variants: string[]; source?: string; constraints?: string;
  summary?: string; conclusion?: string; status: string;
  /** The conversation it was put in: its study and outcome are the agent's answers there. */
  session?: string;
  proxy?: string; metric?: string; start_value?: number | null; steps: RequestStep[];
}

/** A request sent from this tab that the server has not answered yet. */
export interface Sent { token: string; idea: string; mode: Mode; targets: Target[] }

export function newToken(): string {
  return Math.random().toString(36).slice(2, 8) + Date.now().toString(36).slice(-3);
}

export function newDraft(targets: Target[] = [], token = newToken()): Draft {
  return { token, mode: "study", kind: "method", targets, idea: "", variants: "", source: "", constraints: "" };
}

export const sameTarget = (a: Target, b: Target) =>
  (a.file ?? "") === (b.file ?? "") && (a.function ?? "") === (b.function ?? "") && (a.module ?? "") === (b.module ?? "")
  && (a.component ?? "") === (b.component ?? "") && !!a.all === !!b.all;

/** *t* added to *targets*, unless it is there already. */
export function withTarget(targets: Target[], t: Target): Target[] {
  return targets.some((x) => sameTarget(x, t)) ? targets : targets.concat([t]);
}

/** The map's components by tier, highest first as the bands stack. */
export function tiersOf(components: { id: string; tier: number }[]): [number, string[]][] {
  const by = new Map<number, string[]>();
  for (const c of components) by.set(c.tier, (by.get(c.tier) ?? []).concat([c.id]));
  return [...by].sort((a, b) => b[0] - a[0]);
}

/** Whether every one of *components* is among *targets*. */
export function holdsAll(targets: Target[], components: string[]): boolean {
  return components.every((c) => targets.some((t) => sameTarget(t, { component: c })));
}

/** *targets* with the whole of *components* added, or taken off when all are there. */
export function toggleComponents(targets: Target[], components: string[]): Target[] {
  if (holdsAll(targets, components)) return targets.filter((t) => !components.includes(t.component ?? ""));
  return components.reduce((ts, c) => withTarget(ts, { component: c }), targets);
}

/** How a target reads: `relax in solver.py`, a file's path, `module pkg/mesh`, `component
 *  pkg`, or `the whole code`. */
export function targetLabel(t: Target): string {
  if (t.all) return "the whole code";
  if (t.function) return t.function + " in " + (t.file ?? "").split("/").pop();
  if (t.file) return t.file;
  return t.module ? "module " + t.module : "component " + (t.component ?? "");
}

export function kindLabel(kind: string): string {
  return KINDS.find(([k]) => k === kind)?.[1] ?? kind;
}

/** The variants written in the form, one per line. */
export function variantsOf(d: Draft): string[] {
  return d.variants.split("\n").map((v) => v.trim()).filter(Boolean);
}

/** Whether the draft can be sent: an idea and something to apply it to. */
export function ready(d: Draft): boolean {
  return !!d.idea.trim() && d.targets.length > 0;
}

/** The message the form puts to the agent, as the user's own: what to do, on what, and
 *  the call that records it first. *share* gives a target's share of the time, when the
 *  map has a profile. */
export function requestPrompt(d: Draft, share: (t: Target) => string | null = () => null, findings: string[] = []): string {
  const variants = variantsOf(d);
  const kind = kindLabel(d.kind).toLowerCase();
  const lines = [
    "Request from the code map (token " + d.token + "): " + (d.mode === "study" ? "study" : "try") + " — " + kind + ".",
    "",
    "Targets:",
    ...d.targets.map((t) => {
      const where = t.function ? t.file + ": " + t.function + (t.line ? " (line " + t.line + ")" : "") : targetLabel(t);
      const s = share(t);
      return "- " + where + (s ? ", " + s + " of the baseline's time" : "");
    }),
    "",
    "Idea: " + d.idea.trim(),
  ];
  if (variants.length) {
    lines.push("", "Variants, each measured from the same start:", ...variants.map((v, i) => i + 1 + ". " + v));
  }
  if (d.kind === "refactor" && findings.length) {
    lines.push("", "What the map finds there:", ...findings.map((f) => "- " + f));
  }
  if (d.kind === "external" && d.source.trim()) {
    lines.push("", "External code: " + d.source.trim() + " (anything to install or clone, ask me first)");
  }
  if (d.constraints.trim()) lines.push("", "Constraints: " + d.constraints.trim());
  const args = [
    "op='open'", "token=" + JSON.stringify(d.token), "mode=" + JSON.stringify(d.mode), "kind=" + JSON.stringify(d.kind),
    "targets=" + JSON.stringify(d.targets), "idea=" + JSON.stringify(d.idea.trim()),
    ...(variants.length ? ["variants=" + JSON.stringify(variants)] : []),
    ...(d.kind === "external" && d.source.trim() ? ["source=" + JSON.stringify(d.source.trim())] : []),
    ...(d.constraints.trim() ? ["constraints=" + JSON.stringify(d.constraints.trim())] : []),
  ];
  lines.push("", "Record it first with proxy_request(" + args.join(", ") + "), then follow the map-request skill."
    + (d.mode === "study" ? " Study only: change no file." : ""));
  return lines.join("\n");
}

/** What asks the agent to try a request it studied. */
export function tryPrompt(r: Request): string {
  return "Try request " + r.id + " as proposed, following the map-request skill: each variant from its start "
    + "(proxy_eval(op='reset_to_request', request_id='" + r.id + "')), its runs tagged [" + r.id + ".k].";
}

/** What asks the agent to drop a request. */
export function dropPrompt(r: Request): string {
  return "Drop request " + r.id + ": proxy_request(op='update', request_id='" + r.id + "', status='dropped', "
    + "summary='dropped from the code map'). Leave the code as the session's best.";
}

export const STATUS_LABEL: Record<string, string> = {
  sent: "sent", open: "open", studying: "studying", proposed: "proposed", trying: "trying", done: "done", dropped: "dropped",
};

/** Whether the agent is still at it: the tab keeps asking the server while one is. */
export function live(status: string): boolean {
  return status === "sent" || status === "open" || status === "studying" || status === "trying";
}

/** The request and variant an axis names (`[R3.2] …`), or null. */
export function axisTag(axis: string): { id: string; variant: number | null } | null {
  const m = /^\s*\[R(\d+)(?:\.(\d+))?\]/.exec(axis || "");
  return m ? { id: "R" + m[1], variant: m[2] ? Number(m[2]) : null } : null;
}

/** One row per variant the request named, with the run that tried it when there is one;
 *  then the runs that named no variant of the list. */
export function variantRows(r: Request): { label: string; step: RequestStep | null }[] {
  const rows = r.variants.map((v, i) => ({ label: i + 1 + ". " + v, step: r.steps.filter((s) => s.variant === i + 1).pop() ?? null }));
  const named = new Set(rows.map((x) => x.step).filter(Boolean));
  for (const s of r.steps) {
    if (!named.has(s)) rows.push({ label: s.axis.replace(/^\s*\[R\d+(?:\.\d+)?\]\s*/, "") || "(run)", step: s });
  }
  return rows;
}

/** Whether a request is about the module *module* of *component*, whose files are *files*.
 *  A request on the whole code is drawn on no module: the header says it. */
export function touches(r: { targets: Target[] }, module: string, component: string, files: Set<string>): boolean {
  return r.targets.some((t) => (t.module && t.module === module) || (t.component && t.component === component)
    || (t.file && files.has(t.file)));
}
