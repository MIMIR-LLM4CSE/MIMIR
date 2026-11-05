// A merge of proxies on the map: each proxy's gain alone on the left, the files two of
// them share and their conflicts as arcs between them, and the whole on the right. While
// it runs: which proxy is merged, the conflict it stopped on or the question it waits on.
// Measured: each proxy's kernels in the merge against alone, and the whole. Without a
// merge (proxies ticked) it is the forecast, with a Merge button that asks the agent for it.

import { esc, pct } from "./format";
import { mergePrompt } from "./picker";

/** A relative is a value against its baseline, lower is better whatever the goal: 0.8 is
 *  a fifth better. */
export interface MergeSource { proxy: string; branch?: string; suite?: string; relative: number; files: string[] }
/** *together*: they ran at the same time in one tree, so each one's patch may hold the other's work. */
export interface MergePair { a: string; b: string; shared: string[]; conflicts: string[]; together?: boolean }
/** A proxy's kept patch picked into the merge: the files git stopped on, and how they were resolved. */
export interface Merged { proxy: string; commit?: string; conflicts: string[]; resolution: string }
/** One case of a source's suite: its gain alone, in the merge, and their ratio. */
export interface Kernel {
  proxy: string; case: string; alone: number | null; in_merge: number | null;
  interaction: number | null; flagged: boolean;
}
export interface MergeState {
  phase: string; sources: MergeSource[]; pairs: MergePair[];
  merged?: Merged[]; dropped?: string[];
  conflict?: { proxy: string; files: string[] } | null;
  question?: { proxy: string; question: string } | null;
  kernels?: Kernel[] | null;
  whole?: { value?: number | null; base_value?: number | null; relative?: number | null } | null;
}
export interface Forecast {
  sources: MergeSource[]; pairs: MergePair[];
  refused?: { proxy: string; why: string }[]; refusal?: string;
}

/** Where one source stands, and its kernel that did worst in the merge. */
export interface Outcome { state: string; kernel?: Kernel }

/** Where *proxy* stands in the merge *m*. */
export function outcome(m: MergeState | null, proxy: string): Outcome {
  if (!m) return { state: "forecast" };
  if ((m.dropped ?? []).includes(proxy)) return { state: "dropped" };
  const mine = (m.kernels ?? []).filter((k) => k.proxy === proxy);
  const worst = mine.reduce<Kernel | undefined>((w, k) => (w && (w.interaction ?? 0) >= (k.interaction ?? 0) ? w : k), undefined);
  if (m.conflict?.proxy === proxy) return { state: "conflict" };
  if (m.question?.proxy === proxy) return { state: "waiting" };
  if ((m.merged ?? []).some((x) => x.proxy === proxy)) {
    if (m.phase !== "done") return { state: "merged" };
    return { state: worst?.flagged ? "flagged" : "kept", kernel: worst };
  }
  return { state: "pending" };
}

/** A gain, from a relative: "20%" for 0.8. */
export function gainOf(rel: number | null | undefined): string {
  if (rel === null || rel === undefined || !isFinite(rel)) return "–";
  const g = Math.round((1 - rel) * 1e6) / 1e6;
  return (g < 0 ? "−" : "") + pct(g);
}

// Half a percent: closer than that, a kernel's gain in the merge is its gain alone.
const SAME = 0.005;

const WHY: Record<string, string> = {
  kept: "kept", flagged: "lost gain in the merge: your call", merged: "merged", pending: "to merge",
  conflict: "stopped on a conflict", waiting: "waiting for your answer", dropped: "taken back out", forecast: "",
};

/** Whether the conflict between *p*'s two proxies is resolved, and how. */
function resolution(p: MergePair, m: MergeState | null): Merged | undefined {
  return (m?.merged ?? []).find((x) => (x.proxy === p.a || x.proxy === p.b) && x.conflicts.some((f) => p.conflicts.includes(f)));
}

/** The line above the graph: where the merge stands and what it measured. */
function summary(shown: MergeState | Forecast, m: MergeState | null, f: Forecast | null): string {
  const open = shown.pairs.filter((p) => p.conflicts.length && !resolution(p, m)).length;
  const parts: string[] = [];
  if (f) {
    parts.push(shown.sources.length + " proxies");
    parts.push(open ? "<b>" + open + "</b> conflict" + (open > 1 ? "s" : "") + " to resolve" : "no conflict");
    if (f.refusal) parts.push(esc(f.refusal));
  } else if (m) {
    const n = (m.merged ?? []).length, all = m.sources.length;
    if (m.question) parts.push("waiting for your answer: " + esc(m.question.question));
    else if (m.conflict) parts.push("stopped on a conflict: " + esc(m.conflict.proxy) + " on " + esc(m.conflict.files.join(", ")));
    else if (m.phase === "baseline") parts.push("measuring the copy untouched");
    else if (m.phase === "merging") parts.push("merging " + n + "/" + all);
    else if (m.phase === "measure") parts.push("measuring the whole");
    if (m.whole?.relative !== undefined && m.whole?.relative !== null) parts.push("the whole <b>" + gainOf(m.whole.relative) + "</b> better");
    const flagged = [...new Set((m.kernels ?? []).filter((k) => k.flagged).map((k) => k.proxy))];
    if (flagged.length) parts.push("lost gain: " + esc(flagged.join(", ")));
    if ((m.dropped ?? []).length) parts.push("taken back out: " + esc((m.dropped ?? []).join(", ")));
  }
  const together = shown.pairs.filter((p) => p.together);
  if (together.length) parts.push('<span class="warn" title="Each one\'s patch and gain alone may hold the other\'s work">ran together: '
    + together.map((p) => esc(p.a) + " + " + esc(p.b)).join(", ") + "</span>");
    if (f?.refused?.length) parts.push("left out: " + f.refused.map((r) => esc(r.proxy) + " (" + esc(r.why) + ")").join(", "));
  return parts.join(" · ");
}

/** The forecast's Merge button: the proxies ticked, put to the agent, whatever the forecast
 *  says; it is the agent's to say what cannot be merged. */
function mergeButton(f: Forecast): string {
  const names = [...f.sources.map((s) => s.proxy), ...(f.refused ?? []).map((r) => r.proxy)];
  const ok = names.length > 1;
  return '<button class="open merge-go" data-prompt="' + esc(ok ? mergePrompt(names) : "") + '"' + (ok ? "" : " disabled")
    + ' title="Ask the agent to merge them with git in a copy of its own and measure the whole">Merge'
    + (ok ? " " + names.length : "") + "</button>";
}

/** The graph, *width* pixels wide: a merge session's (it has a phase) or a forecast. */
export function mergeGraph(shown: MergeState | Forecast, width = 760): string {
  const m = "phase" in shown ? shown : null, f = "phase" in shown ? null : shown;
  const { sources, pairs } = shown;
  const W = Math.max(420, Math.round(width)), ROW = 46, T = 8, ARC = 54;
  const H = Math.max(96, T * 2 + sources.length * ROW);
  const boxX = ARC + 8, boxW = Math.min(300, Math.round(W * 0.42)), mx = W - 170, mw = 160;
  const y = (i: number) => T + i * ROW + ROW / 2;
  const at = new Map(sources.map((s, i) => [s.proxy, i]));
  const f1 = (x: number) => x.toFixed(1);
  const drawn = new Set<string>();
  const arcs = pairs.filter((p) => p.shared.length && at.has(p.a) && at.has(p.b)).map((p) => {
    const ya = y(at.get(p.a)!), yb = y(at.get(p.b)!), depth = Math.min(ARC - 6, 12 + Math.abs(yb - ya) / 3);
    const done = p.conflicts.length ? resolution(p, m) : undefined;
    const kind = !p.conflicts.length ? "shared" : done ? "resolved" : "conflict";
    drawn.add(kind);
    const tip = (p.together ? "ran at the same time in one tree: each one's patch may hold the other's work. " : "")
      + p.a + " and " + p.b + (kind === "shared" ? " both change " + p.shared.join(", ")
      : kind === "resolved" ? " conflicted on " + p.conflicts.join(", ") + ", resolved: " + done!.resolution
      : " conflict on " + p.conflicts.join(", ") + ": to resolve when merged");
    return '<path class="arc ' + kind + '" d="M' + boxX + "," + f1(ya) + " C" + f1(boxX - depth) + "," + f1(ya) + " "
      + f1(boxX - depth) + "," + f1(yb) + " " + boxX + "," + f1(yb) + '"><title>' + esc(tip) + "</title></path>";
  }).join("");
  const my = H / 2;
  const rows = sources.map((s, i) => {
    const o = outcome(m, s.proxy), yy = y(i), k = o.kernel;
    const out = o.state === "dropped";
    const shift = k?.interaction != null ? k.interaction - 1 : 0;
    if (k) drawn.add(Math.abs(shift) > SAME ? (shift < 0 ? "syn" : "intf") : "");
    const linkCls = "mlink " + o.state + (k && Math.abs(shift) > SAME ? (shift < 0 ? " syn" : " intf") : "");
    const status = [k?.in_merge != null ? "in the merge " + gainOf(k.in_merge) : "", WHY[o.state]].filter(Boolean).join(" · ");
    const kernels = (m?.kernels ?? []).filter((x) => x.proxy === s.proxy)
      .map((x) => x.case + ": alone " + gainOf(x.alone) + ", merged " + gainOf(x.in_merge)).join("; ");
    const link = '<path class="' + linkCls + '" d="M' + (boxX + boxW) + "," + f1(yy) + " C" + f1((boxX + boxW + mx) / 2) + "," + f1(yy)
      + " " + f1((boxX + boxW + mx) / 2) + "," + f1(my) + " " + mx + "," + f1(my) + '"/>';
    return link + '<g class="src ' + (out ? "is-out" : "") + '"><title>' + esc(s.proxy + (s.branch ? " · " + s.branch : "")
      + (s.suite ? " · suite " + s.suite : "") + " · " + s.files.join(", ") + (kernels ? " · " + kernels : "")) + "</title>"
      + '<rect class="box" x="' + boxX + '" y="' + f1(yy - ROW / 2 + 3) + '" width="' + boxW + '" height="' + (ROW - 6) + '" rx="4"/>'
      + '<text class="sname" x="' + (boxX + 10) + '" y="' + f1(yy - (status ? 3 : -4)) + '">' + esc(s.proxy) + "</text>"
      + '<text class="sgain" x="' + (boxX + boxW - 10) + '" y="' + f1(yy - (status ? 3 : -4)) + '" text-anchor="end">alone <tspan class="v">'
      + esc(gainOf(s.relative)) + "</tspan></text>"
      + (status ? '<text class="sstate ' + o.state + '" x="' + (boxX + 10) + '" y="' + f1(yy + 12) + '">' + esc(status) + "</text>" : "") + "</g>";
  }).join("");
  const measured = m?.whole?.relative ?? null;
  const open = pairs.filter((p) => p.conflicts.length && !resolution(p, m)).length;
  const sub = measured !== null ? "measured " + gainOf(measured)
    : m ? (m.phase === "done" ? "not measured" : "not measured yet")
    : open ? open + " conflict" + (open > 1 ? "s" : "") + " to resolve" : "never measured";
  const node = '<g class="whole"><rect class="box" x="' + mx + '" y="' + f1(my - 30) + '" width="' + mw + '" height="60" rx="6"/>'
    + '<text class="wname" x="' + (mx + 12) + '" y="' + f1(my - 11) + '">' + (m ? "Merged" : "If merged") + "</text>"
    + '<text class="wsub" x="' + (mx + 12) + '" y="' + f1(my + 6) + '">' + (m ? "total time of every case" : "each on its own cases") + "</text>"
    + '<text class="wsub ' + (measured === null ? "" : measured < 1 - SAME ? "good" : "short") + '" x="' + (mx + 12) + '" y="' + f1(my + 21) + '">'
    + esc(sub) + "</text></g>";
  const svg = '<svg class="mgraph" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Merge of '
    + esc(sources.map((s) => s.proxy).join(", ")) + '">' + arcs + rows + node + "</svg>";
  // Only what the graph draws.
  const LEGEND: [string, string][] = [
    ["shared", '<span><i class="sw arc-shared"></i>both change a file</span>'],
    ["conflict", '<span><i class="sw arc-conflict"></i>conflict: to resolve when merged</span>'],
    ["resolved", '<span><i class="sw arc-resolved"></i>conflict resolved</span>'],
    ["syn", '<span><i class="sw link-syn"></i>better in the merge than alone</span>'],
    ["intf", '<span><i class="sw link-intf"></i>worse in the merge than alone</span>'],
  ];
  const legend = LEGEND.filter(([k]) => drawn.has(k)).map(([, html]) => html).join("");
  const of = sources.map((s) => s.proxy).join(" + ");
  return '<div class="merge"><div class="session-head"><div class="title">' + esc((m ? "Merge of " : "Merge forecast: ") + of) + '</div><span class="sum">'
    + summary(shown, m, f) + "</span>" + (f ? mergeButton(f) : "") + "</div>" + '<div class="chart-wrap">' + svg + "</div>"
    + (legend ? '<div class="mlegend">' + legend + "</div>" : "") + "</div>";
}

/** The files a merge has taken in so far: those of the proxies picked into it, and of those
 *  still to come. The map marks them while no run of the merge has settled a step. */
export function mergeFiles(m: MergeState): { now: Set<string>; all: Set<string> } {
  const out = new Set(m.dropped ?? []);
  const merged = new Set((m.merged ?? []).map((x) => x.proxy));
  const files = (keep: (p: string) => boolean) =>
    new Set(m.sources.filter((s) => !out.has(s.proxy) && keep(s.proxy)).flatMap((s) => s.files));
  return { now: files((p) => merged.has(p)), all: files(() => true) };
}
