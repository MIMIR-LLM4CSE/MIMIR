// A merge of proxies on the map: each proxy's gain alone on the left, the files two of them
// share and their conflicts as arcs between them, and the whole on the right with the gain
// it was expected to bring and the one measured. Without a measurement (proxies ticked, not
// merged yet) it is the forecast: what merging them promises and what stands in the way,
// and a Merge button that asks the agent for it.

import { esc, pct } from "./format";
import { mergePrompt } from "./picker";

/** A relative is a value against its baseline, lower is better whatever the goal: 0.8 is
 *  a fifth better. */
export interface MergeSource { proxy: string; branch?: string; relative: number; files: string[] }
export interface MergePair { a: string; b: string; shared: string[]; conflicts: string[] }
export interface MergeStep {
  proxy: string; kept: boolean; why?: string; files?: string[];
  /** Its gain on top of what was kept before it, and that against its gain alone. */
  marginal?: number | null; interaction?: number | null;
}
export interface MergeState {
  phase: string; sources: MergeSource[]; pairs: MergePair[]; expected: number;
  all?: { applied: string[]; conflicts: { proxy: string; files: string[] }[]; expected: number;
    relative?: number | null; holds?: boolean } | null;
  steps?: MergeStep[];
  /** The code kept: from the whole, the one-at-a-time chain or nothing; why each other was set aside. */
  final?: { from: string; relative: number | null; kept: string[]; set_aside: { proxy: string; why: string }[] } | null;
}
export interface Forecast {
  sources: MergeSource[]; pairs: MergePair[]; expected: number;
  refused?: { proxy: string; why: string }[]; refusal?: string;
}

/** What became of one source: kept, set aside and why, or still to be measured. */
export interface Outcome { state: string; marginal?: number | null }

/** Where *proxy* stands in the merge *m*: as the server decided once it is done, else as
 *  far as it has gone. */
export function outcome(m: MergeState | null, proxy: string): Outcome {
  if (!m) return { state: "forecast" };
  const step = [...(m.steps ?? [])].reverse().find((s) => s.proxy === proxy);
  const marginal = step?.marginal;
  if (m.final) {
    if (m.final.kept.includes(proxy)) return { state: "kept", marginal };
    return { state: m.final.set_aside.find((x) => x.proxy === proxy)?.why ?? "", marginal };
  }
  if (step) return { state: step.kept ? "kept" : step.why ?? "", marginal };
  if (m.phase === "incremental") return { state: "pending" };
  if ((m.all?.conflicts ?? []).some((c) => c.proxy === proxy)) return { state: "conflict" };
  return { state: m.all?.applied.includes(proxy) ? "applied" : "pending" };
}

/** A gain, from a relative: "20%" for 0.8. */
export function gainOf(rel: number | null | undefined): string {
  if (rel === null || rel === undefined || !isFinite(rel)) return "–";
  const g = Math.round((1 - rel) * 1e6) / 1e6;
  return (g < 0 ? "−" : "") + pct(g);
}

// Half a percent: closer than that, a gain in the merge is its gain alone, and a merge's
// measured gain the one expected.
const SAME = 0.005;

const WHY: Record<string, string> = {
  kept: "kept", conflict: "set aside: conflict", interference: "set aside: slowed the whole",
  infeasible: "set aside: failed its requirements", no_gain: "set aside: the merge gained nothing", applied: "applied, being measured", pending: "to measure", forecast: "",
};

/** The line above the graph: expected, measured, kept, and where the merge stands. */
function summary(expected: number, m: MergeState | null, f: Forecast | null): string {
  if (f?.refusal) return esc(f.refusal);
  const parts = ["expected together <b>" + gainOf(expected) + "</b> better"];
  if (m?.all?.relative !== undefined && m.all.relative !== null) parts.push("all at once <b>" + gainOf(m.all.relative) + "</b>");
  if (m?.final) {
    parts.push("kept <b>" + gainOf(m.final.relative) + "</b> with " + (m.final.kept.length ? esc(m.final.kept.join(", ")) : "nothing"));
  } else if (m) {
    parts.push(m.phase === "incremental" ? "now one at a time" : m.phase === "baseline" ? "measuring the copy untouched" : "measuring them all at once");
  }
  if (f?.refused?.length) parts.push("left out: " + f.refused.map((r) => esc(r.proxy) + " (" + esc(r.why) + ")").join(", "));
  return parts.join(" · ");
}

/** The forecast's Merge button: what it puts to the agent, the proxies the forecast keeps;
 *  disabled when they are fewer than two or the server refuses to merge them. */
function mergeButton(f: Forecast): string {
  const names = f.sources.map((s) => s.proxy);
  const ok = names.length > 1 && !f.refusal;
  return '<button class="open merge-go" data-prompt="' + esc(ok ? mergePrompt(names) : "") + '"' + (ok ? "" : " disabled")
    + ' title="' + (ok ? "Ask the agent to merge them in a copy of its own and measure the whole" : "Nothing these proxies can merge")
    + '">Merge' + (ok ? " " + names.length : "") + "</button>";
}

/** The graph, *width* pixels wide: a merge session's (it has a phase) or a forecast. */
export function mergeGraph(shown: MergeState | Forecast, width = 760): string {
  const m = "phase" in shown ? shown : null, f = "phase" in shown ? null : shown;
  const { sources, pairs, expected } = shown;
  const W = Math.max(420, Math.round(width)), ROW = 46, T = 8, ARC = 54;
  const H = Math.max(96, T * 2 + sources.length * ROW);
  const boxX = ARC + 8, boxW = Math.min(300, Math.round(W * 0.42)), mx = W - 170, mw = 160;
  const y = (i: number) => T + i * ROW + ROW / 2;
  const at = new Map(sources.map((s, i) => [s.proxy, i]));
  const f1 = (x: number) => x.toFixed(1);
  const arcs = pairs.filter((p) => p.shared.length && at.has(p.a) && at.has(p.b)).map((p) => {
    const ya = y(at.get(p.a)!), yb = y(at.get(p.b)!), depth = Math.min(ARC - 6, 12 + Math.abs(yb - ya) / 3);
    const cls = p.conflicts.length ? "arc conflict" : "arc shared";
    const tip = p.a + " and " + p.b + (p.conflicts.length ? " conflict on " + p.conflicts.join(", ") : " both change " + p.shared.join(", "));
    return '<path class="' + cls + '" d="M' + boxX + "," + f1(ya) + " C" + f1(boxX - depth) + "," + f1(ya) + " "
      + f1(boxX - depth) + "," + f1(yb) + " " + boxX + "," + f1(yb) + '"><title>' + esc(tip) + "</title></path>";
  }).join("");
  const my = H / 2;
  const rows = sources.map((s, i) => {
    const o = outcome(m, s.proxy), yy = y(i);
    const out = !["kept", "applied", "pending", "forecast"].includes(o.state);
    const marginal = o.marginal ?? null;
    const shift = marginal === null ? 0 : marginal - s.relative;
    const linkCls = "mlink " + o.state + (Math.abs(shift) > SAME ? (shift < 0 ? " syn" : " intf") : "");
    const status = [marginal === null ? "" : "in the merge " + gainOf(marginal), WHY[o.state]].filter(Boolean).join(" · ");
    const link = '<path class="' + linkCls + '" d="M' + (boxX + boxW) + "," + f1(yy) + " C" + f1((boxX + boxW + mx) / 2) + "," + f1(yy)
      + " " + f1((boxX + boxW + mx) / 2) + "," + f1(my) + " " + mx + "," + f1(my) + '"/>';
    return link + '<g class="src ' + (out ? "is-out" : "") + '"><title>' + esc(s.proxy + (s.branch ? " · " + s.branch : "")
      + " · " + s.files.join(", ")) + "</title>"
      + '<rect class="box" x="' + boxX + '" y="' + f1(yy - ROW / 2 + 3) + '" width="' + boxW + '" height="' + (ROW - 6) + '" rx="4"/>'
      + '<text class="sname" x="' + (boxX + 10) + '" y="' + f1(yy - (status ? 3 : -4)) + '">' + esc(s.proxy) + "</text>"
      + '<text class="sgain" x="' + (boxX + boxW - 10) + '" y="' + f1(yy - (status ? 3 : -4)) + '" text-anchor="end">alone <tspan class="v">'
      + esc(gainOf(s.relative)) + "</tspan></text>"
      + (status ? '<text class="sstate ' + o.state + '" x="' + (boxX + 10) + '" y="' + f1(yy + 12) + '">' + esc(status) + "</text>" : "") + "</g>";
  }).join("");
  const measured = m?.final?.relative ?? m?.all?.relative ?? null;
  const node = '<g class="whole"><rect class="box" x="' + mx + '" y="' + f1(my - 30) + '" width="' + mw + '" height="60" rx="6"/>'
    + '<text class="wname" x="' + (mx + 12) + '" y="' + f1(my - 11) + '">' + (m ? "Merged" : "If merged") + "</text>"
    + '<text class="wsub" x="' + (mx + 12) + '" y="' + f1(my + 6) + '">expected ' + gainOf(expected) + "</text>"
    + '<text class="wsub ' + (measured === null ? "" : measured <= expected + SAME ? "good" : "short") + '" x="' + (mx + 12) + '" y="' + f1(my + 21) + '">'
    + (measured === null ? (m ? "not measured yet" : "never measured") : "measured " + gainOf(measured)) + "</text></g>";
  const svg = '<svg class="mgraph" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Merge of '
    + esc(sources.map((s) => s.proxy).join(", ")) + '">' + arcs + rows + node + "</svg>";
  const legend = '<div class="mlegend"><span><i class="sw arc-shared"></i>both change a file</span>'
    + '<span><i class="sw arc-conflict"></i>conflict: cannot both be kept</span>'
    + (m ? '<span><i class="sw link-syn"></i>better in the merge than alone</span><span><i class="sw link-intf"></i>worse in the merge than alone</span>' : "")
    + "</div>";
  return '<div class="merge"><div class="session-head"><div class="title">' + (m ? "Merge" : "Merge forecast") + '</div><span class="sum">'
    + summary(expected, m, f) + "</span>" + (f ? mergeButton(f) : "") + "</div>" + '<div class="chart-wrap">' + svg + "</div>" + legend + "</div>";
}
