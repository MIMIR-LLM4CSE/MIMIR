// The optimisation session on the code map: its steps across the top, and what one step
// (or every kept step since the baseline) changed, as the map marks it. What each step
// changed comes from the proxy's own snapshots; the map itself is today's code.

export interface StepEdge { from: string; to: string; line: number; kind: string; via: string; sure: string; breaks: string }
/** Where a run's time went, per workspace file, in fractions of the baseline's metric. */
export interface Profile {
  tool: string; samples?: number; case?: string;
  files?: Record<string, number>; outside?: number; unavailable?: string; note?: string;
}
export interface Step {
  n: number; run_id: string; axis: string; verdict: string;
  value: number | null; incumbent_value: number | null; delta: number | null;
  changed: string[]; added: StepEdge[]; removed: StepEdge[];
  profile?: Profile | null;
}
export interface Session {
  proxy: string; metric: string; goal: string;
  baseline: { run_id: string; value: number | null; profile?: Profile | null };
  steps: Step[];
}

/** The time the map shows at a step: whose profile, and the one it is compared with. */
export interface TimeView {
  files: Record<string, number>;
  before: Record<string, number> | null;
  from: string;
  tool: string;
  note?: string;
}

const usable = (p?: Profile | null): p is Profile & { files: Record<string, number> } =>
  !!p && !p.unavailable && !!p.files;

/** The profile of step *k* if it has one, else of the last kept step before it, else of
 *  the baseline; with "This step" and a kept step, also the profile it improved on. */
export function timeAt(session: Session, k: number, scope: Scope): TimeView | null {
  const chain: { label: string; p: Profile | null | undefined; kept: boolean; n: number }[] = [
    { label: "the baseline", p: session.baseline.profile, kept: true, n: 0 },
    ...session.steps.map((st) => ({ label: "#" + st.n, p: st.profile, kept: st.verdict === "accept", n: st.n })),
  ];
  const own = chain.find((c) => c.n === k && usable(c.p));
  const prior = chain.filter((c) => c.n < k && c.kept && usable(c.p)).pop();
  const shown = own ?? prior;
  if (!shown || !usable(shown.p)) return null;
  const before = scope === "step" && own && own.kept && k > 0 && prior && usable(prior.p) ? prior.p.files : null;
  return { files: shown.p.files, before, from: shown.label, tool: shown.p.tool, note: shown.p.note };
}
export type Scope = "step" | "all";

/** What the map marks for step *k*: by that step alone, or by every kept step up to it. */
export interface Overlay {
  added: Set<string>;
  removed: StepEdge[];
  touchedNow: Set<string>;
  touched: Set<string>;
  steps: Step[];
}

export const edgeKey = (e: { from: string; to: string; kind: string }) => `${e.from}|${e.to}|${e.kind}`;
const kept = (s: Step) => s.verdict === "accept";

export function overlay(session: Session, k: number, scope: Scope): Overlay {
  const current = session.steps.find((s) => s.n === k);
  const steps = scope === "step" ? (current ? [current] : [])
    : session.steps.filter((s) => s.n <= k && (kept(s) || s.n === k));
  const added = new Set<string>();
  const removed = new Map<string, StepEdge>();
  for (const s of steps) {
    for (const e of s.added) { added.add(edgeKey(e)); removed.delete(edgeKey(e)); }
    for (const e of s.removed) { if (!added.delete(edgeKey(e))) removed.set(edgeKey(e), e); }
  }
  return {
    added, removed: [...removed.values()], steps,
    touchedNow: new Set(current ? current.changed : []),
    touched: new Set(steps.flatMap((s) => s.changed)),
  };
}

function esc(s: unknown): string {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
}
export function signed(d: number | null): string {
  if (d === null || !isFinite(d)) return "";
  return (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d * 100).toFixed(Math.abs(d) < 0.1 ? 1 : 0) + "%";
}
function value(v: number | null): string {
  return v === null ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3);
}
/** Whether a step moved the metric the way the session wants. */
function better(session: Session, d: number | null): boolean {
  return d !== null && (session.goal === "max" ? d > 0 : d < 0);
}

/** The best value after each stop (baseline first): what the session kept as it went. */
function bestSoFar(session: Session): (number | null)[] {
  const out: (number | null)[] = [session.baseline.value];
  let best = session.baseline.value;
  for (const s of session.steps) {
    if (kept(s) && s.value !== null) best = s.value;
    out.push(best);
  }
  return out;
}

/** The strip: the best value as the session went, and one stop per step. */
export function strip(session: Session, k: number, scope: Scope): string {
  const n = session.steps.length + 1;
  const best = bestSoFar(session);
  const nums = best.filter((x): x is number => x !== null).concat(
    session.steps.map((s) => s.value).filter((x): x is number => x !== null));
  const lo = Math.min(...nums), hi = Math.max(...nums), span = hi - lo || 1;
  const H = 40;
  const sx = (i: number) => ((i + 0.5) / n) * 100;
  const sy = (v: number) => 6 + ((hi - v) / span) * (H - 12);
  const pts = best.map((v, i) => (v === null ? null : sx(i) + "," + sy(v))).filter(Boolean);
  const dots = session.steps.map((s, i) => s.value === null ? ""
    : '<i class="pt' + (s.n === k ? " is-cur" : "") + (kept(s) ? "" : " is-rej") + '" style="left:'
      + sx(i + 1) + "%;top:" + sy(s.value) + 'px" title="#' + s.n + " " + esc(session.metric) + " " + value(s.value) + '"></i>').join("");
  const first = best[0] === null ? "" : '<i class="pt' + (k === 0 ? " is-cur" : "") + '" style="left:' + sx(0)
    + "%;top:" + sy(best[0]) + 'px" title="baseline ' + value(best[0]) + '"></i>';
  const stops = ['<button class="stop" data-step="0" aria-current="' + (k === 0) + '"><span class="n">Baseline</span>'
    + '<span class="ax">' + esc(session.metric) + " " + value(session.baseline.value) + "</span>"
    + '<span class="t">' + esc(session.baseline.run_id) + "</span></button>"]
    .concat(session.steps.map((s) => '<button class="stop' + (s.n > k ? " is-after" : "") + '" data-step="' + s.n
      + '" aria-current="' + (s.n === k) + '"><span class="n">#' + s.n + ' <span class="v '
      + (kept(s) ? "ok" : "no") + '">' + (kept(s) ? "✓ kept" : "✕ " + esc(s.verdict.replace("reject", "ruled out"))) + "</span></span>"
      + '<span class="ax" title="' + esc(s.axis) + '">' + esc(s.axis || "(no axis given)") + "</span>"
      + '<span class="t">' + (s.delta === null ? "–" : signed(s.delta)) + " · " + s.changed.length + " files</span></button>"));
  const keptCount = session.steps.filter(kept).length;
  const last = best[best.length - 1], base = session.baseline.value;
  const total = last !== null && base ? (last - base) / Math.abs(base) : null;
  return '<div class="session-head"><div class="title">Optimisation session</div><span class="sum">'
    + session.steps.length + " steps tried · <b>" + keptCount + "</b> kept · " + esc(session.metric) + " <b>"
    + signed(total) + "</b> since baseline</span>"
    + '<div class="seg push" role="group" aria-label="Which changes to show">'
    + '<button data-scope="step" aria-pressed="' + (scope === "step") + '">This step</button>'
    + '<button data-scope="all" aria-pressed="' + (scope === "all") + '">Since baseline</button></div></div>'
    + '<div class="steps-scroll"><div class="steps" style="grid-template-columns:repeat(' + n + ',minmax(118px,1fr))">'
    + '<svg class="spark" viewBox="0 0 100 ' + H + '" preserveAspectRatio="none" aria-hidden="true">'
    + '<path class="line" d="M' + pts.join(" L") + '"/></svg><div class="pts">' + first + dots + "</div>"
    + stops.join("") + "</div></div>";
}

/** The card of step *k* in the side pane, for when nothing on the map is selected. */
const PROFILE_PROMPT = "Profile the optimisation session's code as it stands now with "
  + "proxy_eval(op='profile', confirm=True), then tell me where its time goes.";

/** What a step's card says of its time: its costliest files, why it has no profile, or
 *  (the latest step only: the code as it is now) a button that asks for one. */
function timeBlock(session: Session, s: Step, fileRow: (id: string, right: string) => string): string {
  const p = s.profile;
  if (usable(p)) {
    const top = Object.entries(p.files).slice(0, 5);
    return '<div class="cap">Where its time goes (' + esc(p.tool) + ")</div>"
      + '<div class="list">' + top.map(([f, v]) => fileRow(f, pctOf(v))).join("") + "</div>"
      + (p.outside ? '<div class="empty">' + pctOf(p.outside) + " outside the workspace</div>" : "")
      + (p.note ? '<div class="empty">' + esc(p.note) + "</div>" : "");
  }
  if (p && p.unavailable) return '<div class="cap">Not profiled</div><div class="empty">' + esc(p.unavailable) + "</div>";
  if (s.n === session.steps.length) {
    return '<button class="open" data-prompt="' + esc(PROFILE_PROMPT) + '">Profile the code as it is now</button>';
  }
  return '<div class="empty">Not profiled. Only the code as it is now can be profiled.</div>';
}

/** A share of the baseline's time, as the map writes it. */
export function pctOf(v: number): string {
  return v >= 0.0995 ? (v * 100).toFixed(0) + "%" : v >= 0.001 ? (v * 100).toFixed(1) + "%" : v > 0 ? "<0.1%" : "–";
}

export function stepCard(session: Session, k: number, fileRow: (id: string, right: string) => string): string {
  const s = session.steps.find((x) => x.n === k);
  if (!s) return "";
  const line = (l: string, v: string, cls = "") =>
    '<div class="line"><span>' + esc(l) + '</span><span class="' + cls + '">' + v + "</span></div>";
  return '<div class="section step"><div class="title">Step #' + s.n + "</div>"
    + '<div class="path">' + esc(s.axis || "(no axis given)") + "</div>"
    + line("verdict", kept(s) ? "✓ kept" : "✕ " + esc(s.verdict.replace("reject", "ruled out")), kept(s) ? "is-ok" : "is-danger")
    + line(session.metric, value(s.value) + " against " + value(s.incumbent_value)
      + (s.delta === null ? "" : " (" + signed(s.delta) + ")"), better(session, s.delta) ? "is-ok" : "")
    + line("dependencies", '<span class="is-ok">+' + s.added.length + '</span> / <span class="is-danger">−' + s.removed.length + "</span>")
    + '<div class="cap">Files it changed (' + s.changed.length + ")</div>"
    + '<div class="list">' + (s.changed.map((f) => fileRow(f, "")).join("") || '<span class="empty">none</span>') + "</div>"
    + timeBlock(session, s, fileRow) + "</div>";
}
