// The optimisation session on the code map: its steps across the top, and what one step
// (or every kept step since the baseline) changed, as the map marks it. What each step
// changed comes from the proxy's own snapshots; the map itself is today's code. One proxy's
// session, or every proxy's at once: their steps in the order they settled, each value as a
// multiple of its own proxy's baseline (lower is better), since their metrics differ.

export interface StepEdge { from: string; to: string; line: number; kind: string; via: string; sure: string; breaks: string }
/** Where a run's time went, per workspace file, in fractions of the baseline's metric. */
/** A profile's GPU side: kernels, copies and idle time as shares of the baseline's time. */
export interface GpuProfile {
  tool: string; busy?: number; idle?: number; memcpy?: Record<string, number>;
  files?: Record<string, number>; kernels?: Record<string, { share: number; calls?: number; file?: string; line?: number }>;
  memcpy_mb?: Record<string, number>; peak_device_mb?: number;
  unavailable?: string; note?: string;
}
/** Where the memory live at the heap's peak was allocated: shares of that peak per file. */
export interface MemoryProfile {
  tool: string; peak_mb?: number; files?: Record<string, number>; unavailable?: string; note?: string;
}
/** A function's time: its own, and with what it calls; the line to open it at, when known. */
export interface FnTime { self: number; total: number; line?: number }
export interface Profile {
  tool: string; samples?: number; case?: string;
  files?: Record<string, number>; outside?: number; unavailable?: string; note?: string;
  /** Keyed `file::function`, costliest self time first. */
  functions?: Record<string, FnTime>;
  gpu?: GpuProfile;
  memory?: MemoryProfile;
}
export interface Step {
  n: number; run_id: string; axis: string; verdict: string;
  /** The proxy it ran on, and (every proxy at once) that proxy's metric. */
  proxy?: string; metric?: string;
  value: number | null; incumbent_value: number | null; delta: number | null;
  changed: string[]; added: StepEdge[]; removed: StepEdge[];
  profile?: Profile | null;
  peak_rss_mb?: number | null; incumbent_peak_rss_mb?: number | null;
  /** Every process's peak summed (an MPI run's ranks), and its incumbent's. */
  peak_rss_total_mb?: number | null; incumbent_peak_rss_total_mb?: number | null;
}
export interface Session {
  proxy: string; metric: string; goal: string;
  baseline: { run_id: string; value: number | null; profile?: Profile | null; peak_rss_mb?: number | null };
  steps: Step[];
  /** The proxies with a session, the active one, and whether these are all of theirs. */
  proxies?: string[]; active?: string; combined?: boolean;
}

/** The time the map shows at a step: whose profile, and the one it is compared with. */
export interface TimeView {
  files: Record<string, number>;
  before: Record<string, number> | null;
  /** Where it went per function, when the profile shown kept them; and the one before it. */
  functions: Record<string, FnTime> | null;
  functionsBefore: Record<string, FnTime> | null;
  /** The GPU's time per file, when the profile shown traced one; and the one before it. */
  gpuFiles: Record<string, number> | null;
  gpuBefore: Record<string, number> | null;
  gpu: GpuProfile | null;
  /** The allocations at the heap's peak, when the profile shown has them. */
  memory: MemoryProfile | null;
  from: string;
  tool: string;
  note?: string;
}

const usable = (p?: Profile | null): p is Profile & { files: Record<string, number> } =>
  !!p && !p.unavailable && !!p.files;
const traced = (g?: GpuProfile | null): g is GpuProfile => !!g && !g.unavailable;

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
  const compared = scope === "step" && own && own.kept && k > 0 && prior && usable(prior.p) ? prior.p : null;
  const gpu = traced(shown.p.gpu) ? shown.p.gpu : null;
  return {
    files: shown.p.files, before: compared ? compared.files : null,
    functions: shown.p.functions ?? null, functionsBefore: compared ? compared.functions ?? null : null,
    gpuFiles: gpu ? gpu.files ?? {} : null,
    gpuBefore: gpu && compared && traced(compared.gpu) ? compared.gpu.files ?? {} : null,
    gpu, memory: shown.p.memory && !shown.p.memory.unavailable ? shown.p.memory : null,
    from: shown.label, tool: shown.p.tool, note: shown.p.note,
  };
}
export type Scope = "step" | "all";

/** What the map marks for step *k*: by that step alone, or by every kept step up to it. */
export interface Overlay {
  added: Set<string>;
  removed: StepEdge[];
  touchedNow: Set<string>;
  touched: Set<string>;
  steps: Step[];
  /** The proxies with a session, the active one, and whether these are all of theirs. */
  proxies?: string[]; active?: string; combined?: boolean;
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

/** Each proxy's last kept value against its baseline, for every proxy at once. */
function perProxy(session: Session): string {
  const last = new Map<string, number>();
  for (const s of session.steps) if (s.proxy && !last.has(s.proxy)) last.set(s.proxy, 1);
  for (const s of session.steps) if (s.proxy && kept(s) && s.value !== null) last.set(s.proxy, s.value);
  return [...last].map(([p, v]) => esc(p) + " <b>" + signed(v - 1) + "</b>").join(" · ");
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

/** The strip: a chart of the session (the baseline, the best as it went, one dot per step)
 *  *width* pixels wide, and one chip per step to pick it by. */
export function strip(session: Session, k: number, scope: Scope, width = 760): string {
  const W = Math.max(360, Math.round(width)), H = 132, L = 54, R = 18, T = 12, B = 22;
  const best = bestSoFar(session);
  const base = session.baseline.value;
  const nums = best.filter((x): x is number => x !== null).concat(
    session.steps.map((s) => s.value).filter((x): x is number => x !== null));
  const lo0 = nums.length ? Math.min(...nums) : 0, hi0 = nums.length ? Math.max(...nums) : 1;
  const pad = (hi0 - lo0 || Math.abs(hi0) || 1) * 0.1;
  const lo = lo0 - pad, hi = hi0 + pad, span = hi - lo;
  const n = session.steps.length + 1;
  const sx = (i: number) => (n === 1 ? (L + W - R) / 2 : L + (i / (n - 1)) * (W - L - R));
  const sy = (v: number) => T + ((hi - v) / span) * (H - T - B);
  const f1 = (x: number) => x.toFixed(1);
  const ticks = [hi0, (lo0 + hi0) / 2, lo0].filter((v, i, all) => all.indexOf(v) === i);
  const grid = ticks.map((v) => '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + f1(sy(v)) + '" y2="' + f1(sy(v)) + '"/>'
    + '<text class="tick" x="' + (L - 8) + '" y="' + f1(sy(v) + 3.5) + '" text-anchor="end">' + value(v) + "</text>").join("");
  const baseLine = base === null ? "" : '<line class="base-line" x1="' + L + '" x2="' + (W - R) + '" y1="' + f1(sy(base)) + '" y2="' + f1(sy(base)) + '"/>'
    + '<text class="base-label" x="' + (W - R) + '" y="' + f1(sy(base) - 4) + '" text-anchor="end">baseline</text>';
  // Every proxy at once has no single best to follow: the dots alone.
  let line = "";
  if (!session.combined) {
    const pts = best.map((v, i) => (v === null ? null : [sx(i), sy(v)] as const)).filter((p): p is readonly [number, number] => !!p);
    line = pts.length > 1 ? '<path class="best" d="M' + pts.map(([x, y], i) => (i ? "H" + f1(x) + " V" + f1(y) : f1(x) + "," + f1(y))).join(" ") + '"/>' : "";
  }
  const showN = (W - L - R) / Math.max(1, n - 1) >= 26;
  const dot = (i: number, n_: number, v: number | null, cls: string, tip: string) => {
    if (v === null) return "";
    const x = f1(sx(i)), y = f1(sy(v)), cur = n_ === k;
    return '<g class="dot ' + cls + (cur ? " is-cur" : "") + (n_ > k ? " is-after" : "") + '" data-step="' + n_ + '">'
      + "<title>" + esc(tip) + "</title>"
      + '<circle class="hit" cx="' + x + '" cy="' + y + '" r="11"/>'
      + (cur ? '<circle class="ring" cx="' + x + '" cy="' + y + '" r="9"/>' : "")
      + '<circle class="mark" cx="' + x + '" cy="' + y + '" r="' + (cur ? 5.5 : 4.5) + '"/></g>';
  };
  const dots = dot(0, 0, base, "base", "Baseline · " + session.metric + " " + value(base))
    + session.steps.map((s, i) => dot(i + 1, s.n, s.value, kept(s) ? "kept" : "rej",
      "#" + s.n + " " + (kept(s) ? "kept" : s.verdict.replace("reject", "ruled out")) + " · "
      + (session.combined && s.proxy ? s.proxy + " · " : "") + (s.delta === null ? "–" : signed(s.delta))
      + " · " + (session.combined ? (s.metric ?? "") + " × baseline " : session.metric + " ") + value(s.value)
      + (s.axis ? " · " + s.axis : ""))).join("");
  const labels = showN ? ['<text class="xl" x="' + f1(sx(0)) + '" y="' + (H - 6) + '" text-anchor="middle">base</text>']
    .concat(session.steps.map((s, i) => '<text class="xl" x="' + f1(sx(i + 1)) + '" y="' + (H - 6) + '" text-anchor="middle">#' + s.n + "</text>")).join("") : "";
  const chart = '<svg class="chart" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + " " + H
    + '" role="img" aria-label="' + esc(session.metric) + ' at each step">' + grid + baseLine + line + dots + labels + "</svg>";
  const chip = (n_: number, cls: string, body: string, tip: string) => '<button class="chip ' + cls + (n_ > k ? " is-after" : "")
    + '" data-step="' + n_ + '" aria-current="' + (n_ === k) + '" title="' + esc(tip) + '">' + body + "</button>";
  const chips = chip(0, "base", "Baseline <b>" + value(base) + "</b>", session.metric + " " + value(base) + " · " + session.baseline.run_id)
    + session.steps.map((s) => chip(s.n, kept(s) ? "kept" : "rej",
      "#" + s.n + (kept(s) ? "" : " ✕") + (session.combined && s.proxy ? " " + esc(s.proxy) : "")
      + " <b>" + (s.delta === null ? "–" : signed(s.delta)) + "</b>",
      (s.axis || "(no axis given)") + " · " + s.changed.length + " files")).join("");
  const keptCount = session.steps.filter(kept).length;
  const last = best[best.length - 1];
  const total = last !== null && base ? (last - base) / Math.abs(base) : null;
  const since = session.combined ? perProxy(session) + " since their baselines"
    : esc(session.metric) + " <b>" + signed(total) + "</b> since baseline";
  return '<div class="session-head"><div class="title">' + (session.combined ? "Every proxy" : "Optimisation session")
    + '</div><span class="sum">' + session.steps.length + " steps tried · <b>" + keptCount + "</b> kept · " + since + "</span>"
    + '<div class="seg push" role="group" aria-label="Which changes to show">'
    + '<button data-scope="step" aria-pressed="' + (scope === "step") + '">This step</button>'
    + '<button data-scope="all" aria-pressed="' + (scope === "all") + '">Since baseline</button></div></div>'
    + '<div class="chart-wrap">' + chart + "</div>"
    + '<div class="chips" role="group" aria-label="Steps">' + chips + "</div>";
}

/** The card of step *k* in the side pane, for when nothing on the map is selected. */
const profilePrompt = (proxy?: string) => "Profile the optimisation session's code as it stands now with "
  + "proxy_eval(op='profile'" + (proxy ? ", proxy_name='" + proxy + "'" : "") + ", confirm=True), then tell me where its time goes.";

/** Whether *s* is the last step of its session: of its own proxy, for every proxy at once. */
function latest(session: Session, s: Step): boolean {
  return !session.steps.some((x) => x.n > s.n && x.proxy === s.proxy);
}

/** What a step's card says of its time: its costliest files, why it has no profile, or
 *  (the latest step only: the code as it is now) a button that asks for one. */
function timeBlock(session: Session, s: Step, fileRow: Row): string {
  const p = s.profile;
  if (usable(p)) {
    const top = Object.entries(p.files).slice(0, 5);
    return '<div class="cap">Where its time goes (' + esc(p.tool) + ")</div>"
      + '<div class="list">' + top.map(([f, v]) => fileRow(f, pctOf(v))).join("") + "</div>"
      + (p.outside ? '<div class="empty">' + pctOf(p.outside) + " outside the workspace</div>" : "")
      + (p.note ? '<div class="empty">' + esc(p.note) + "</div>" : "")
      + functionsBlock(p.functions, fileRow, 5)
      + gpuBlock(p.gpu) + memoryBlock(p.memory, fileRow);
  }
  if (p && p.unavailable) return '<div class="cap">Not profiled</div><div class="empty">' + esc(p.unavailable) + "</div>";
  if (latest(session, s)) {
    return '<button class="open" data-prompt="' + esc(profilePrompt(session.combined ? s.proxy : undefined))
      + '">Profile the code as it is now</button>';
  }
  return '<div class="empty">Not profiled. Only the code as it is now can be profiled.</div>';
}

/** A file in a list: *label* (else the file's name), *right* beside it, opened at *line*. */
export type Row = (id: string, right: string, line?: number, label?: string, tip?: string) => string;

/** A function's key split at its first `::`: no path has one, a Fortran name keeps its own. */
export function fnParts(key: string): { file: string; name: string } {
  const at = key.indexOf("::");
  return at < 0 ? { file: key, name: key } : { file: key.slice(0, at), name: key.slice(at + 2) };
}

/** The *n* costliest functions by self time, of the files *within* when given. */
export function topFunctions(fns: Record<string, FnTime> | null | undefined, n: number,
                             within?: (file: string) => boolean): [string, FnTime][] {
  return Object.entries(fns ?? {}).filter(([k]) => !within || within(fnParts(k).file))
    .sort((a, b) => b[1].self - a[1].self || b[1].total - a[1].total).slice(0, n);
}

/** The costliest functions as rows: self time beside them, the total in their tooltip. */
export function functionsBlock(fns: Record<string, FnTime> | null | undefined, row: Row, n: number,
                               within?: (file: string) => boolean): string {
  const top = topFunctions(fns, n, within).filter(([, v]) => v.self > 0 || v.total > 0);
  if (!top.length) return "";
  return '<div class="cap">Functions, own time</div><div class="list">' + top.map(([k, v]) => {
    const { file, name } = fnParts(k);
    return row(file, pctOf(v.self), v.line, name, name + " in " + file + ": " + pctOf(v.self) + " own, "
      + pctOf(v.total) + " with what it calls");
  }).join("") + "</div>";
}

/** What a step's card says of its GPU: the costliest kernels, copies and idle time, or
 *  why the GPU could not be traced. */
export function gpuBlock(g?: GpuProfile | null): string {
  if (!g) return "";
  if (g.unavailable) return '<div class="cap">GPU not traced</div><div class="empty">' + esc(g.unavailable) + "</div>";
  const kernels = Object.entries(g.kernels ?? {}).slice(0, 5);
  const copies = Object.values(g.memcpy ?? {}).reduce((n, v) => n + v, 0);
  const moved = Object.entries(g.memcpy_mb ?? {}).map(([dir, v]) => dir + " " + megabytes(v)).join(", ");
  return '<div class="cap">On the GPU (' + esc(g.tool) + ")</div>"
    + '<div class="empty">kernels ' + pctOf(g.busy ?? 0) + " · copies " + pctOf(copies) + " · idle " + pctOf(g.idle ?? 0) + "</div>"
    + (g.peak_device_mb !== undefined || moved ? '<div class="empty">'
      + (g.peak_device_mb !== undefined ? "peak device memory " + megabytes(g.peak_device_mb) : "")
      + (g.peak_device_mb !== undefined && moved ? " · " : "") + (moved ? "copied " + esc(moved) : "") + "</div>" : "")
    + '<div class="list">' + kernels.map(([name, k]) => '<div class="row"><span title="' + esc(name) + '">'
      + esc(kernelName(name)) + "</span><span>" + pctOf(k.share) + "</span></div>").join("") + "</div>"
    + (g.note ? '<div class="empty">' + esc(g.note) + "</div>" : "");
}

/** Where the heap's peak was allocated, or why it could not be profiled. */
export function memoryBlock(m: MemoryProfile | null | undefined, fileRow: Row): string {
  if (!m) return "";
  if (m.unavailable) return '<div class="cap">Memory not profiled</div><div class="empty">' + esc(m.unavailable) + "</div>";
  const top = Object.entries(m.files ?? {}).slice(0, 5);
  return '<div class="cap">Where memory goes (' + esc(m.tool) + (m.peak_mb !== undefined ? ", heap peak " + megabytes(m.peak_mb) : "") + ")</div>"
    + '<div class="list">' + top.map(([f, v]) => fileRow(f, pctOf(v))).join("") + "</div>"
    + (m.note ? '<div class="empty">' + esc(m.note) + "</div>" : "");
}

/** A size in MB, as the pane writes it. */
export function megabytes(mb: number): string {
  return mb >= 1024 ? (mb / 1024).toFixed(1) + " GB" : mb >= 10 ? mb.toFixed(0) + " MB" : mb.toFixed(1) + " MB";
}

/** A kernel's name without its parameters, as short as the pane needs it. */
export function kernelName(name: string): string {
  const bare = name.replace(/^void\s+/, "").split("(")[0];
  return bare.length > 48 ? bare.slice(0, 47) + "…" : bare;
}

/** A share of the baseline's time, as the map writes it. */
export function pctOf(v: number): string {
  return v >= 0.0995 ? (v * 100).toFixed(0) + "%" : v >= 0.001 ? (v * 100).toFixed(1) + "%" : v > 0 ? "<0.1%" : "–";
}

/** A step's memory as the card writes it: summed over its processes when there were
 *  several (an MPI run), with the largest one; its move against the incumbent's; and
 *  whether it grew. */
function memoryLine(s: Step): { text: string; grew: boolean } | null {
  const summed = !!s.peak_rss_total_mb && !!s.peak_rss_mb && s.peak_rss_total_mb > s.peak_rss_mb * 1.05;
  const now = summed ? s.peak_rss_total_mb! : s.peak_rss_mb;
  if (!now) return null;
  const was = summed ? s.incumbent_peak_rss_total_mb : s.incumbent_peak_rss_mb;
  return {
    text: megabytes(now) + (summed ? " summed, largest process " + megabytes(s.peak_rss_mb!) : "")
      + (was ? " (" + signed((now - was) / was) + ")" : ""),
    grew: !!was && now > was * 1.05,
  };
}

export function stepCard(session: Session, k: number, fileRow: Row): string {
  const s = session.steps.find((x) => x.n === k);
  if (!s) return "";
  const memory = memoryLine(s);
  const line = (l: string, v: string, cls = "") =>
    '<div class="line"><span>' + esc(l) + '</span><span class="' + cls + '">' + v + "</span></div>";
  return '<div class="section step"><div class="title">Step #' + s.n + "</div>"
    + '<div class="path">' + esc(s.axis || "(no axis given)") + "</div>"
    + (session.combined && s.proxy ? line("proxy", esc(s.proxy)) : "")
    + line("verdict", kept(s) ? "✓ kept" : "✕ " + esc(s.verdict.replace("reject", "ruled out")), kept(s) ? "is-ok" : "is-danger")
    + line(session.combined ? (s.metric ?? "metric") + " × baseline" : session.metric,
      value(s.value) + " against " + value(s.incumbent_value)
      + (s.delta === null ? "" : " (" + signed(s.delta) + ")"), better(session, s.delta) ? "is-ok" : "")
    + (memory ? line("peak memory", memory.text, memory.grew ? "is-danger" : "") : "")
    + line("dependencies", '<span class="is-ok">+' + s.added.length + '</span> / <span class="is-danger">−' + s.removed.length + "</span>")
    + '<div class="cap">Files it changed (' + s.changed.length + ")</div>"
    + '<div class="list">' + (s.changed.map((f) => fileRow(f, "")).join("") || '<span class="empty">none</span>') + "</div>"
    + timeBlock(session, s, fileRow) + "</div>";
}
