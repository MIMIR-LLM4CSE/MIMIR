// The code map tab: the workspace's modules in bands, one per component, stacked by tier;
// the dependencies between them, coloured by what they are; the selection beside. It draws
// what the server answered (`shared.code_map`) and reads nothing else.

import { STYLE } from "./style";
import { basename, esc, signed } from "./format";
import { barWidth, cap, countersBlock, fnParts, functionsBlock, gpuBlock, kernelBlock, memoryBlock, note, overlay, pctOf,
  stepCard, strip, timeAt, topFunctions, type Overlay, type RowExtra, type Scope, type Session, type TimeView } from "./session";
import { newDraft, ready, requestPrompt, sameTarget, targetLabel, touches, withTarget,
  type Draft, type Request, type Sent, type Target } from "./request";
import { formHtml, requestDetail, requestsSection } from "./requestView";
import { picked, Picker } from "./picker";
import { mergeGraph, type Forecast, type MergeState } from "./merge";
import { barsHeight, BW, curve, derive, edgeClass, FN_N, FN_ROW, GX, LW, marked, midOf, pathOf, visible, type ModEdge, type View } from "./layout";
import { fitText, svg } from "./svgText";
import type { CodeMap, Edge, File, Selection } from "./types";


/** What the tab keeps across its reloads: its choices, the request being written, and
 *  those sent that the server has not answered yet. */
interface TabState { skin?: string; tests?: boolean; fns?: boolean; pick?: string; draft?: Draft | null; sent?: Sent[] }
declare function acquireVsCodeApi(): {
  postMessage(msg: unknown): void;
  getState(): TabState | undefined;
  setState(state: TabState): void;
};
const vscode = acquireVsCodeApi();

// A bar's figure takes PCT_W at the box's right, "<0.1%" included.
const PCT_W = 46;
const LANG: Record<string, string> = { python: "py", cython: "pyx", cpp: "c++", c: "c", fortran: "f", pyf: "f2py" };


let map: CodeMap | null = null;
let sel: Selection = null;
// The optimisation session, when the workspace has one: the step shown, and whether the
// map marks that step's changes alone or every kept step's since the baseline.
let session: Session | null = null;
let step = 0;
let scope: Scope = "step";
let skin = vscode.getState()?.skin === "light" ? "light" : "mimir";
// The tests are folded away unless asked for: they depend on everything and would bury
// the code's own architecture under their edges.
let showTests = vscode.getState()?.tests === true;
// Each module's costliest functions in its box: off unless asked for, the boxes grow with them.
let showFns = vscode.getState()?.fns === true;
// The columns of modules the tab's width holds, and how wide that makes a box.
let cols = 5, boxW = BW;
// Whose steps the map shows, and the picker that changes it: a pick shown is asked of the server.
const picker = new Picker(vscode.getState()?.pick ?? "", {
  choose: () => { remember(); remap(); },
  prompt: (text) => vscode.postMessage({ type: "prompt", text }),
});

// The request being written, the ones sent and not yet recorded, and the recorded ones.
let draft: Draft | null = vscode.getState()?.draft ?? null;
let sent: Sent[] = vscode.getState()?.sent ?? [];
let requests: Request[] = [];

function remember(): void { vscode.setState({ skin, tests: showTests, fns: showFns, pick: picker.pick, draft, sent }); }

/** Ask the server again, for the steps the pick names. */
function ask(): void {
  vscode.postMessage({ type: "refresh", ...picker.query() });
}

// A mapping asked for (⟳, the proxy picker, a step settled): under way, done (and when), or
// failed and why. The map drawn again can look exactly the same, so the header says it.
let mapping: "idle" | "busy" | "failed" = "idle";
let mappedAt: Date | null = null;
let mappingNote = "";
let mappingTimer: ReturnType<typeof setTimeout> | undefined;
// Past this, an answer that never came is said to have not come.
const MAPPING_PATIENCE_MS = 60000;

// What the agent did may have moved the map (the server says so after a run settles or a
// request is recorded): the tab maps the code again on its own. A burst of calls is one
// re-map, and a tab out of sight waits until it is seen.
const STALE_SETTLE_MS = 1500;
let staleTimer: ReturnType<typeof setTimeout> | undefined;
let staleWhileHidden = false;

/** Map the code again (⟳, the proxy picker, or the server saying something moved),
 *  saying so until the answer is in. */
function remap(): void {
  awaitMap();
  ask();
}

/** Say the map is on its way until it is in, or until it is late. */
function awaitMap(): void {
  mapping = "busy";
  clearTimeout(mappingTimer);
  mappingTimer = setTimeout(() => {
    if (mapping === "busy") settleMapping("failed", "no answer from the server; try ⟳ again");
  }, MAPPING_PATIENCE_MS);
  renderStatus();
}

function settleMapping(to: "idle" | "failed", note = ""): void {
  const was = mapping;
  clearTimeout(mappingTimer);
  mapping = to;
  mappingNote = note;
  if (to === "idle") mappedAt = new Date();
  renderStatus(was === "busy");
}

/** The ⟳ button and the line beside the counts: mapping, mapped at, or why not. */
function renderStatus(flash = false): void {
  const button = document.getElementById("reload");
  const status = document.getElementById("status");
  if (!button || !status) return;
  const busy = mapping === "busy";
  button.classList.toggle("is-busy", busy);
  button.toggleAttribute("disabled", busy);
  button.setAttribute("aria-busy", String(busy));
  button.title = busy ? "Mapping the code…"
    : "Scan the workspace again: its files, the dependencies between them, the optimisation steps (no model call)";
  // Toggled, not rewritten: a redraw right after must not cut the flash short.
  status.classList.toggle("is-warn", mapping === "failed");
  status.textContent = busy ? "· mapping the code…"
    : mapping === "failed" ? "· " + mappingNote
    : mappedAt ? "· mapped at " + mappedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
  if (flash) {
    status.classList.remove("flash");
    void status.offsetWidth;
    status.classList.add("flash");
  }
}

// A merge session's state, or what merging the ticked proxies promises.
let mergeState: MergeState | null = null;
let forecast: Forecast | null = null;

/** The map laid out to the tab's width and the boxes' height. */
function layoutOf(m: CodeMap): View {
  return derive(m, cols, boxW, boxHeight());
}

/** How tall a box is: with its bars, and its functions when they are shown and the profile has them. */
function boxHeight(): number {
  const t = time();
  return barsHeight(t) + (showFns && t?.functions ? 4 + FN_N * FN_ROW : 0);
}

function selectedModule(v: View): string | null {
  if (!sel || sel.kind === "comp" || sel.kind === "request") return null;
  return sel.kind === "module" ? sel.id : v.byId.get(sel.id)?.module ?? null;
}

/** The modules a selection stands for: a component's all, else the one module. */
function selectedModules(v: View): Set<string> {
  if (sel?.kind === "comp") return new Set(map!.modules.filter((x) => x.component === sel!.id).map((x) => x.id));
  if (sel?.kind === "request") {
    const r = requests.find((x) => x.id === sel!.id);
    return new Set(r ? [...v.modFiles].filter(([id, fs]) => requestOn(r, v, id, fs)).map(([id]) => id) : []);
  }
  const one = selectedModule(v);
  return new Set(one ? [one] : []);
}

/** Whether a selection still names something on the map drawn. */
function stillThere(v: View, s: Selection): boolean {
  if (!s) return true;
  if (s.kind === "request") return true;
  return s.kind === "comp" ? v.compOf.has(s.id) : s.kind === "module" ? v.modFiles.has(s.id) : v.byId.has(s.id);
}

/** Whether request *r* is about module *id*, whose files are *files*. */
function requestOn(r: { targets: Target[] }, v: View, id: string, files: File[]): boolean {
  return touches(r, id, files[0]?.component ?? "", new Set(files.map((f) => f.id)));
}

// Set when a component arrives from the drawer: the map scrolls to its band once drawn.
let scrollToBand = false;

/** One time bar in a module's box *bw* wide: its share now, hatched where it was larger before. */
function timeBar(g: SVGElement, bw: number, now: number, was: number | null, scale: number, y: number, cls: string): void {
  // The figure has a column of its own, right of the track: it never reaches the bar.
  const BW = bw, tw = BW - 9 - PCT_W;
  svg("rect", { class: "track", x: 9, y, width: tw, height: 4, rx: 2 }, g);
  if (now > 0) svg("rect", { class: ("fill " + cls).trim(), x: 9, y, width: Math.max(2, (now / scale) * tw), height: 4, rx: 2 }, g);
  if (was !== null && was - now > 1e-6) {
    svg("rect", { x: 9 + (now / scale) * tw, y: y - 1, width: ((was - now) / scale) * tw, height: 6, fill: cls ? "url(#hatch-gpu)" : "url(#hatch)" }, g);
  }
  svg("text", { class: ("pct " + cls).trim(), x: BW - 8, y: y + 4.5, "text-anchor": "end" }, g).textContent = pctOf(now);
}

/** A module's costliest functions under its bars: each name, a thin bar on the module's
 *  scale, and its own share of the baseline's time. */
function functionLines(g: SVGElement, bw: number, fns: NonNullable<TimeView["functions"]>, scale: number, top: number): void {
  const tw = 34, x0 = bw - PCT_W - tw;
  topFunctions(fns, FN_N).filter(([, f]) => f.self > 0).forEach(([k, f], i) => {
    const y = top + i * FN_ROW + 4;
    const name = fnParts(k).name;
    const t = svg("text", { class: "fn", x: 9, y: y + 3 }, g);
    t.textContent = fitText(name, x0 - 15, "fn", (x) => x.slice(0, -2) + "…");
    svg("title", {}, t).textContent = name + ": " + pctOf(f.self) + " own, " + pctOf(f.total) + " with what it calls";
    svg("rect", { class: "fill fn-bar", x: x0, y: y - 2, width: Math.max(1.5, Math.min(1, f.self / scale) * tw), height: 3, rx: 1.5 }, g);
    svg("text", { class: "fn-pct", x: bw - 8, y: y + 3, "text-anchor": "end" }, g).textContent = pctOf(f.self);
  });
}

/** A module's share of the baseline's time in *files*: the sum of its files'. */
function moduleTime(v: View, id: string, files: Record<string, number>): number {
  return (v.modFiles.get(id) ?? []).reduce((n, f) => n + (files[f.id] ?? 0), 0);
}

/** A profile's functions grouped by their file, once per profile. */
const fnsByFile = new WeakMap<object, Map<string, [string, NonNullable<TimeView["functions"]>[string]][]>>();
function functionsOf(fns: NonNullable<TimeView["functions"]>, files: File[]): NonNullable<TimeView["functions"]> {
  let by = fnsByFile.get(fns);
  if (!by) {
    by = new Map();
    for (const e of Object.entries(fns)) {
      const f = fnParts(e[0]).file;
      (by.get(f) ?? by.set(f, []).get(f)!).push(e);
    }
    fnsByFile.set(fns, by);
  }
  return Object.fromEntries(files.flatMap((f) => by!.get(f.id) ?? []));
}

/** What the map drawn was drawn from, and how to bring its selection up to date: a click
 *  that only selects restyles the map, it does not draw it again. */
let drawn: { from: unknown[]; restyle: () => void } | null = null;

function renderMap(v: View, ov: Overlay | null, time: TimeView | null): void {
  const from = [v, session, step, scope, showFns, requests, time];
  if (drawn && drawn.from.length === from.length && drawn.from.every((x, i) => x === from[i])) { drawn.restyle(); return; }
  const root = document.getElementById("map")!;
  root.innerHTML = "";
  const s = svg("svg", { viewBox: `0 0 ${v.width} ${v.height}`, role: "img", "aria-label": "Modules by component, with the dependencies between them" }, root);
  // Drawn at its own size: the columns follow the tab's width, so nothing is scaled.
  s.setAttribute("width", String(v.width));
  s.setAttribute("height", String(v.height));
  const defs = svg("defs", {}, s);
  for (const [id, cls] of [["hatch", "hatch"], ["hatch-gpu", "hatch gpu"]]) {
    const pat = svg("pattern", { id, width: 4, height: 4, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    svg("rect", { class: cls, width: 2, height: 4 }, pat);
  }
  // One scale for every step and both sides: the costliest module of whichever profile is drawn.
  const tables = time ? [time.files, time.before, time.gpuFiles, time.gpuBefore].filter((x): x is Record<string, number> => !!x) : [];
  const scale = Math.max(1e-9, ...[...v.modFiles.keys()].flatMap((id) => tables.map((t) => moduleTime(v, id, t))));
  const bandRects: SVGRectElement[] = [];
  for (const b of v.bands) {
    const band = svg("rect", { class: "band-bg", x: 0, y: b.y, width: v.width, height: b.h, rx: 6 }, s);
    band.dataset.comp = b.c.id;
    bandRects.push(band);
    band.addEventListener("click", () => {
      sel = sel?.kind === "comp" && sel.id === b.c.id ? null : { kind: "comp", id: b.c.id };
      render();
    });
    svg("rect", { class: "band-bar", x: 12, y: b.y + 15, width: 3, height: 11, rx: 1.5 }, s);
    svg("text", { class: "band-name", x: 21, y: b.y + 25 }, s).textContent = b.c.name;
    svg("text", { class: "band-sub", x: 12, y: b.y + 42 }, s).textContent = `tier ${b.c.tier}`;
    const dl = b.c.declared.map(basename);
    const t = svg("text", { class: "band-sub", x: 12, y: b.y + 57 }, s);
    t.textContent = dl.length ? (dl.length > 2 ? `→ ${dl.length} components` : `→ ${dl.join(", ")}`) : "";
    if (dl.length > 2) svg("title", {}, t).textContent = dl.join(", ");
  }
  const gE = svg("g", {}, s), gX = svg("g", {}, s), gM = svg("g", {}, s);
  const rank = (e: ModEdge) => (e.removed || e.added ? 4 : e.breaks ? 3 : e.cycle ? 2 : e.binding ? 1 : 0);
  // Each path with whether it keeps its own colour when lit (a break, a cycle).
  const paths: { p: SVGPathElement; from: string; to: string; loud: boolean }[] = [];
  for (const me of marked(v, ov).sort((a, b) => rank(a) - rank(b))) {
    const c = curve(v, me.from, me.to), cls = edgeClass(me);
    const p = svg("path", { d: pathOf(c), class: cls,
      "stroke-width": (me.breaks || me.removed || me.added ? 2.5 : 1.2) + Math.log2(me.edges.length) * 0.6 }, gE);
    paths.push({ p, from: me.from, to: me.to, loud: /is-(break|cycle)/.test(cls) });
    if (me.breaks) {
      const mid = midOf(c);
      svg("text", { class: "xmark", x: mid.x, y: mid.y + 4, "text-anchor": "middle" }, gX).textContent = "✕";
    }
  }
  const boxes = new Map<string, SVGGElement>();
  const BW = v.bw, BH = v.bh;
  const fns = showFns && time ? time.functions : null;
  for (const [id, files] of v.modFiles) {
    const p = v.pos.get(id);
    if (!p) continue;
    const g = svg("g", { class: "m", transform: `translate(${p.x},${p.y})`, tabindex: 0, role: "button", "aria-label": id }, gM);
    svg("rect", { class: "box", width: BW, height: BH, rx: 6 }, g);
    const comp = files[0].component;
    const nm = svg("text", { class: "mname", x: 9, y: 18 }, g);
    nm.textContent = fitText(id === comp ? basename(id) : id.slice(comp.length + 1), BW - 30, "mname", (x) => "…" + x.slice(2));
    const lines = files.reduce((n, f) => n + f.lines, 0);
    const langs = [...new Set(files.map((f) => LANG[f.lang] ?? f.lang))].join(" ");
    svg("text", { class: "msub", x: 9, y: 33 }, g).textContent = `${files.length} files · ${(lines / 1000).toFixed(1)}k · ${langs}`;
    if (time) {
      // With a GPU traced, its bar sits under the CPU's, on the same scale; alone when the CPU was not sampled.
      const gpuNow = time.gpuFiles ? moduleTime(v, id, time.gpuFiles) : 0;
      const cpu = !time.cpuUnavailable, both = cpu && !!time.gpuFiles;
      if (cpu) timeBar(g, BW, moduleTime(v, id, time.files), time.before ? moduleTime(v, id, time.before) : null, scale, both ? 39 : 40, "");
      if (time.gpuFiles) timeBar(g, BW, gpuNow, time.gpuBefore ? moduleTime(v, id, time.gpuBefore) : null, scale, both ? 50 : 40, "gpu");
    }
    if (fns) functionLines(g, BW, functionsOf(fns, files), scale, barsHeight(time) + 4);
    let dx = BW - 9;
    if (v.breakMods.has(id)) { svg("circle", { class: "dot-break", cx: dx, cy: 12, r: 4 }, g); dx -= 11; }
    if (v.cycleOf.has(id)) svg("circle", { class: "dot-cycle", cx: dx, cy: 12, r: 4 }, g);
    // Pills ride the box's top edge, clear of its bars and functions: the steps that changed
    // it on the right, its requests on the left, as many as the room between them allows.
    let stepPills = 0;
    if (ov) {
      if (files.some((f) => ov.touchedNow.has(f.id))) g.classList.add("is-touched");
      else if (files.some((f) => ov.touched.has(f.id))) g.classList.add("is-touched-old");
      const ids = new Set(files.map((f) => f.id));
      const mine = ov.steps.filter((st) => st.changed.some((f) => ids.has(f))).slice(-2);
      stepPills = mine.length;
      mine.forEach((st, i) => {
        const pl = svg("g", { class: "pill", transform: `translate(${BW - 52 - i * 54},-8)` }, g);
        svg("rect", { class: st.verdict === "accept" ? "ok" : "no", width: 48, height: 15, rx: 7.5 }, pl);
        svg("text", { x: 24, y: 11, "text-anchor": "middle" }, pl).textContent = "#" + st.n + " " + (st.delta === null ? "" : signed(st.delta, true));
        svg("title", {}, pl).textContent = "#" + st.n + " " + st.axis;
      });
    }
    const room = (stepPills ? BW - 52 - (stepPills - 1) * 54 - 6 : BW - 9) - 9;
    const fit = Math.max(0, Math.min(3, Math.floor((room + 6) / 42)));
    const asked = requests.filter((r) => r.status !== "dropped" && requestOn(r, v, id, files));
    asked.slice(asked.length - Math.min(fit, asked.length)).forEach((r, i) => {
      const pl = svg("g", { class: "rq-pill st-" + r.status, transform: `translate(${9 + i * 42},-8)` }, g);
      svg("rect", { width: 36, height: 15, rx: 7.5 }, pl);
      svg("text", { x: 18, y: 11, "text-anchor": "middle" }, pl).textContent = r.id;
      svg("title", {}, pl).textContent = r.id + " · " + r.status + " · " + r.idea;
      pl.addEventListener("click", (ev) => { ev.stopPropagation(); sel = { kind: "request", id: r.id }; render(); });
    });
    svg("title", {}, g).textContent = id;
    boxes.set(id, g);
    const pick = (ev?: Event) => {
      // With a request being written, Shift+click adds the module to its targets.
      if (draft && (ev as MouseEvent | undefined)?.shiftKey) { addTarget({ module: id }); return; }
      sel = sel?.kind === "module" && sel.id === id ? null : { kind: "module", id }; render();
    };
    g.addEventListener("click", pick);
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
    g.addEventListener("mouseenter", () => highlight(new Set([id])));
    g.addEventListener("mouseleave", () => highlight(selectedModules(v)));
  }
  /** *core* and the modules linked to it stand out, with the links between them. */
  function highlight(core: Set<string>): void {
    const any = core.size > 0;
    const near = new Set(core);
    for (const { p, from, to, loud } of paths) {
      const on = any && (core.has(from) || core.has(to));
      p.classList.toggle("is-hot", on && !loud);
      p.classList.toggle("is-faded", any && !on);
      if (on) { near.add(from); near.add(to); }
    }
    const chosen = selectedModules(v);
    for (const [mid, g] of boxes) {
      g.classList.toggle("is-dim", any && !near.has(mid));
      g.classList.toggle("is-sel", chosen.has(mid));
    }
  }
  /** The selection marked: its band, its modules and what they link. */
  function restyle(): void {
    v.bands.forEach((b, i) => bandRects[i].classList.toggle("is-sel", sel?.kind === "comp" && sel.id === b.c.id));
    highlight(selectedModules(v));
    if (scrollToBand && sel?.kind === "comp") {
      scrollToBand = false;
      s.querySelector(`[data-comp="${CSS.escape(sel.id)}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }
  drawn = { from, restyle };
  restyle();
}

/** A file in a list, or a function (*label*) in it: a click shows the file on the map, a
 *  double-click opens it at *line*. */
function fileRow(id: string, right: string, line?: number, label?: string, tip?: string, extra: RowExtra = {}): string {
  const how = "click: show on the map · double-click: open" + (line ? " at line " + line : "");
  const meter = extra.w !== undefined;
  return '<div class="row' + (extra.sub ? " two" : "") + (meter ? " meter" + (extra.cls ? " " + extra.cls : "") + '" style="--w:' + barWidth(extra.w!) : "")
    + '"><button class="go" data-file="' + esc(id) + '" data-line="' + (line ?? "")
    + (label ? '" data-fn="' + esc(label) : "")
    + '" title="' + esc((tip ?? id) + "\n" + how) + '"><span class="nm">' + esc(label ?? basename(id)) + "</span>"
    + (extra.sub ? '<span class="sub">' + esc(extra.sub) + "</span>" : "") + "</button><span>" + esc(right) + "</span></div>";
}
/** A module in a list, selecting it in the map. */
function moduleRow(id: string, right: string | number = "", label = id): string {
  return '<div class="row"><button class="go" data-module="' + esc(id) + '" title="' + esc(id) + '">'
    + esc(label) + "</button><span>" + esc(right) + "</span></div>";
}
function line(label: string, value: string, cls = ""): string {
  return '<div class="line"><span>' + esc(label) + '</span><span class="' + cls + '">' + value + "</span></div>";
}
function list(rows: string[], empty = "nothing"): string {
  return '<div class="list">' + (rows.join("") || '<span class="empty">' + empty + "</span>") + "</div>";
}
/** How an edge reads beside the file at its other end: its line, how it crosses, its break. */
function edgeRight(e: Edge): string {
  const crossing = e.kind === "binding" ? " " + e.via + (e.sure === "by_name" ? " (by name)" : "") : "";
  return ":" + e.line + crossing + (e.breaks ? " ✕" : "");
}
function breakFlag(e: Edge): string {
  return '<div class="flag danger">✕ ' + esc(e.breaks)
    + list([fileRow(e.from, ":" + e.line, e.line), fileRow(e.to, "")]) + "</div>";
}
function cycleFlag(cycle: string[]): string {
  return '<div class="flag cycle">' + list(cycle.map((id) => moduleRow(id))) + "</div>";
}
function section(cls: string, title: string, body: string): string {
  return '<div class="section ' + cls + '"><div class="title">' + esc(title) + "</div>" + body + "</div>";
}
const BACK = '<button class="back" id="back">← Overview</button>';

function overview(m: CodeMap): string {
  const vias = new Map<string, number>();
  for (const e of m.edges) {
    if (e.kind !== "binding") continue;
    const k = e.via + (e.sure === "by_name" ? " (by name)" : "");
    vias.set(k, (vias.get(k) ?? 0) + 1);
  }
  const breaks = m.edges.filter((e) => e.breaks);
  const unresolved = Object.entries(m.unresolved);
  const code = line("files", m.files.length + " · " + m.modules.length + " modules")
    + line("components", String(m.components.length))
    + (m.truncated ? line("scan", "stopped early: a very large tree", "is-warn") : "");
  const look = (breaks.map(breakFlag).join("") || '<div class="empty">No rule broken.</div>')
    + '<div class="cap">Module cycles · ' + m.cycles.length + "</div>" + m.cycles.map(cycleFlag).join("");
  const across = ([...vias].map(([k, n]) => line(k, String(n))).join("")
    || '<div class="empty">No binding between languages.</div>')
    + (unresolved.length ? '<div class="cap">Not placed</div>'
      + unresolved.map(([k, n]) => line(k.split("_").join(" "), String(n))).join("") : "");
  return section("sel", "Code", code) + section("flags", "To look at", look) + section("bind", "Across languages", across);
}

/** The time of the session's step shown, or null without a session or a profile: the
 *  same object while the step shown is, which the map compares to know it is drawn. */
let timeOf: { session: Session | null; step: number; scope: Scope; t: TimeView | null } | null = null;
function time(): TimeView | null {
  if (!timeOf || timeOf.session !== session || timeOf.step !== step || timeOf.scope !== scope) {
    timeOf = { session, step, scope, t: session ? timeAt(session, step, scope) : null };
  }
  return timeOf.t;
}

/** Where the time goes over the whole code, from the profile shown: whose profile it is,
 *  then each side it took, its costliest first, a bar behind each on the list's largest. */
function timeSection(t: TimeView): string {
  const meta = ["profile of " + t.from, t.cpuUnavailable ? "" : t.tool,
    t.samples ? t.samples.toLocaleString("en") + " samples" : "", t.case ? "case " + t.case : ""].filter(Boolean);
  const top = Object.entries(t.files).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const max = top.length ? top[0][1] : 0;
  const cpu = t.cpuUnavailable
    ? cap("CPU not sampled") + note(t.cpuUnavailable)
    : cap("CPU", "share of the baseline's time")
      + list(top.map(([f, v]) => fileRow(f, pctOf(v), undefined, undefined, undefined, { w: max ? v / max : 0 })))
      + (t.outside ? '<div class="empty">' + pctOf(t.outside) + " outside the workspace</div>" : "")
      + note(t.note);
  return section("time", "Where time goes", '<div class="meta-line">' + meta.map((x) => "<span>" + esc(x) + "</span>").join("") + "</div>" + cpu
    + functionsBlock(t.functions, fileRow, 5)
    + gpuBlock(t.gpu) + memoryBlock(t.memory, fileRow) + countersBlock(t.counters) + kernelBlock(t.kernel));
}

/** A component: where it stacks, what the build says it links, its modules, and the
 *  components it depends on and is used by. */
function componentDetail(m: CodeMap, v: View, id: string): string {
  const c = v.compOf.get(id)!;
  const t = time();
  const files = m.files.filter((f) => f.component === id);
  const mods = m.modules.filter((x) => x.component === id).map((x) => x.id);
  const of = (f: string) => v.byId.get(f)!.component;
  const out = new Map<string, number>(), inn = new Map<string, number>();
  for (const e of m.edges) {
    const a = of(e.from), b = of(e.to);
    if (a === id && b !== id) out.set(b, (out.get(b) ?? 0) + 1);
    if (b === id && a !== id) inn.set(a, (inn.get(a) ?? 0) + 1);
  }
  const ids = new Set(files.map((f) => f.id));
  const breaks = m.edges.filter((e) => e.breaks && ids.has(e.from));
  const cycles = m.cycles.filter((cy) => cy.some((x) => mods.includes(x)));
  const compRows = (mm: Map<string, number>) => [...mm].sort((a, b) => b[1] - a[1]).map(([k, n]) =>
    '<div class="row"><button class="go" data-comp="' + esc(k) + '" title="' + esc(k) + '">' + esc(basename(k))
    + "</button><span>" + n + "</span></div>");
  const modTime = (x: string) => (t ? moduleTime(v, x, t.files) : 0);
  const modRows = [...mods].sort((a, b) => modTime(b) - modTime(a) || a.localeCompare(b)).map((x) =>
    moduleRow(x, t && modTime(x) ? pctOf(modTime(x)) : (v.modFiles.get(x) ?? []).length + " files",
              x === id ? basename(x) : x.slice(id.length + 1)));
  return BACK + section("sel", c.name, '<div class="path">' + esc(id) + "</div>"
    + line("tier", String(c.tier))
    + line("files", files.length + " · " + files.reduce((n, f) => n + f.lines, 0).toLocaleString("en") + " lines")
    + (c.declared.length ? line("build links", esc(c.declared.map(basename).join(", "))) : "")
    + (t && !t.cpuUnavailable ? line("time", pctOf(mods.reduce((n, x) => n + modTime(x), 0)) + " of the baseline's") : "")
    + (t?.gpuFiles ? line("GPU time", pctOf(mods.reduce((n, x) => n + moduleTime(v, x, t.gpuFiles!), 0)) + " of the baseline's") : "")
    + (t?.memory?.files ? line("memory", pctOf(mods.reduce((n, x) => n + moduleTime(v, x, t.memory!.files!), 0)) + " of the heap's peak") : "")
    + (breaks.length ? line("rule breaks", String(breaks.length), "is-danger") : "")
    + (cycles.length ? line("module cycles", String(cycles.length), "is-cycle") : "")
    + '<div class="cap">Modules (' + mods.length + ")</div>" + list(modRows)
    + '<div class="cap">Depends on</div>' + list(compRows(out))
    + '<div class="cap">Used by</div>' + list(compRows(inn))
    + breaks.map(breakFlag).join("")
    + newButtons({ component: id }));
}

function moduleDetail(m: CodeMap, v: View, id: string): string {
  const t = time();
  const files = [...(v.modFiles.get(id) ?? [])].sort((a, b) =>
    (t ? (t.files[b.id] ?? 0) - (t.files[a.id] ?? 0) : 0) || b.lines - a.lines);
  const ids = new Set(files.map((f) => f.id));
  const out = new Map<string, number>(), inn = new Map<string, number>();
  for (const e of m.edges) {
    const a = v.byId.get(e.from)!.module, b = v.byId.get(e.to)!.module;
    if (ids.has(e.from) && b !== id) out.set(b, (out.get(b) ?? 0) + 1);
    if (ids.has(e.to) && a !== id) inn.set(a, (inn.get(a) ?? 0) + 1);
  }
  const rows = (mm: Map<string, number>) => [...mm].sort((a, b) => b[1] - a[1]).map(([k, n]) => moduleRow(k, n));
  const cycle = v.cycleOf.has(id)
    ? line("cycle", "with " + (m.cycles[v.cycleOf.get(id)!].length - 1) + " modules", "is-cycle") : "";
  return BACK + section("sel", basename(id), '<div class="path">' + esc(id) + "</div>" + cycle
    + (t && !t.cpuUnavailable ? line("time", pctOf(moduleTime(v, id, t.files)) + " of the baseline's, profile of " + esc(t.from)) : "")
    + (t?.gpuFiles ? line("GPU time", pctOf(moduleTime(v, id, t.gpuFiles)) + " of the baseline's") : "")
    + (t?.memory?.files ? line("memory", pctOf(moduleTime(v, id, t.memory.files)) + " of the heap's peak") : "")
    + '<div class="cap">Files</div>' + list(files.map((f) => fileRow(f.id, t && t.files[f.id] ? pctOf(t.files[f.id]) : f.lines + " l")))
    + functionsBlock(t?.functions, fileRow, 10, (f) => ids.has(f))
    + '<div class="cap">Depends on</div>' + list(rows(out))
    + '<div class="cap">Used by</div>' + list(rows(inn))
    + newButtons({ module: id }));
}

function fileDetail(m: CodeMap, v: View, id: string): string {
  const f = v.byId.get(id)!;
  const out = m.edges.filter((e) => e.from === f.id), inn = m.edges.filter((e) => e.to === f.id);
  return BACK + section("sel", basename(f.id), '<div class="path">' + esc(f.id) + "</div>"
    + line("module", '<button class="link" data-module="' + esc(f.module) + '">' + esc(f.module) + "</button>")
    + line("language", esc(f.lang)) + line("lines", f.lines.toLocaleString("en"))
    + (time()?.files[f.id] ? line("time", pctOf(time()!.files[f.id]) + " of the baseline's") : "")
    + (time()?.gpuFiles?.[f.id] ? line("GPU time", pctOf(time()!.gpuFiles![f.id]) + " of the baseline's") : "")
    + (time()?.memory?.files?.[f.id] ? line("memory", pctOf(time()!.memory!.files![f.id]) + " of the heap's peak") : "")
    + functionsBlock(time()?.functions, fileRow, 12, (x) => x === f.id)
    + '<div class="cap">Depends on (' + out.length + ")</div>" + list(out.map((e) => fileRow(e.to, edgeRight(e))))
    + '<div class="cap">Used by (' + inn.length + ")</div>" + list(inn.map((e) => fileRow(e.from, edgeRight(e), e.line)))
    + '<button class="open" data-open="' + esc(f.id) + '" data-line="">Open ' + esc(basename(f.id)) + "</button>"
    + newButtons({ file: f.id }));
}

/** The buttons that start a request on *t*: any kind, or an analysis of its architecture. */
function newButtons(t: Target): string {
  return '<div class="actions"><button class="open ghost" data-new="' + esc(JSON.stringify(t)) + '">New request…</button>'
    + (t.file ? "" : '<button class="open ghost" data-new="' + esc(JSON.stringify(t)) + '" data-kind="refactor">Analyse the architecture</button>')
    + "</div>";
}

/** A request's target as a row: a file or function selects its file, a module or a
 *  component selects it, the whole code nothing. */
function targetRow(t: Target): string {
  if (t.file) return fileRow(t.file, t.function ? "" : "", t.line, t.function ?? undefined, targetLabel(t));
  if (t.module) return moduleRow(t.module, "module");
  if (t.component) return '<div class="row"><button class="go" data-comp="' + esc(t.component) + '">' + esc(t.component) + "</button><span>component</span></div>";
  return '<div class="row"><span class="go">the whole code</span></div>';
}

/** The modules a target stands for on the map. */
function modulesOf(v: View, t: Target): string[] {
  if (t.all) return [...v.modFiles.keys()];
  if (t.component) return map!.modules.filter((x) => x.component === t.component).map((x) => x.id);
  if (t.module) return [t.module];
  const f = t.file ? v.byId.get(t.file) : undefined;
  return f ? [f.module] : [];
}

/** What the map sees in a refactor's scope: its module cycles, the rule breaks that start
 *  or end there, and its most coupled modules. */
function findings(v: View, targets: Target[]): string[] {
  const mods = new Set(targets.flatMap((t) => modulesOf(v, t)));
  if (!mods.size) return [];
  const out: string[] = [];
  for (const c of map!.cycles) if (c.some((x) => mods.has(x))) out.push("module cycle: " + c.concat([c[0]]).join(" → "));
  for (const e of map!.edges) {
    const a = v.byId.get(e.from)?.module, b = v.byId.get(e.to)?.module;
    if (e.breaks && ((a && mods.has(a)) || (b && mods.has(b)))) out.push("rule break: " + e.from + ":" + e.line + " → " + e.to + " (" + e.breaks + ")");
  }
  const uses = new Map<string, Set<string>>(), usedBy = new Map<string, Set<string>>();
  for (const me of v.modEdges) {
    (uses.get(me.from) ?? uses.set(me.from, new Set()).get(me.from)!).add(me.to);
    (usedBy.get(me.to) ?? usedBy.set(me.to, new Set()).get(me.to)!).add(me.from);
  }
  const degree = (m: string) => (uses.get(m)?.size ?? 0) + (usedBy.get(m)?.size ?? 0);
  [...mods].filter((m) => degree(m) >= 3).sort((a, b) => degree(b) - degree(a)).slice(0, 4).forEach((m) =>
    out.push("coupling: " + m + " is used by " + (usedBy.get(m)?.size ?? 0) + " modules and uses " + (uses.get(m)?.size ?? 0)));
  return out.slice(0, 14);
}

/** The costliest functions of the draft's targets not yet among them, to add with a click. */
function suggestions(v: View, d: Draft): string {
  const fns = time()?.functions;
  if (!fns || d.kind === "refactor") return "";
  const mods = new Set(d.targets.filter((t) => !t.function).flatMap((t) => modulesOf(v, t)));
  const fileIn = (f: string) => mods.has(v.byId.get(f)?.module ?? "");
  const rows = topFunctions(fns, 12, fileIn).map(([k, f]) => ({ ...fnParts(k), f }))
    .filter(({ file, name }) => !d.targets.some((t) => sameTarget(t, { file, function: name })))
    .slice(0, 5).map(({ file, name, f }) => '<div class="row"><button class="go" data-add-fn="' + esc(JSON.stringify({ file, function: name, line: f.line }))
      + '" title="Add ' + esc(name) + ' to the targets">+ ' + esc(name) + "</button><span>" + pctOf(f.self) + "</span></div>");
  return rows.length ? '<div class="cap">Their costliest functions</div><div class="list">' + rows.join("") + "</div>" : "";
}

/** Whether *d* is the analysis of the whole code's architecture the header's button opens. */
function wholeCodeAnalysis(d: Draft): boolean {
  return d.kind === "refactor" && d.targets.length === 1 && !!d.targets[0].all;
}

/** Open *d* in the form and bring it to the user: the form may sit below the map (a narrow
 *  tab stacks the pane under it) or above what they were reading, and a click that changes
 *  something out of sight reads as a button that does nothing. */
function openDraft(d: Draft): void {
  draft = d;
  remember();
  render();
  const form = document.querySelector<HTMLElement>(".req-form");
  if (!form) return;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Moved to only when out of sight, its title first: a form already in view stays put.
  const r = form.getBoundingClientRect();
  if (r.top < 0 || r.top > innerHeight - 120) form.scrollIntoView({ block: "start", behavior: still ? "auto" : "smooth" });
  form.classList.add("is-new");
  form.querySelector<HTMLElement>("#rq-idea")?.focus({ preventScroll: true });
}

/** *t* added to the draft's targets. */
function addTarget(t: Target): void {
  if (!draft) return;
  draft.targets = withTarget(draft.targets, t);
  remember();
  render();
}

/** The draft sent to the agent as the user's message: it stays listed as sent until the
 *  server has recorded it. */
function sendDraft(v: View): void {
  if (!draft || !ready(draft)) return;
  const t = time();
  const share = (x: Target) => {
    if (!t) return null;
    if (x.function && x.file) {
      const f = t.functions?.[x.file + "::" + x.function];
      return f ? pctOf(f.self) : null;
    }
    const total = modulesOf(v, x).reduce((n, m) => n + moduleTime(v, m, t.files), 0);
    return total ? pctOf(total) : null;
  };
  vscode.postMessage({ type: "prompt", text: requestPrompt(draft, share, findings(v, draft.targets)) });
  sent.push({ token: draft.token, idea: draft.idea.trim(), mode: draft.mode, targets: draft.targets });
  draft = null;
  remember();
  render();
}

/** The form's text fields, read into the draft as they are typed, without drawing again. */
function bindForm(side: HTMLElement, v: View): void {
  if (!draft) return;
  const d = draft;
  const send = side.querySelector<HTMLButtonElement>("[data-send]");
  const field = (id: string, k: "idea" | "variants" | "source" | "constraints") => {
    const el = side.querySelector<HTMLInputElement | HTMLTextAreaElement>("#" + id);
    el?.addEventListener("input", () => {
      d[k] = el.value;
      remember();
      if (send) send.disabled = !ready(d);
    });
  };
  field("rq-idea", "idea"); field("rq-variants", "variants"); field("rq-source", "source"); field("rq-constraints", "constraints");
  side.querySelector<HTMLSelectElement>("#rq-kind")?.addEventListener("change", (e) => {
    d.kind = (e.target as HTMLSelectElement).value; remember(); render();
  });
  side.querySelectorAll<HTMLElement>("[data-mode]").forEach((b) => b.addEventListener("click", () => {
    d.mode = b.dataset.mode === "try" ? "try" : "study"; remember(); render();
  }));
  side.querySelectorAll<HTMLElement>("[data-untarget]").forEach((b) => b.addEventListener("click", () => {
    d.targets.splice(Number(b.dataset.untarget), 1); remember(); render();
  }));
  side.querySelectorAll<HTMLElement>("[data-add-fn]").forEach((b) => b.addEventListener("click", () => addTarget(JSON.parse(b.dataset.addFn!))));
  send?.addEventListener("click", () => sendDraft(v));
  side.querySelector("[data-cancel]")?.addEventListener("click", () => { draft = null; remember(); render(); });
}

/** The session's step whose run is *runId*, as the strip numbers it; null when not shown. */
function stepOfRun(runId: string): number | null {
  return session?.steps.find((s) => s.run_id === runId)?.n ?? null;
}

function renderSide(v: View): void {
  const m = map!, side = document.getElementById("side")!;
  const card = session && step ? stepCard(session, step, fileRow) : "";
  const t = time();
  const request = sel?.kind === "request" ? requests.find((r) => r.id === sel!.id) : undefined;
  const html = (draft ? formHtml(draft, suggestions(v, draft), findings(v, draft.targets)) : "")
    + (!sel ? requestsSection(requests, sent) + card + (t ? timeSection(t) : "") + overview(m)
    : sel.kind === "request" ? BACK + (request ? requestDetail(request, targetRow, stepOfRun) : '<div class="empty">No such request.</div>')
    : sel.kind === "comp" ? componentDetail(m, v, sel.id)
    : sel.kind === "module" ? moduleDetail(m, v, sel.id) : fileDetail(m, v, sel.id));
  side.innerHTML = html;
  bindForm(side, v);
  side.querySelectorAll<HTMLElement>("[data-new]").forEach((b) => b.addEventListener("click", () => {
    const d = newDraft([JSON.parse(b.dataset.new!)]);
    if (b.dataset.kind) d.kind = b.dataset.kind;
    openDraft(d);
  }));
  side.querySelectorAll<HTMLElement>("[data-request]").forEach((b) => b.addEventListener("click", () => { sel = { kind: "request", id: b.dataset.request! }; render(); }));
  side.querySelectorAll<HTMLElement>("[data-step-n]").forEach((b) => b.addEventListener("click", () => { step = Number(b.dataset.stepN); render(); }));
  side.querySelectorAll<HTMLElement>("[data-reveal]").forEach((b) => b.addEventListener("click", () => {
    vscode.postMessage({ type: "reveal_in_chat", token: b.dataset.reveal!, session: b.dataset.session || undefined });
  }));
  side.querySelectorAll<HTMLElement>("[data-ask]").forEach((b) => b.addEventListener("click", () => {
    vscode.postMessage({ type: "prompt", text: b.dataset.ask! });
    b.setAttribute("disabled", "");
    b.textContent = "Asked the agent";
  }));
  document.getElementById("back")?.addEventListener("click", () => { sel = null; render(); });
  side.querySelectorAll<HTMLElement>("[data-module]").forEach((b) => b.addEventListener("click", () => { sel = { kind: "module", id: b.dataset.module! }; render(); }));
  side.querySelectorAll<HTMLElement>("[data-comp]").forEach((b) => b.addEventListener("click", () => { sel = { kind: "comp", id: b.dataset.comp! }; scrollToBand = true; render(); }));
  side.querySelectorAll<HTMLElement>("[data-prompt]").forEach((b) => b.addEventListener("click", () => {
    vscode.postMessage({ type: "prompt", text: b.dataset.prompt! });
    b.setAttribute("disabled", "");
    b.textContent = "Asked the agent to profile";
  }));
  side.querySelectorAll<HTMLElement>("[data-open]").forEach((b) => b.addEventListener("click", () => {
    openRel(b.dataset.open!, b.dataset.line ? Number(b.dataset.line) : undefined);
  }));
}

/** Opens a file of the map at *line*. Its paths are relative to the server's root, which
 *  need not be the folder VS Code has open: the host is given both and opens whichever
 *  exists here. */
function openRel(rel: string, line?: number): void {
  const root = map?.root ?? "";
  const file = root ? root.replace(/[\\/]+$/, "") + "/" + rel : rel;
  vscode.postMessage({ type: "open_file", file, rel, line });
}

// A click on a file in the side pane waits this long for a second one: two open the file.
let pendingSelect: ReturnType<typeof setTimeout> | undefined;

/** A file row clicked: once shows it on the map, twice (or with Ctrl/Cmd) opens it. The
 *  listener sits on the pane, which outlives the rows each render replaces. */
function onSideClick(ev: MouseEvent): void {
  const b = (ev.target as Element).closest<HTMLElement>("[data-file]");
  if (!b) return;
  clearTimeout(pendingSelect);
  const id = b.dataset.file!;
  // With a request being written, Shift+click adds the file, or the function, to its targets.
  if (draft && ev.shiftKey) {
    addTarget({ file: id, ...(b.dataset.fn ? { function: b.dataset.fn } : {}),
      ...(b.dataset.line ? { line: Number(b.dataset.line) } : {}) });
    return;
  }
  if (ev.detail >= 2 || ev.ctrlKey || ev.metaKey) {
    openRel(id, b.dataset.line ? Number(b.dataset.line) : undefined);
    return;
  }
  const choose = () => { sel = { kind: "file", id }; render(); };
  // From the keyboard there is no second click to wait for.
  if (ev.detail === 0) choose();
  else pendingSelect = setTimeout(choose, 220);
}

function renderLegend(): void {
  document.getElementById("legend")!.innerHTML = `
    <span><i class="sw" style="border-color:var(--edge)"></i>depends on</span>
    <span><i class="sw" style="border-color:var(--e-binding)"></i>across languages</span>
    <span><i class="sw" style="border-color:var(--e-binding);opacity:.5"></i>by name only</span>
    <span><i class="sw" style="border-color:var(--e-break);border-top-width:3px"></i>breaks a rule ✕</span>
    <span><i class="sw" style="border-color:var(--e-cycle)"></i>inside a module cycle</span>
    <span><i class="dot" style="background:var(--e-break)"></i>has a break</span>
    <span><i class="dot" style="background:var(--e-cycle)"></i>in a cycle</span>`
    + (session && step ? stepLegend() : "") + timeLegend();
}

function timeLegend(): string {
  const t = time();
  if (!t) return "";
  return (t.cpuUnavailable ? "" : '<span><i class="sw bar"></i>share of the baseline\'s time, profile of ' + esc(t.from)
    + " (" + esc(t.tool) + ")" + (t.before ? ", hatched where this step saved it" : "") + "</span>")
    + (t.gpu ? '<span><i class="sw bar gpu"></i>' + (t.cpuUnavailable ? "GPU kernels' share of the baseline's time, profile of "
      + esc(t.from) + " (" + esc(t.gpu.tool) + ")" + (t.gpuBefore ? ", hatched where this step saved it" : "")
      : "its GPU kernels (" + esc(t.gpu.tool) + ")") + "</span>" : "")
    + (showFns && t.functions ? "<span>under the bars: each module's costliest functions, their own time</span>" : "");
}

function stepLegend(): string {
  const who = scope === "step" ? "this step" : "the kept steps";
  return '<span><i class="sw" style="border-color:var(--e-added);border-top-width:3px"></i>added by ' + who + "</span>"
    + '<span><i class="sw dash" style="border-color:var(--edge);border-top-width:3px"></i>removed by ' + who + "</span>"
    + "<span>green outline: the files " + who + " changed</span>"
    + "<span>the dependencies drawn are today's code</span>";
}

function renderSession(): void {
  const el = document.getElementById("session")!;
  const m = mergeState ?? forecast;
  el.hidden = (!session || !session.steps.length) && !m;
  if (el.hidden) return;
  const width = el.clientWidth - 30;
  el.innerHTML = (m ? mergeGraph(m, width) : "")
    + (session ? strip(session, step, scope, width) : "");
  el.querySelectorAll<HTMLElement>("[data-step]").forEach((b) => b.addEventListener("click", () => {
    step = Number(b.dataset.step); render();
  }));
  el.querySelectorAll<HTMLElement>("[data-scope]").forEach((b) => b.addEventListener("click", () => {
    scope = b.dataset.scope === "all" ? "all" : "step"; render();
  }));
}

let view: View | null = null;
/** The map as the server answered it, tests included. */
let raw: CodeMap | null = null;

/** The map, drawn again only when what it shows has changed, else restyled. */
function drawMap(): void {
  if (!map || !view || !map.files.length) return;
  // The boxes' height follows the functions shown, which follow the step shown.
  if (view.bh !== boxHeight()) view = layoutOf(map);
  renderMap(view, session && step ? overlay(session, step, scope) : null, time());
}

function render(): void {
  document.documentElement.dataset.skin = skin;
  document.querySelectorAll<HTMLButtonElement>("button[data-skin]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.skin === skin)));
  document.getElementById("tests")?.setAttribute("aria-pressed", String(showTests));
  // Pressed while the analysis of the whole code is being written.
  document.getElementById("analyse")?.setAttribute("aria-pressed", String(!!draft && wholeCodeAnalysis(draft)));
  const fnsButton = document.getElementById("fns");
  if (fnsButton) {
    // Offered when the profile shown kept its functions.
    fnsButton.hidden = !time()?.functions;
    fnsButton.setAttribute("aria-pressed", String(showFns));
  }
  renderStatus();
  if (!map || !view) return;
  const meta = document.getElementById("meta")!;
  const hiddenTests = raw ? raw.files.length - map.files.length : 0;
  meta.textContent = `${map.files.length} files · ${map.modules.length} modules`
    + (hiddenTests ? ` · ${hiddenTests} test files folded` : "");
  if (!map.files.length) {
    document.getElementById("map")!.innerHTML = '<p class="empty">No Python, C/C++ or Fortran code in this workspace.</p>';
    drawn = null;
    document.getElementById("side")!.innerHTML = "";
    return;
  }
  picker.render();
  renderSession();
  drawMap();
  renderLegend();
  renderSide(view);
}

function mount(): void {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.appendChild(style);
  document.getElementById("root")!.innerHTML = `
    <header>
      <h1>Code map</h1><span class="meta" id="meta">waiting for the server…</span>
      <span class="meta status" id="status" role="status" aria-live="polite"></span>
      <div class="picker push" id="proxy" hidden></div>
      <button class="toggle" id="analyse" title="Ask the agent to analyse the whole code's architecture and propose a refactor">Analyse the architecture</button>
      <button class="toggle" id="fns" aria-pressed="false" title="List each module's costliest functions on the map" hidden>Functions</button>
      <button class="toggle" id="tests" aria-pressed="false" title="Show the tests and what they depend on">Tests</button>
      <div class="seg" role="group" aria-label="Theme">
        <button data-skin="mimir">MIMIR</button><button data-skin="light">Light</button>
      </div>
      <button class="icon" id="reload" title="Scan the workspace again: its files, the dependencies between them, the optimisation steps (no model call)" aria-label="Map the code again">⟳</button>
    </header>
    <div class="ribbon" aria-hidden="true"></div>
    <section class="session" id="session" aria-label="Optimisation session" hidden></section>
    <div class="tab"><div class="tab-map"><div class="lmap" id="map"></div><div class="legend" id="legend"></div></div>
      <div class="tab-side" id="side"></div></div>`;
  document.querySelectorAll<HTMLButtonElement>("button[data-skin]").forEach((b) => b.addEventListener("click", () => {
    skin = b.dataset.skin === "light" ? "light" : "mimir";
    remember();
    render();
  }));
  document.getElementById("reload")!.addEventListener("click", remap);
  picker.mount(document.getElementById("proxy")!);
  document.getElementById("side")!.addEventListener("click", onSideClick);
  document.getElementById("analyse")!.addEventListener("click", () => {
    // Pressed again while open: back to the form already written, not a blank one.
    if (draft && wholeCodeAnalysis(draft)) { openDraft(draft); return; }
    sel = null;
    openDraft({ ...newDraft([{ all: true }]), kind: "refactor" });
  });
  // A hidden tab maps the code once it is seen again, if something moved meanwhile.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && staleWhileHidden) { staleWhileHidden = false; remap(); }
  });
  document.getElementById("fns")!.addEventListener("click", () => {
    showFns = !showFns;
    remember();
    render();
  });
  // The columns follow the tab's width, each box widening to fill what is left over.
  let resizing = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizing);
    resizing = requestAnimationFrame(() => {
      const w = document.getElementById("map")!.clientWidth;
      if (!w) return;
      const c = Math.max(3, Math.min(14, Math.floor((w - LW - 4) / (BW + GX))));
      const bw = Math.max(BW, Math.floor((w - LW - 4) / c) - GX);
      if (c === cols && bw === boxW) return;
      cols = c; boxW = bw;
      // Only the map follows the width: the pane beside it stays as it is.
      if (map) { view = layoutOf(map); drawMap(); }
    });
  }).observe(document.getElementById("map")!);
  // The session's chart is drawn to its strip's width.
  let lastWidth = 0;
  new ResizeObserver(() => {
    const el = document.getElementById("session")!;
    if (el.hidden || Math.abs(el.clientWidth - lastWidth) < 4) return;
    lastWidth = el.clientWidth;
    renderSession();
  }).observe(document.getElementById("session")!);
  document.getElementById("tests")!.addEventListener("click", () => {
    showTests = !showTests;
    remember();
    if (raw) { map = visible(raw, showTests); view = layoutOf(map); }
    if (view && !stillThere(view, sel)) sel = null;
    render();
  });
  window.addEventListener("keydown", (e) => {
    // Typing in the request form moves nothing on the map.
    if ((e.target as Element | null)?.closest?.("input, textarea, select, #proxy")) return;
    if (e.key === "Escape" && sel) { sel = null; render(); }
    else if (session && e.key === "ArrowRight" && step < session.steps.length) { step++; render(); }
    else if (session && e.key === "ArrowLeft" && step > 0) { step--; render(); }
  });
  window.addEventListener("message", (ev: MessageEvent) => {
    const msg = ev.data as { type?: string; map?: CodeMap; session?: Session; focus?: string; reason?: string };
    // A run settled or a request was recorded: map the code again, once for a burst.
    if (msg?.type === "stale") {
      clearTimeout(staleTimer);
      staleTimer = setTimeout(() => {
        if (document.visibilityState === "visible") remap();
        else staleWhileHidden = true;
      }, STALE_SETTLE_MS);
      return;
    }
    // Opened from the drawer before its answer: the map is on its way, already asked.
    if (msg?.type === "mapping") { awaitMap(); return; }
    // The server had no map to give: said where the user is waiting for one.
    if (msg?.type === "unavailable") {
      if (mapping === "busy") settleMapping("failed", msg.reason || "the server gave no map: is the agent connected?");
      return;
    }
    if (msg?.type !== "map" || !msg.map) return;
    mergeState = msg.session?.merge ?? null;
    forecast = msg.session?.forecast ?? null;
    const took = picker.answer(msg.session);
    if (took === "again") { ask(); return; }
    if (took === "moved") remember();
    // A new answer opens on the session's latest step; one that adds no step, of the same
    // proxy, keeps the one shown.
    const steps = msg.session?.steps?.length ?? 0;
    if (!session || steps !== session.steps.length || picked(session) !== picked(msg.session)) step = steps;
    session = steps ? msg.session! : null;
    requests = msg.session?.requests ?? [];
    // A request the server has recorded is no longer only sent.
    const seen = new Set(requests.map((r) => r.token));
    if (sent.some((x) => seen.has(x.token))) { sent = sent.filter((x) => !seen.has(x.token)); remember(); }
    raw = msg.map;
    map = visible(raw, showTests);
    view = layoutOf(map);
    if (!stillThere(view, sel)) sel = null;
    // A component clicked in the drawer: the map opens on it.
    if (msg.focus && view.compOf.has(msg.focus)) {
      sel = { kind: "comp", id: msg.focus };
      scrollToBand = true;
    }
    // The answer the user waited for says it is in, even when the map is the same.
    if (mapping === "busy") settleMapping("idle");
    else mappedAt = new Date();
    render();
  });
  render();
  vscode.postMessage({ type: "ready" });
}

mount();
