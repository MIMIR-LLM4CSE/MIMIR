// The code map tab: the workspace's modules in bands, one per component, stacked by tier;
// the dependencies between them, coloured by what they are; the selection beside. It draws
// what the server answered (`shared.code_map`) and reads nothing else.

import { STYLE } from "./style";
import { edgeKey, overlay, stepCard, strip, type Overlay, type Scope, type Session } from "./session";

interface Component { id: string; name: string; tier: number; declared: string[]; external: Record<string, number>; tests: boolean }
interface Module { id: string; component: string }
interface File { id: string; lang: string; lines: number; module: string; component: string }
interface Edge {
  from: string; to: string; line: number; kind: string; via: string;
  sure: "exact" | "by_name"; deferred: boolean; breaks: string;
}
interface CodeMap {
  source_root: string; components: Component[]; modules: Module[]; files: File[];
  edges: Edge[]; cycles: string[][]; unresolved: Record<string, number>; truncated: boolean;
}
type Selection = { kind: "module" | "file"; id: string } | null;

declare function acquireVsCodeApi(): {
  postMessage(msg: unknown): void;
  getState(): { skin?: string; tests?: boolean } | undefined;
  setState(state: { skin?: string; tests?: boolean }): void;
};
const vscode = acquireVsCodeApi();

const NS = "http://www.w3.org/2000/svg";
const BW = 156, BH = 52, GX = 12, GY = 12, LW = 128, COLS = 5, PAD = 14, BGAP = 30;
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

function remember(): void { vscode.setState({ skin, tests: showTests }); }

/** The map as drawn: without its tests unless they are shown. */
function visible(m: CodeMap): CodeMap {
  if (showTests) return m;
  const hidden = new Set(m.components.filter((c) => c.tests).map((c) => c.id));
  if (!hidden.size) return m;
  const files = m.files.filter((f) => !hidden.has(f.component));
  const kept = new Set(files.map((f) => f.id));
  const modules = new Set(files.map((f) => f.module));
  return { ...m, files, components: m.components.filter((c) => !hidden.has(c.id)),
    modules: m.modules.filter((x) => modules.has(x.id)),
    edges: m.edges.filter((e) => kept.has(e.from) && kept.has(e.to)),
    cycles: m.cycles.filter((c) => c.every((id) => modules.has(id))) };
}

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  parent?.appendChild(e);
  return e;
}
function esc(s: unknown): string {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
}
const base = (p: string) => p.split("/").pop() ?? p;
const key = (a: string, b: string) => `${a}|${b}`;

/** What the map draws, derived once per answer: positions, module edges, lookups. */
interface View {
  byId: Map<string, File>;
  modFiles: Map<string, File[]>;
  compOf: Map<string, Component>;
  cycleOf: Map<string, number>;
  pos: Map<string, { x: number; y: number }>;
  bands: { c: Component; y: number; h: number }[];
  width: number; height: number;
  modEdges: ModEdge[];
}
interface ModEdge {
  from: string; to: string; edges: Edge[]; breaks: number; binding: number; byName: number;
  cycle: boolean; deferred: boolean; added: number; removed: boolean;
}

function derive(m: CodeMap): View {
  const byId = new Map(m.files.map((f) => [f.id, f]));
  const modFiles = new Map<string, File[]>();
  for (const f of m.files) (modFiles.get(f.module) ?? modFiles.set(f.module, []).get(f.module)!).push(f);
  const compOf = new Map(m.components.map((c) => [c.id, c]));
  const cycleOf = new Map<string, number>();
  m.cycles.forEach((c, i) => c.forEach((id) => cycleOf.set(id, i)));
  const size = (c: Component) => m.files.filter((f) => f.component === c.id).length;
  const order = [...m.components].sort((a, b) => b.tier - a.tier || size(b) - size(a));
  const pos = new Map<string, { x: number; y: number }>();
  const bands: View["bands"] = [];
  let y = 4;
  for (const c of order) {
    const mods = m.modules.filter((x) => x.component === c.id).map((x) => x.id).sort();
    if (!mods.length) continue;
    const rows = Math.ceil(mods.length / COLS), h = PAD * 2 + rows * BH + (rows - 1) * GY;
    bands.push({ c, y, h });
    mods.forEach((id, i) => pos.set(id, { x: LW + (i % COLS) * (BW + GX), y: y + PAD + Math.floor(i / COLS) * (BH + GY) }));
    y += h + BGAP;
  }
  const agg = new Map<string, ModEdge>();
  for (const e of m.edges) {
    const a = byId.get(e.from)!.module, b = byId.get(e.to)!.module;
    if (a === b) continue;
    let me = agg.get(key(a, b));
    if (!me) agg.set(key(a, b), me = { from: a, to: b, edges: [], breaks: 0, binding: 0, byName: 0, cycle: false, deferred: true, added: 0, removed: false });
    me.edges.push(e);
    if (e.breaks) me.breaks++;
    if (e.kind === "binding") me.binding++;
    if (e.sure === "by_name") me.byName++;
    me.deferred = me.deferred && e.deferred;
  }
  for (const me of agg.values()) {
    me.cycle = cycleOf.has(me.from) && cycleOf.get(me.from) === cycleOf.get(me.to);
  }
  return { byId, modFiles, compOf, cycleOf, pos, bands, width: LW + COLS * (BW + GX) + 4, height: y - BGAP + 4, modEdges: [...agg.values()] };
}

function path(v: View, a: string, b: string): string {
  const pa = v.pos.get(a)!, pb = v.pos.get(b)!, ax = pa.x + BW / 2, bx = pb.x + BW / 2;
  if (pb.y > pa.y + 4) { const y1 = pa.y + BH, y2 = pb.y, k = (y2 - y1) / 2; return `M${ax},${y1} C${ax},${y1 + k} ${bx},${y2 - k} ${bx},${y2}`; }
  if (pb.y < pa.y - 4) {
    const x1 = pa.x + BW, x2 = pb.x + BW, y1 = pa.y + BH / 2, y2 = pb.y + BH / 2, k = 36 + Math.abs(y2 - y1) * 0.15;
    return `M${x1},${y1} C${x1 + k},${y1} ${x2 + k},${y2} ${x2},${y2}`;
  }
  const lift = 16 + Math.abs(bx - ax) * 0.1;
  return `M${ax},${pa.y} C${ax},${pa.y - lift} ${bx},${pb.y - lift} ${bx},${pb.y}`;
}

/** The map's module edges with a step's marks: its added dependencies counted on the
 *  edges that hold them, its removed ones drawn where both modules still are. */
function marked(v: View, ov: Overlay | null): ModEdge[] {
  if (!ov) return v.modEdges;
  const out = v.modEdges.map((me) => ({ ...me, added: me.edges.filter((e) => ov.added.has(edgeKey(e))).length }));
  const gone = new Map<string, ModEdge>();
  for (const e of ov.removed) {
    const a = v.byId.get(e.from)?.module, b = v.byId.get(e.to)?.module;
    if (!a || !b || a === b || !v.pos.has(a) || !v.pos.has(b)) continue;
    const k = key(a, b);
    const me = gone.get(k) ?? { from: a, to: b, edges: [], breaks: 0, binding: 0, byName: 0, cycle: false,
      deferred: false, added: 0, removed: true };
    me.edges.push({ ...e, sure: e.sure === "by_name" ? "by_name" : "exact", deferred: false });
    gone.set(k, me);
  }
  return out.concat([...gone.values()]);
}

function edgeClass(me: ModEdge): string {
  if (me.removed) return "edge is-removed";
  if (me.added && me.added === me.edges.length) return "edge is-added";
  if (me.breaks) return "edge is-break";
  if (me.cycle) return "edge is-cycle";
  if (me.binding === me.edges.length) return me.byName === me.edges.length ? "edge is-binding is-byname" : "edge is-binding";
  return me.deferred ? "edge is-deferred" : "edge";
}

function selectedModule(v: View): string | null {
  if (!sel) return null;
  return sel.kind === "module" ? sel.id : v.byId.get(sel.id)?.module ?? null;
}

function renderMap(v: View, ov: Overlay | null): void {
  const root = document.getElementById("map")!;
  root.innerHTML = "";
  const s = svg("svg", { viewBox: `0 0 ${v.width} ${v.height}`, role: "img", "aria-label": "Modules by component, with the dependencies between them" }, root);
  s.style.cssText = `width:100%;min-width:720px;max-width:${v.width * 1.15}px;height:auto`;
  for (const b of v.bands) {
    svg("rect", { class: "band-bg", x: 0, y: b.y, width: v.width, height: b.h, rx: 6 }, s);
    svg("rect", { class: "band-bar", x: 12, y: b.y + 15, width: 3, height: 11, rx: 1.5 }, s);
    svg("text", { class: "band-name", x: 21, y: b.y + 25 }, s).textContent = b.c.name;
    svg("text", { class: "band-sub", x: 12, y: b.y + 42 }, s).textContent = `tier ${b.c.tier}`;
    const dl = b.c.declared.map(base);
    const t = svg("text", { class: "band-sub", x: 12, y: b.y + 57 }, s);
    t.textContent = dl.length ? (dl.length > 2 ? `→ ${dl.length} components` : `→ ${dl.join(", ")}`) : "";
    if (dl.length > 2) svg("title", {}, t).textContent = dl.join(", ");
  }
  const gE = svg("g", {}, s), gX = svg("g", {}, s), gM = svg("g", {}, s);
  const rank = (e: ModEdge) => (e.removed || e.added ? 4 : e.breaks ? 3 : e.cycle ? 2 : e.binding ? 1 : 0);
  const paths: SVGPathElement[] = [];
  for (const me of marked(v, ov).sort((a, b) => rank(a) - rank(b))) {
    const p = svg("path", { d: path(v, me.from, me.to), class: edgeClass(me),
      "stroke-width": (me.breaks || me.removed || me.added ? 2.5 : 1.2) + Math.log2(me.edges.length) * 0.6 }, gE);
    p.dataset.from = me.from; p.dataset.to = me.to;
    paths.push(p);
    if (me.breaks) {
      const mid = p.getPointAtLength(p.getTotalLength() / 2);
      svg("text", { class: "xmark", x: mid.x, y: mid.y + 4, "text-anchor": "middle" }, gX).textContent = "✕";
    }
  }
  const boxes = new Map<string, SVGGElement>();
  for (const [id, files] of v.modFiles) {
    const p = v.pos.get(id);
    if (!p) continue;
    const g = svg("g", { class: "m", transform: `translate(${p.x},${p.y})`, tabindex: 0, role: "button", "aria-label": id }, gM);
    svg("rect", { class: "box", width: BW, height: BH, rx: 6 }, g);
    const comp = files[0].component;
    const nm = svg("text", { class: "mname", x: 9, y: 18 }, g);
    nm.textContent = id === comp ? base(id) : id.slice(comp.length + 1);
    while (nm.getComputedTextLength() > BW - 30 && (nm.textContent ?? "").length > 4) nm.textContent = "…" + nm.textContent!.slice(2);
    const lines = files.reduce((n, f) => n + f.lines, 0);
    const langs = [...new Set(files.map((f) => LANG[f.lang] ?? f.lang))].join(" ");
    svg("text", { class: "msub", x: 9, y: 35 }, g).textContent = `${files.length} files · ${(lines / 1000).toFixed(1)}k · ${langs}`;
    let dx = BW - 9;
    if (map!.edges.some((e) => e.breaks && v.byId.get(e.from)!.module === id)) { svg("circle", { class: "dot-break", cx: dx, cy: 12, r: 4 }, g); dx -= 11; }
    if (v.cycleOf.has(id)) svg("circle", { class: "dot-cycle", cx: dx, cy: 12, r: 4 }, g);
    if (ov) {
      if (files.some((f) => ov.touchedNow.has(f.id))) g.classList.add("is-touched");
      else if (files.some((f) => ov.touched.has(f.id))) g.classList.add("is-touched-old");
      const ids = new Set(files.map((f) => f.id));
      ov.steps.filter((st) => st.changed.some((f) => ids.has(f))).slice(-2).forEach((st, i) => {
        const pl = svg("g", { class: "pill", transform: `translate(${BW - 52 - i * 54},-8)` }, g);
        svg("rect", { class: st.verdict === "accept" ? "ok" : "no", width: 48, height: 15, rx: 7.5 }, pl);
        svg("text", { x: 24, y: 11, "text-anchor": "middle" }, pl).textContent = "#" + st.n + " " + (st.delta === null ? "" : signedShort(st.delta));
        svg("title", {}, pl).textContent = "#" + st.n + " " + st.axis;
      });
    }
    svg("title", {}, g).textContent = id;
    boxes.set(id, g);
    const pick = () => { sel = sel?.kind === "module" && sel.id === id ? null : { kind: "module", id }; render(); };
    g.addEventListener("click", pick);
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
    g.addEventListener("mouseenter", () => highlight(id));
    g.addEventListener("mouseleave", () => highlight(selectedModule(v)));
  }
  function highlight(id: string | null): void {
    const near = new Set(id ? [id] : []);
    for (const p of paths) {
      const on = !!id && (p.dataset.from === id || p.dataset.to === id);
      p.classList.toggle("is-hot", on && !/is-(break|cycle)/.test(p.getAttribute("class") ?? ""));
      p.classList.toggle("is-faded", !!id && !on);
      if (on) { near.add(p.dataset.from!); near.add(p.dataset.to!); }
    }
    for (const [mid, g] of boxes) {
      g.classList.toggle("is-dim", !!id && !near.has(mid));
      g.classList.toggle("is-sel", selectedModule(v) === mid);
    }
  }
  highlight(selectedModule(v));
}

/** A file in a list: its name selects it in the map, ↗ opens it at *line*. */
function fileRow(id: string, right: string, line?: number): string {
  const where = line ? " at line " + line : "";
  return '<div class="row"><button class="go" data-file="' + esc(id) + '" title="' + esc(id) + '">'
    + esc(base(id)) + "</button><span>" + esc(right) + '</span><button class="ext" data-open="'
    + esc(id) + '" data-line="' + (line ?? "") + '" title="Open ' + esc(base(id)) + where
    + '" aria-label="Open ' + esc(base(id)) + '">↗</button></div>';
}
/** A module in a list, selecting it in the map. */
function moduleRow(id: string, right: string | number = ""): string {
  return '<div class="row"><button class="go" data-module="' + esc(id) + '" title="' + esc(id) + '">'
    + esc(id) + "</button><span>" + esc(right) + "</span></div>";
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

function moduleDetail(m: CodeMap, v: View, id: string): string {
  const files = [...(v.modFiles.get(id) ?? [])].sort((a, b) => b.lines - a.lines);
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
  return BACK + section("sel", base(id), '<div class="path">' + esc(id) + "</div>" + cycle
    + '<div class="cap">Files</div>' + list(files.map((f) => fileRow(f.id, f.lines + " l")))
    + '<div class="cap">Depends on</div>' + list(rows(out))
    + '<div class="cap">Used by</div>' + list(rows(inn)));
}

function fileDetail(m: CodeMap, v: View, id: string): string {
  const f = v.byId.get(id)!;
  const out = m.edges.filter((e) => e.from === f.id), inn = m.edges.filter((e) => e.to === f.id);
  return BACK + section("sel", base(f.id), '<div class="path">' + esc(f.id) + "</div>"
    + line("module", '<button class="link" data-module="' + esc(f.module) + '">' + esc(f.module) + "</button>")
    + line("language", esc(f.lang)) + line("lines", f.lines.toLocaleString("en"))
    + '<div class="cap">Depends on (' + out.length + ")</div>" + list(out.map((e) => fileRow(e.to, edgeRight(e))))
    + '<div class="cap">Used by (' + inn.length + ")</div>" + list(inn.map((e) => fileRow(e.from, edgeRight(e), e.line)))
    + '<button class="open" data-open="' + esc(f.id) + '" data-line="">Open ' + esc(base(f.id)) + "</button>");
}

function renderSide(v: View): void {
  const m = map!, side = document.getElementById("side")!;
  const card = session && step ? stepCard(session, step, fileRow) : "";
  const html = !sel ? card + overview(m) : sel.kind === "module" ? moduleDetail(m, v, sel.id) : fileDetail(m, v, sel.id);
  side.innerHTML = html;
  document.getElementById("back")?.addEventListener("click", () => { sel = null; render(); });
  side.querySelectorAll<HTMLElement>("[data-module]").forEach((b) => b.addEventListener("click", () => { sel = { kind: "module", id: b.dataset.module! }; render(); }));
  side.querySelectorAll<HTMLElement>("[data-file]").forEach((b) => b.addEventListener("click", () => { sel = { kind: "file", id: b.dataset.file! }; render(); }));
  side.querySelectorAll<HTMLElement>("[data-open]").forEach((b) => b.addEventListener("click", () => {
    const line = b.dataset.line ? Number(b.dataset.line) : undefined;
    vscode.postMessage({ type: "open_file", file: b.dataset.open!, line });
  }));
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
    + (session && step ? stepLegend() : "");
}

function stepLegend(): string {
  const who = scope === "step" ? "this step" : "the kept steps";
  return '<span><i class="sw" style="border-color:var(--e-added);border-top-width:3px"></i>added by ' + who + "</span>"
    + '<span><i class="sw dash" style="border-color:var(--edge);border-top-width:3px"></i>removed by ' + who + "</span>"
    + "<span>green outline: the files " + who + " changed</span>"
    + "<span>the dependencies drawn are today's code</span>";
}

/** ±12% in the width of a pill. */
function signedShort(d: number): string {
  return (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d * 100).toFixed(0) + "%";
}

function renderSession(): void {
  const el = document.getElementById("session")!;
  el.hidden = !session || !session.steps.length;
  if (el.hidden || !session) return;
  el.innerHTML = strip(session, step, scope);
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

function render(): void {
  document.documentElement.dataset.skin = skin;
  document.querySelectorAll<HTMLButtonElement>("button[data-skin]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.skin === skin)));
  document.getElementById("tests")?.setAttribute("aria-pressed", String(showTests));
  if (!map || !view) return;
  const meta = document.getElementById("meta")!;
  const hiddenTests = raw ? raw.files.length - map.files.length : 0;
  meta.textContent = `${map.files.length} files · ${map.modules.length} modules`
    + (hiddenTests ? ` · ${hiddenTests} test files folded` : "");
  if (!map.files.length) {
    document.getElementById("map")!.innerHTML = '<p class="empty">No Python, C/C++ or Fortran code in this workspace.</p>';
    document.getElementById("side")!.innerHTML = "";
    return;
  }
  renderSession();
  renderMap(view, session && step ? overlay(session, step, scope) : null);
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
      <button class="toggle push" id="tests" aria-pressed="false" title="Show the tests and what they depend on">Tests</button>
      <div class="seg" role="group" aria-label="Theme">
        <button data-skin="mimir">MIMIR</button><button data-skin="light">Light</button>
      </div>
      <button class="icon" id="reload" title="Map the code again" aria-label="Map the code again">⟳</button>
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
  document.getElementById("reload")!.addEventListener("click", () => vscode.postMessage({ type: "refresh" }));
  document.getElementById("tests")!.addEventListener("click", () => {
    showTests = !showTests;
    remember();
    if (raw) { map = visible(raw); view = derive(map); }
    if (sel && view && !(sel.kind === "module" ? view.modFiles.has(sel.id) : view.byId.has(sel.id))) sel = null;
    render();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && sel) { sel = null; render(); }
    else if (session && e.key === "ArrowRight" && step < session.steps.length) { step++; render(); }
    else if (session && e.key === "ArrowLeft" && step > 0) { step--; render(); }
  });
  window.addEventListener("message", (ev: MessageEvent) => {
    const msg = ev.data as { type?: string; map?: CodeMap; session?: Session };
    if (msg?.type !== "map" || !msg.map) return;
    // A new answer opens on the session's latest step; one that adds no step keeps the one shown.
    const steps = msg.session?.steps?.length ?? 0;
    if (!session || steps !== session.steps.length) step = steps;
    session = steps ? msg.session! : null;
    raw = msg.map;
    map = visible(raw);
    view = derive(map);
    if (sel && !(sel.kind === "module" ? view.modFiles.has(sel.id) : view.byId.has(sel.id))) sel = null;
    render();
  });
  render();
  vscode.postMessage({ type: "ready" });
}

mount();
