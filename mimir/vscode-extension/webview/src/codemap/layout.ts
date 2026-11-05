// Where the map puts things: the modules in bands, one per component, stacked by tier,
// and the module edges as curves. Pure: index.ts draws what this lays out.

import { edgeKey, type Overlay, type TimeView } from "./session";
import type { CodeMap, Component, Edge, File } from "./types";

// A box is BW wide at least, and widens to fill the tab; it grows by FN_ROW a function listed.
export const BW = 156, BH = 52, GX = 12, GY = 12, LW = 128, PAD = 14, BGAP = 30, FN_N = 3, FN_ROW = 14;

const key = (a: string, b: string) => `${a}|${b}`;

/** The map as drawn: without its tests unless they are shown. */
export function visible(m: CodeMap, showTests: boolean): CodeMap {
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

/** What the map draws, derived once per answer: positions, module edges, lookups. */
export interface View {
  byId: Map<string, File>;
  modFiles: Map<string, File[]>;
  compOf: Map<string, Component>;
  cycleOf: Map<string, number>;
  pos: Map<string, { x: number; y: number }>;
  bands: { c: Component; y: number; h: number }[];
  width: number; height: number;
  /** A module box's size: wider with the tab, taller with its functions listed. */
  bw: number; bh: number;
  modEdges: ModEdge[];
  /** The modules a rule break starts from. */
  breakMods: Set<string>;
}
export interface ModEdge {
  from: string; to: string; edges: Edge[]; breaks: number; binding: number; byName: number;
  cycle: boolean; deferred: boolean; added: number; removed: boolean;
}

/** How tall a box's head is: its name and size, and one bar or two. */
export function barsHeight(t: TimeView | null): number {
  return t?.gpuFiles && !t.cpuUnavailable ? BH + 11 : BH;
}

/** Where every module sits, *cols* boxes *bw* × *bh* to a row, one band per component
 *  stacked by tier, and the module edges with what they carry. */
export function derive(m: CodeMap, COLS: number, bw: number, bh: number): View {
  const byId = new Map(m.files.map((f) => [f.id, f]));
  const modFiles = new Map<string, File[]>();
  for (const f of m.files) (modFiles.get(f.module) ?? modFiles.set(f.module, []).get(f.module)!).push(f);
  const compOf = new Map(m.components.map((c) => [c.id, c]));
  const cycleOf = new Map<string, number>();
  m.cycles.forEach((c, i) => c.forEach((id) => cycleOf.set(id, i)));
  const sizes = new Map<string, number>();
  for (const f of m.files) sizes.set(f.component, (sizes.get(f.component) ?? 0) + 1);
  const compMods = new Map<string, string[]>();
  for (const x of m.modules) (compMods.get(x.component) ?? compMods.set(x.component, []).get(x.component)!).push(x.id);
  const size = (c: Component) => sizes.get(c.id) ?? 0;
  const order = [...m.components].sort((a, b) => b.tier - a.tier || size(b) - size(a));
  const pos = new Map<string, { x: number; y: number }>();
  const bands: View["bands"] = [];
  let y = 4;
  for (const c of order) {
    const mods = (compMods.get(c.id) ?? []).sort();
    if (!mods.length) continue;
    const rows = Math.ceil(mods.length / COLS), h = PAD * 2 + rows * bh + (rows - 1) * GY;
    bands.push({ c, y, h });
    mods.forEach((id, i) => pos.set(id, { x: LW + (i % COLS) * (bw + GX), y: y + PAD + Math.floor(i / COLS) * (bh + GY) }));
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
  const breakMods = new Set<string>();
  for (const me of agg.values()) {
    me.cycle = cycleOf.has(me.from) && cycleOf.get(me.from) === cycleOf.get(me.to);
  }
  for (const e of m.edges) if (e.breaks) breakMods.add(byId.get(e.from)!.module);
  return { byId, modFiles, compOf, cycleOf, pos, bands, width: LW + COLS * (bw + GX) + 4, height: y - BGAP + 4,
    bw, bh, modEdges: [...agg.values()], breakMods };
}

/** The edge from module *a* to *b* as a cubic: start, two controls, end. */
export function curve(v: View, a: string, b: string): number[] {
  const BW = v.bw, BH = v.bh;
  const pa = v.pos.get(a)!, pb = v.pos.get(b)!, ax = pa.x + BW / 2, bx = pb.x + BW / 2;
  if (pb.y > pa.y + 4) { const y1 = pa.y + BH, y2 = pb.y, k = (y2 - y1) / 2; return [ax, y1, ax, y1 + k, bx, y2 - k, bx, y2]; }
  if (pb.y < pa.y - 4) {
    const x1 = pa.x + BW, x2 = pb.x + BW, y1 = pa.y + BH / 2, y2 = pb.y + BH / 2, k = 36 + Math.abs(y2 - y1) * 0.15;
    return [x1, y1, x1 + k, y1, x2 + k, y2, x2, y2];
  }
  const lift = 16 + Math.abs(bx - ax) * 0.1;
  return [ax, pa.y, ax, pa.y - lift, bx, pb.y - lift, bx, pb.y];
}
export function pathOf(c: number[]): string {
  return `M${c[0]},${c[1]} C${c[2]},${c[3]} ${c[4]},${c[5]} ${c[6]},${c[7]}`;
}
/** A cubic's middle, computed: measuring the drawn path would lay the page out once per edge. */
export function midOf(c: number[]): { x: number; y: number } {
  return { x: (c[0] + 3 * c[2] + 3 * c[4] + c[6]) / 8, y: (c[1] + 3 * c[3] + 3 * c[5] + c[7]) / 8 };
}

/** The map's module edges with a step's marks: its added dependencies counted on the
 *  edges that hold them, its removed ones drawn where both modules still are. */
export function marked(v: View, ov: Overlay | null): ModEdge[] {
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

export function edgeClass(me: ModEdge): string {
  if (me.removed) return "edge is-removed";
  if (me.added && me.added === me.edges.length) return "edge is-added";
  if (me.breaks) return "edge is-break";
  if (me.cycle) return "edge is-cycle";
  if (me.binding === me.edges.length) return me.byName === me.edges.length ? "edge is-binding is-byname" : "edge is-binding";
  return me.deferred ? "edge is-deferred" : "edge";
}
