// The proxy picker in the map's header: one line per proxy with a session (where it ran,
// how many runs, its gain so far), grouped as running, merges and frozen; a click shows one,
// ticks show several together, and the ticked ones can be merged.

import { esc, pct } from "./format";
import type { Session } from "./session";

/** One proxy as the server describes it for the picker. */
export interface ProxyMeta {
  name: string; branch: string; metric: string; goal: string; steps: number;
  /** 1 − best/baseline whatever the goal: 0.2 is a fifth better. */
  gain: number | null;
  active: boolean;
  /** Why it was frozen (cleaned, rebaselined), empty while it runs. */
  frozen: string;
  merge: boolean; sources: string[];
  /** Whether it kept an accepted run a merge can take. */
  mergeable?: boolean;
}

/** Whose steps the map shows: "" the active session's, a proxy's name, ALL for every
 *  proxy, or the ticked ones joined by commas. */
export const ALL = "*";

/** The proxies a pick ticks together; none for one proxy or every proxy. */
export function ticked(pick: string): string[] {
  return pick.includes(",") ? pick.split(",").filter(Boolean) : [];
}

/** What the picker's button says for *pick*. */
export function pickLabel(pick: string, active: string): string {
  if (pick === ALL) return "Every proxy";
  const t = ticked(pick);
  if (t.length) return t.length > 3 ? t.length + " proxies" : t.join(" + ");
  return pick || active || "active session";
}

/** A gain as the picker shows it: "20% better", "3% worse". */
export function gainText(g: number | null): string {
  if (g === null || !isFinite(g)) return "";
  return g > 0 ? pct(g) + " better" : g < 0 ? pct(g) + " worse" : "even";
}

/** The ticked proxies a merge can take: live, not merges themselves, with an accepted run. */
export function mergeable(meta: ProxyMeta[], names: string[]): string[] {
  const ok = new Set(meta.filter((m) => m.mergeable && !m.merge && !m.frozen).map((m) => m.name));
  return names.filter((n) => ok.has(n));
}

/** What the "Merge" button puts to the agent. */
export function mergePrompt(names: string[]): string {
  const list = names.map((n) => "'" + n + "'").join(", ");
  return "Merge the kept optimisations of the proxies " + names.join(", ") + " and measure the gain of the whole: "
    + "proxy_merge(op='plan', proxies=[" + list + "]), then follow the proxy-merge skill. The merge is measured "
    + "in a sub-agent's copy of its own; my workspace stays as it is until I merge its branch.";
}

function row(m: ProxyMeta, pick: string, ticks: Set<string>): string {
  const value = m.active ? "" : m.name;
  const cur = pick === value;
  const badges = (m.active ? '<span class="badge on">active</span>' : "")
    + (m.merge ? '<span class="badge merge" title="Merges ' + esc(m.sources.join(", ")) + '">merge</span>' : "")
    + (m.frozen ? '<span class="badge">' + esc(m.frozen) + "</span>" : "");
  const g = m.gain;
  return '<div class="pick-row' + (cur ? " is-cur" : "") + '">'
    + '<input type="checkbox" data-tick="' + esc(m.name) + '"' + (ticks.has(m.name) ? " checked" : "")
    + ' aria-label="Show ' + esc(m.name) + ' with the others ticked">'
    + '<button class="pick-name" data-pick="' + esc(value) + '" aria-current="' + cur + '" title="Show '
    + esc(m.name) + "'s steps alone" + '">' + esc(m.name) + "</button>"
    + badges
    + (m.branch ? '<span class="pick-branch" title="Branch">' + esc(m.branch) + "</span>" : "")
    + '<span class="pick-steps">' + m.steps + (m.steps === 1 ? " run" : " runs") + "</span>"
    + '<span class="pick-gain' + (g === null ? "" : g > 0 ? " good" : g < 0 ? " bad" : "") + '" title="Best against its baseline, '
    + esc(m.metric) + '">' + esc(gainText(g) || "–") + "</span></div>";
}

/** The picker: its button and, when *open*, the list. */
export function pickerHtml(meta: ProxyMeta[], pick: string, active: string, open: boolean, ticks: Set<string>): string {
  const groups: [string, ProxyMeta[]][] = [
    ["Running", meta.filter((m) => !m.frozen && !m.merge)],
    ["Merges", meta.filter((m) => !m.frozen && m.merge)],
    ["Frozen", meta.filter((m) => m.frozen)],
  ];
  const ready = mergeable(meta, [...ticks]);
  const list = groups.filter(([, ms]) => ms.length).map(([title, ms]) =>
    '<div class="pick-group">' + esc(title) + "</div>" + ms.map((m) => row(m, pick, ticks)).join("")).join("");
  return '<button class="pick-btn" aria-haspopup="true" aria-expanded="' + open + '" title="Whose optimisation steps the map shows">'
    + '<span class="pick-label">' + esc(pickLabel(pick, active)) + '</span><span class="caret">▾</span></button>'
    + '<div class="pick-panel" role="group" aria-label="Proxies"' + (open ? "" : " hidden") + ">"
    + '<button class="pick-name pick-all' + (pick === ALL ? " is-cur" : "") + '" data-pick="' + ALL + '">Every proxy, together'
    + '<span class="hint">each against its own baseline</span></button>'
    + list
    + '<div class="pick-foot"><span class="hint">' + (ticks.size > 1 ? ticks.size + " ticked, drawn together" : "Tick two or more to draw them together")
    + '</span><button class="open" data-merge' + (ready.length > 1 ? "" : " disabled")
    + ' title="Ask the agent to merge them in a copy of its own and measure the whole">Merge' + (ready.length > 1 ? " " + ready.length : "")
    + "</button></div></div>";
}

/** What the picker reads of an answer. */
export type Answer = Pick<Session, "proxy" | "active" | "combined" | "selected" | "proxies" | "proxies_meta">;

/** Whose steps an answer holds, written as a pick writes it. */
export function picked(s: Partial<Answer> | undefined): string {
  if (s?.selected && s.selected.length > 1) return s.selected.join(",");
  if (s?.combined) return ALL;
  return !s?.proxy || s.proxy === s.active ? "" : s.proxy;
}

/** *p* as the server can answer it, offering *proxies*: the active proxy named is the
 *  active session, a proxy it no longer has is dropped. */
export function reachable(p: string, proxies: string[], active: string): string {
  if (p === ALL) return proxies.length > 1 ? ALL : "";
  const t = ticked(p);
  if (t.length) {
    const there = t.filter((x) => proxies.includes(x));
    return there.length > 1 ? there.join(",") : reachable(there[0] ?? "", proxies, active);
  }
  return p && p !== active && proxies.includes(p) ? p : "";
}

/** What the tab is told when the pick moves (to ask the server for it) or Merge is pressed. */
export interface PickerHost { choose(): void; prompt(text: string): void }

/** The picker in the map's header: the pick, the proxies offered, the list open or not and
 *  the proxies ticked in it (two or more are the pick). */
export class Picker {
  private el: HTMLElement | null = null;
  private open = false;
  private ticks: Set<string>;
  private proxies: string[] = [];
  private meta: ProxyMeta[] = [];
  private active = "";
  /** The pick last asked again for an answer that did not hold it, until one does. */
  private reasked: string | null = null;

  constructor(public pick: string, private host: PickerHost) {
    this.ticks = new Set(ticked(pick));
  }

  /** What to ask the server for the pick's steps. */
  query(): { proxies?: string[]; combined?: boolean; proxy?: string } {
    const t = ticked(this.pick);
    return t.length ? { proxies: t } : this.pick === ALL ? { combined: true } : this.pick ? { proxy: this.pick } : {};
  }

  /** Takes an answer: "again" when it holds other steps than the ones picked (the drawer
   *  opens the active session's) and they are to be asked again, once; "moved" when the
   *  pick follows what it holds. */
  answer(s: Answer | undefined): "again" | "moved" | "same" {
    this.proxies = s?.proxies ?? [];
    this.meta = s?.proxies_meta ?? [];
    this.active = s?.active ?? "";
    const want = reachable(this.pick, this.proxies, this.active), got = picked(s);
    // Once: an answer that still differs is what the server can give, and is drawn.
    if (want !== got && this.reasked !== want) { this.pick = this.reasked = want; return "again"; }
    this.reasked = null;
    const moved = this.pick !== got;
    this.pick = got;
    // The ticks follow the pick the answer holds (a single tick is that proxy alone), and
    // only name proxies it still offers.
    if (ticked(got).length) this.ticks = new Set(ticked(got));
    else if (this.ticks.size > 1) this.ticks = new Set();
    this.ticks = new Set([...this.ticks].filter((t) => this.proxies.includes(t)));
    return moved ? "moved" : "same";
  }

  mount(el: HTMLElement): void {
    this.el = el;
    el.addEventListener("click", (e) => this.onClick(e));
    el.addEventListener("change", (e) => this.onTick(e));
    el.addEventListener("keydown", (e) => this.onKey(e));
    // A click anywhere else closes the list.
    document.addEventListener("click", (e) => {
      if (this.open && !el.contains(e.target as Node | null)) this.close();
    });
  }

  render(): void {
    const el = this.el;
    if (!el) return;
    el.hidden = this.proxies.length < 2;
    if (el.hidden) { this.open = false; return; }
    // Rebuilt with each answer: the focus goes back to the control that had it.
    const had = document.activeElement instanceof HTMLElement && el.contains(document.activeElement) ? document.activeElement : null;
    const again = had?.dataset.tick !== undefined ? '[data-tick="' + had.dataset.tick + '"]'
      : had?.dataset.pick !== undefined ? '[data-pick="' + had.dataset.pick + '"]' : had ? ".pick-btn" : "";
    el.innerHTML = pickerHtml(this.meta, reachable(this.pick, this.proxies, this.active), this.active, this.open, this.ticks);
    if (again) el.querySelector<HTMLElement>(again)?.focus();
  }

  private close(): void {
    this.open = false;
    this.render();
  }

  /** Show *p*'s steps, the list closed or not. */
  private choose(p: string, close: boolean): void {
    this.pick = p;
    if (close) this.open = false;
    this.host.choose();
  }

  private onClick(e: Event): void {
    // The list is rebuilt under the click: the document's "click outside" must not see it.
    e.stopPropagation();
    const t = e.target as HTMLElement;
    if (t.closest(".pick-btn")) {
      this.open = !this.open;
      this.render();
      if (this.open) this.el?.querySelector<HTMLElement>(".pick-panel button, .pick-panel input")?.focus();
      return;
    }
    const name = t.closest<HTMLElement>("[data-pick]");
    if (name) { this.ticks = new Set(); this.choose(name.dataset.pick ?? "", true); return; }
    if (t.closest("[data-merge]")) {
      const names = mergeable(this.meta, [...this.ticks]);
      if (names.length < 2) return;
      this.close();
      this.host.prompt(mergePrompt(names));
    }
  }

  private onTick(e: Event): void {
    const box = e.target as HTMLInputElement;
    const name = box.dataset.tick;
    if (name === undefined) return;
    if (box.checked) this.ticks.add(name); else this.ticks.delete(name);
    // Two or more are drawn together; one alone is that proxy's steps.
    const t = [...this.ticks];
    if (t.length > 1) this.choose(t.join(","), false);
    else if (t.length === 1) this.choose(this.meta.find((m) => m.name === t[0])?.active ? "" : t[0], false);
    // At once, without waiting for the answer: the button and the Merge count follow the ticks.
    this.render();
  }

  /** Escape closes the list, the arrows move between its lines. */
  private onKey(e: KeyboardEvent): void {
    if (e.key === "Escape" && this.open) {
      e.stopPropagation();
      this.close();
      this.el?.querySelector<HTMLElement>(".pick-btn")?.focus();
      return;
    }
    if (!this.open || (e.key !== "ArrowDown" && e.key !== "ArrowUp")) return;
    e.preventDefault();
    const lines = Array.from(this.el?.querySelectorAll<HTMLElement>(".pick-name, [data-merge]:not([disabled])") ?? []);
    // From a tick, the line it is on.
    const focused = document.activeElement as HTMLElement | null;
    const here = focused?.closest(".pick-row")?.querySelector<HTMLElement>(".pick-name") ?? focused;
    const at = here ? lines.indexOf(here) : -1;
    lines[(at + (e.key === "ArrowDown" ? 1 : lines.length - 1)) % lines.length]?.focus();
  }
}
