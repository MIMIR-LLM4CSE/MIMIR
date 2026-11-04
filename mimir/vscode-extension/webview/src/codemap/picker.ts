// The proxy picker in the map's header: one line per proxy with a session (where it ran,
// how many runs, its gain so far), grouped as running, merges and frozen; a click shows one,
// ticks show several together, and the ticked ones can be merged.

import { esc } from "./session";

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
  const pct = Math.abs(g * 100).toFixed(Math.abs(g) < 0.1 ? 1 : 0) + "%";
  return g > 0 ? pct + " better" : g < 0 ? pct + " worse" : "even";
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
