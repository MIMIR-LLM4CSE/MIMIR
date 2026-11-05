// The text helpers every part of the tab draws with: HTML escaping and the percentages.

export function esc(s: unknown): string {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
}

/** |x| as a percentage, a decimal under 10%: "12%", "3.4%". */
export function pct(x: number): string {
  return Math.abs(x * 100).toFixed(Math.abs(x) < 0.1 ? 1 : 0) + "%";
}

/** A change with its sign: "+12%", "−3.4%", "±0.0%"; *short* drops the decimal, for a pill. */
export function signed(d: number | null, short = false): string {
  if (d === null || !isFinite(d)) return "";
  return (d > 0 ? "+" : d < 0 ? "−" : "±") + (short ? Math.abs(d * 100).toFixed(0) + "%" : pct(d));
}

/** A path's last part: its file's name. */
export const basename = (p: string) => p.slice(p.lastIndexOf("/") + 1);
