/**
 * How wide a side drawer is, and where that answer is kept.
 *
 * What a drawer shows (a scheduler next to its hardware, a sub-agent's task) is often
 * wider than a fixed width, so the reader sets it, and it is remembered.
 */

/** Narrow enough to park a drawer out of the way, wide enough to stay readable. */
export const MIN_PANEL_WIDTH = 180;
/** The chat must not be squeezed out of its own window by a drawer. */
const MAX_PANEL_FRACTION = 0.8;

export function clampPanelWidth(px: number, viewportWidth: number): number {
  const ceiling = Math.max(MIN_PANEL_WIDTH, Math.round(viewportWidth * MAX_PANEL_FRACTION));
  if (!Number.isFinite(px)) return MIN_PANEL_WIDTH;
  return Math.min(ceiling, Math.max(MIN_PANEL_WIDTH, Math.round(px)));
}

const KEY_PREFIX = "mimir.panelWidth.";

/**
 * The stored width for *key*, or *fallback*.
 *
 * Storage can be absent or throw (a webview with site data blocked), and what it holds
 * was written by an older build — so anything unreadable is the fallback rather than an
 * error, and a stored value is clamped on the way out: a width saved on a wide monitor
 * must not open a drawer over the whole of a narrow one.
 */
export function readPanelWidth(key: string, fallback: number, viewportWidth: number): number {
  try {
    const held = window.localStorage.getItem(KEY_PREFIX + key);
    const px = held === null ? NaN : Number(held);
    return clampPanelWidth(Number.isFinite(px) ? px : fallback, viewportWidth);
  } catch {
    return clampPanelWidth(fallback, viewportWidth);
  }
}

export function storePanelWidth(key: string, px: number): void {
  try {
    window.localStorage.setItem(KEY_PREFIX + key, String(Math.round(px)));
  } catch {
    /* A width nobody can save is still a width that works for this window. */
  }
}
