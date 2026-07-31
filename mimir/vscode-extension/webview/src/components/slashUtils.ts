// Pure helpers for the "/"-command autocomplete in the chat input.
//
// Slash commands invoke a skill explicitly (e.g. "/fix-bug …"). The backend only
// honours a slash when it is the very first non-space character of the query
// (see agent_loop.py: `query.strip().startswith("/")`), so — unlike "@" mentions —
// the dropdown opens ONLY at the start of the input. Kept free of React/DOM so the
// caret/token logic is unit-testable with vitest (see slashUtils.test.ts).

import type { ToggleItem } from "../types";

export interface SlashQuery {
  /** Index of the "/" that opens the command (after any leading whitespace). */
  start: number;
  /** The partial command name typed after "/" (may be empty right after "/"). */
  query: string;
}

/**
 * If the caret sits inside a leading "/"-command being typed, return its start index
 * and the partial query; otherwise null. A slash only counts when it is the first
 * non-whitespace character of the input and the caret has not moved past the first
 * word (skill names never contain spaces). `caret` is the textarea selectionStart.
 */
export function detectSlashQuery(text: string, caret: number): SlashQuery | null {
  if (caret < 0 || caret > text.length) return null;

  // Find the first non-whitespace character; it must be the "/".
  const start = text.search(/\S/);
  if (start < 0 || text[start] !== "/") return null;

  // The caret must be at or after the slash, and everything from "/"+1 up to the
  // caret is the query — which must be a single word (no whitespace).
  if (caret <= start) return null;
  const query = text.slice(start + 1, caret);
  if (/\s/.test(query)) return null;
  return { start, query };
}

/** Case-insensitive filter of skills by name / description. */
export function filterSkills(skills: ToggleItem[], query: string): ToggleItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return skills;
  return skills.filter(
    (s) =>
      s.name.toLowerCase().includes(q) ||
      (s.description || "").toLowerCase().includes(q)
  );
}

export interface ApplySlashResult {
  text: string;
  /** New caret position (just past the inserted token + trailing space). */
  caret: number;
}

/**
 * Replace the "/partial" spanning [start, caret) with "/name " and return the new
 * text and caret. Preserves any leading whitespace before the slash.
 */
export function applySlash(
  text: string,
  start: number,
  caret: number,
  name: string
): ApplySlashResult {
  const before = text.slice(0, start);
  const after = text.slice(caret);
  const token = `/${name} `;
  return { text: `${before}${token}${after}`, caret: before.length + token.length };
}
