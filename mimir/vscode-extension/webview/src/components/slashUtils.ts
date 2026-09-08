// Pure helpers for the "/"-command autocomplete in the chat input.
//
// TWO KINDS OF SLASH share the input, and they travel differently:
//
//  * a SKILL ("/fix-bug …") is part of the query. It is sent as `type: "query"` and the
//    backend honours it only when the slash is the first non-space character
//    (agent_loop.py: `query.strip().startswith("/")`).
//  * a SESSION COMMAND ("/mode agent", "/memory clear") never reaches the model. It is
//    handled by ws_session._handle_command and must be sent as `type: "command"`.
//
// Until the two were listed together, only skills appeared in the dropdown and typed
// session commands went to the model as ordinary text — so a command like
// "/proxy clean foo" was silently a request for the model to do it rather than an
// instruction to the session. Both are offered here, and `isSessionCommand` is what
// routes them apart at send time.
//
// The dropdown opens ONLY at the start of the input, unlike "@" mentions. Kept free of
// React/DOM so the caret/token logic is unit-testable with vitest.

import type { ToggleItem } from "../types";

/**
 * Session commands the backend handles itself (ws_session._handle_command).
 *
 * Mirrored by hand rather than fetched: they are few and stable, and the alternative is
 * a round-trip before the first keystroke can be completed. `test_slash_commands.py`
 * asserts every name here is actually handled, so the list cannot drift into
 * advertising a command that does nothing.
 */
export const SESSION_COMMANDS: ToggleItem[] = [
  { name: "mode", description: "agent | plan | ask — switch the agent mode", enabled: true },
  { name: "context", description: "compact | full — how much context to send", enabled: true },
  { name: "thinking", description: "on | off — reasoning", enabled: true },
  { name: "thinking-depth", description: "0-4 — how much reasoning", enabled: true },
  { name: "streaming", description: "on | off — stream tokens as they arrive", enabled: true },
  { name: "approvals", description: "manual | auto | all — approval mode", enabled: true },
  { name: "enforcement", description: "strict | light | off — guidance nudges", enabled: true },
  { name: "batch", description: "on | off — batch review of file changes", enabled: true },
  { name: "memory", description: "list | clear | delete <name> — persistent memory", enabled: true },
  { name: "proxy", description: "list | clean <name> — registered proxies; delete one's runs and optimisation state", enabled: true },
  { name: "cancel", description: "stop the run in flight", enabled: true },
];

const SESSION_NAMES = new Set(SESSION_COMMANDS.map((c) => c.name));

/**
 * Whether *text* is a session command rather than a query for the model.
 *
 * Read at send time: a session command must go out as `type: "command"`, and anything
 * else — including a skill invocation — as `type: "query"`.
 */
export function isSessionCommand(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed.startsWith("/")) return false;
  return SESSION_NAMES.has(trimmed.slice(1).split(/\s/)[0]);
}

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

/**
 * Case-insensitive filter over BOTH kinds of slash: session commands first, then skills.
 *
 * Session commands lead because they act on the session immediately, while a skill only
 * steers the next query — and because they are the half that used to be invisible.
 */
export function filterSlashItems(skills: ToggleItem[], query: string): ToggleItem[] {
  const q = query.trim().toLowerCase();
  const all = [...SESSION_COMMANDS, ...skills];
  if (!q) return all;
  return all.filter(
    (s) =>
      s.name.toLowerCase().includes(q) ||
      (s.description || "").toLowerCase().includes(q)
  );
}

/** Back-compat alias: skills only. */
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
