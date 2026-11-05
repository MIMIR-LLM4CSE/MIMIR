// Pure helpers for the "/"-command autocomplete in the chat input.
//
// Two kinds of slash share the input: a skill ("/fix-bug …") is part of the query, sent
// as `type: "query"`; a session command ("/mode agent") never reaches the model and is
// sent as `type: "command"`. Both are offered at the start of the input; anywhere else
// after whitespace, only skills, which the backend loads wherever the query names them.
// `isSessionCommand` routes them. Free of React, so unit-testable.

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
  { name: "temperature", description: "0-2 | default — sampling temperature for this model", enabled: true },
  { name: "streaming", description: "on | off — stream tokens as they arrive", enabled: true },
  { name: "approvals", description: "manual | auto | all — approval mode", enabled: true },
  { name: "enforcement", description: "strict | light | off — guidance nudges", enabled: true },
  { name: "batch", description: "on | off — batch review of file changes", enabled: true },
  { name: "memory", description: "list | clear | delete <name> — persistent memory", enabled: true },
  { name: "proxy", description: "list | clean <name> — registered proxies; delete one's runs and optimisation state", enabled: true },
  { name: "modules", description: "refresh | <term> — environment-module catalogue: status, rebuild, search", enabled: true },
  { name: "cancel", description: "stop the run in flight", enabled: true },
  { name: "diag", description: "what the event chain has done — pump, journal, watermark", enabled: true },
  { name: "subagents", description: "explore | parallel — how far a sub-agent may go", enabled: true },
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
  /** Index of the "/" that opens the command. */
  start: number;
  /** The partial command name typed after "/" (may be empty right after "/"). */
  query: string;
  /** Whether the "/" opens the input — the one place a session command may go. */
  leading: boolean;
}

/**
 * If the caret sits inside a "/"-command being typed, return its start index and the
 * partial query; otherwise null. The "/" opens the input or follows whitespace, and
 * the caret has not moved past the word (skill names never contain spaces). Mid-text,
 * a "/" inside the word reads as a path, not a command. `caret` is the textarea
 * selectionStart.
 */
export function detectSlashQuery(text: string, caret: number): SlashQuery | null {
  if (caret < 0 || caret > text.length) return null;

  // Walk back from the caret to the start of the word it is in.
  let start = caret;
  while (start > 0 && /\S/.test(text[start - 1])) start -= 1;
  if (start === caret || text[start] !== "/") return null;

  const query = text.slice(start + 1, caret);
  const leading = text.slice(0, start).trim() === "";
  if (!leading && query.includes("/")) return null;
  return { start, query, leading };
}

/** A row of the "/" dropdown: a session command or a skill, which the dropdown groups. */
export interface SlashItem extends ToggleItem {
  kind: "command" | "skill";
}

/**
 * Case-insensitive filter over BOTH kinds of slash: session commands first, then skills;
 * skills alone with *skillsOnly* (a "/" past the start of the input).
 *
 * Session commands lead because they act on the session immediately, while a skill only
 * steers the next query — and because they are the half with no other way to be found.
 */
export function filterSlashItems(
  skills: ToggleItem[],
  query: string,
  skillsOnly = false
): SlashItem[] {
  const q = query.trim().toLowerCase();
  const all: SlashItem[] = [
    ...(skillsOnly ? [] : SESSION_COMMANDS.map((c) => ({ ...c, kind: "command" as const }))),
    ...skills.map((s) => ({ ...s, kind: "skill" as const })),
  ];
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
 * text and caret. Preserves the text before the slash.
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
