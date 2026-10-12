// Pure helpers for the "@"-mention autocomplete in the chat input.
//
// Kept free of React/DOM so the tricky caret/token logic is unit-testable with vitest
// (see mentionUtils.test.ts). The dropdown component and App wiring consume these.

import type { ResourceItem } from "../types";

export interface MentionQuery {
  /** Index of the "@" that opens the mention being typed. */
  start: number;
  /** The partial text typed after "@" (may be empty right after "@"). */
  query: string;
}

// A mention token is "@" then run of non-space chars. It only opens a mention when the
// "@" is at the start of the input or preceded by whitespace — so "email@host" and
// mid-word "@" never trigger it (mirrors the backend's (?<!\S)@ rule).
const TOKEN_CHAR = /\S/;

/**
 * If the caret sits inside an "@"-mention being typed, return its start index and the
 * partial query; otherwise null. `caret` is the textarea selectionStart.
 */
export function detectMentionQuery(text: string, caret: number): MentionQuery | null {
  if (caret < 0 || caret > text.length) return null;

  // Walk left from the caret over token chars to find the "@".
  let i = caret - 1;
  while (i >= 0 && TOKEN_CHAR.test(text[i]) && text[i] !== "@") i--;
  if (i < 0 || text[i] !== "@") return null;

  // The "@" must be at start or preceded by whitespace.
  if (i > 0 && TOKEN_CHAR.test(text[i - 1])) return null;

  // Everything from "@"+1 up to the caret is the query (no whitespace by construction).
  const query = text.slice(i + 1, caret);
  if (/\s/.test(query)) return null;
  return { start: i, query };
}

/** Case-insensitive filter of resources by uri / name / description. */
export function filterResources(resources: ResourceItem[], query: string): ResourceItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return resources;
  return resources.filter((r) =>
    r.uri.toLowerCase().includes(q) ||
    (r.name || "").toLowerCase().includes(q) ||
    (r.description || "").toLowerCase().includes(q)
  );
}

export interface ApplyResult {
  text: string;
  /** New caret position (just past the inserted token + trailing space). */
  caret: number;
}

/**
 * Replace the "@partial" spanning [start, caret) with "@insert " and return the new text
 * and caret. `insert` is the token to place after "@" (a resource name or full uri).
 */
export function applyMention(
  text: string,
  start: number,
  caret: number,
  insert: string
): ApplyResult {
  const before = text.slice(0, start);
  const after = text.slice(caret);
  const token = `@${insert} `;
  const needsSpaceBefore = before.length > 0 && /\S$/.test(before);
  const prefix = needsSpaceBefore ? `${before} ` : before;
  return { text: `${prefix}${token}${after}`, caret: prefix.length + token.length };
}

/** Clamp a selection index into [0, len) with wrap-around for keyboard nav. */
export function wrapIndex(index: number, len: number): number {
  if (len <= 0) return 0;
  return ((index % len) + len) % len;
}

/** One row of the "@" dropdown: an MCP resource, or a workspace file. */
export type MentionItem =
  | {
      kind: "resource";
      /** React key, and what `applyMention` inserts after the "@". */
      insert: string;
      label: string;
      detail: string;
      description?: string;
    }
  | {
      kind: "file";
      insert: string;
      /** The file name alone — the part worth reading first. */
      label: string;
      /** Its directory, empty at the workspace root. */
      detail: string;
    };

/**
 * The dropdown's rows for `query`: the matching resources, then the matching files.
 *
 * Resources lead because there are a handful of them against a workspace's thousands,
 * so a file list would bury them, and because a name shorthand like `@memory` is
 * otherwise indistinguishable from a path the user has half-typed.
 *
 * `files` arrives already ranked by the extension host (it owns the index); this only
 * shapes it for display.
 */
export function buildMentionItems(
  resources: ResourceItem[],
  files: string[],
  query: string,
  limit = 40
): MentionItem[] {
  const items: MentionItem[] = filterResources(resources, query).map((r) => ({
    kind: "resource" as const,
    insert: r.name || r.uri,
    label: `@${r.name || r.uri}`,
    detail: r.uri,
    description: r.description || undefined,
  }));
  for (const path of files) {
    if (items.length >= limit) break;
    const name = path.split("/").pop() || path;
    items.push({
      kind: "file",
      insert: path,
      label: name,
      detail: path.slice(0, path.length - name.length).replace(/\/$/, ""),
    });
  }
  return items.slice(0, limit);
}
