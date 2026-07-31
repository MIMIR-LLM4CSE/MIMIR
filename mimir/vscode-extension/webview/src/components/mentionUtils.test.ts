import { describe, it, expect } from "vitest";
import {
  detectMentionQuery,
  filterResources,
  applyMention,
  wrapIndex,
} from "./mentionUtils";
import type { ResourceItem } from "../types";

const RES: ResourceItem[] = [
  { uri: "memory://all", name: "memory", description: "all memories" },
  { uri: "todo://current", name: "todo", description: "task checklist" },
  { uri: "files://list", name: "files", description: "workspace files" },
];

describe("detectMentionQuery", () => {
  it("detects an @ at start", () => {
    expect(detectMentionQuery("@mem", 4)).toEqual({ start: 0, query: "mem" });
  });

  it("detects an @ after whitespace", () => {
    expect(detectMentionQuery("look at @to", 10)).toEqual({ start: 8, query: "to" });
  });

  it("matches the empty query right after @", () => {
    expect(detectMentionQuery("hi @", 4)).toEqual({ start: 3, query: "" });
  });

  it("ignores @ inside a word (email)", () => {
    expect(detectMentionQuery("me@host.com", 11)).toBeNull();
  });

  it("returns null when caret is past a completed token + space", () => {
    expect(detectMentionQuery("@memory done", 12)).toBeNull();
  });

  it("uses the caret, not end of text", () => {
    // caret sits right after "@me" even though more text follows
    expect(detectMentionQuery("@memory extra", 3)).toEqual({ start: 0, query: "me" });
  });
});

describe("filterResources", () => {
  it("returns all for an empty query", () => {
    expect(filterResources(RES, "")).toHaveLength(3);
  });
  it("matches on name", () => {
    expect(filterResources(RES, "mem").map((r) => r.uri)).toEqual(["memory://all"]);
  });
  it("matches on description", () => {
    expect(filterResources(RES, "checklist").map((r) => r.uri)).toEqual(["todo://current"]);
  });
  it("is case-insensitive on uri", () => {
    expect(filterResources(RES, "FILES").map((r) => r.uri)).toEqual(["files://list"]);
  });
});

describe("applyMention", () => {
  it("replaces the partial with @insert + space", () => {
    const { text, caret } = applyMention("show @me", 5, 8, "memory");
    expect(text).toBe("show @memory ");
    expect(caret).toBe(text.length);
  });

  it("preserves trailing text after the caret", () => {
    const { text } = applyMention("a @m b", 2, 4, "memory");
    expect(text).toBe("a @memory  b");
  });

  it("inserts a separating space when the prefix ends non-space", () => {
    // start==caret==1 (an '@' typed immediately after a word, no partial)
    const { text } = applyMention("x@", 1, 2, "todo");
    expect(text).toBe("x @todo ");
  });
});

describe("wrapIndex", () => {
  it("wraps past the end", () => expect(wrapIndex(3, 3)).toBe(0));
  it("wraps before the start", () => expect(wrapIndex(-1, 3)).toBe(2));
  it("handles empty", () => expect(wrapIndex(0, 0)).toBe(0));
});
