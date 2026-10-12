import { describe, it, expect } from "vitest";
import { looksLikeFile, splitMentionSegments } from "./mentionHighlight";

describe("looksLikeFile", () => {
  it("accepts paths and bare file names", () => {
    expect(looksLikeFile("mimir/client/loop.py")).toBe(true);
    expect(looksLikeFile("README.md")).toBe(true);
    expect(looksLikeFile("src/utils")).toBe(true);
  });

  it("rejects a resource uri or a bare resource name", () => {
    expect(looksLikeFile("memory://all")).toBe(false);
    expect(looksLikeFile("memory")).toBe(false);
  });
});

describe("splitMentionSegments", () => {
  it("returns a message with no mention as one text segment", () => {
    expect(splitMentionSegments("hello there")).toEqual([
      { kind: "text", text: "hello there" },
    ]);
  });

  it("marks a file mention and keeps the prose around it", () => {
    expect(splitMentionSegments("look at @src/foo.py please")).toEqual([
      { kind: "text", text: "look at " },
      {
        kind: "mention",
        text: "@src/foo.py",
        target: "src/foo.py",
        line: undefined,
        endLine: undefined,
        isFile: true,
      },
      { kind: "text", text: " please" },
    ]);
  });

  it("reads a line range off the mention", () => {
    const [seg] = splitMentionSegments("@src/foo.py:10-20");
    expect(seg).toMatchObject({ target: "src/foo.py", line: 10, endLine: 20 });
  });

  it("treats a single line as a one-line range", () => {
    const [seg] = splitMentionSegments("@src/foo.py:10");
    expect(seg).toMatchObject({ line: 10, endLine: 10 });
  });

  it("gives sentence punctuation back to the prose", () => {
    expect(splitMentionSegments("see @README.md.")).toEqual([
      { kind: "text", text: "see " },
      {
        kind: "mention",
        text: "@README.md",
        target: "README.md",
        line: undefined,
        endLine: undefined,
        isFile: true,
      },
      { kind: "text", text: "." },
    ]);
  });

  it("leaves a mid-word @ as prose", () => {
    expect(splitMentionSegments("write to me@example.org")).toEqual([
      { kind: "text", text: "write to me@example.org" },
    ]);
  });

  it("marks a resource mention without making it a file", () => {
    const [seg] = splitMentionSegments("@memory://all what do you know");
    expect(seg).toMatchObject({ kind: "mention", target: "memory://all", isFile: false });
  });

  it("keeps a lone @ as prose", () => {
    expect(splitMentionSegments("a @ b")).toEqual([{ kind: "text", text: "a @ b" }]);
  });

  it("handles several mentions in a row", () => {
    const segs = splitMentionSegments("@a.py @b.py");
    expect(segs.filter((s) => s.kind === "mention")).toHaveLength(2);
  });
});
