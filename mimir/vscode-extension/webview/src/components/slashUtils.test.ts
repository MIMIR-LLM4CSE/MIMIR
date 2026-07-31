import { describe, it, expect } from "vitest";
import { detectSlashQuery, filterSkills, applySlash } from "./slashUtils";
import type { ToggleItem } from "../types";

const SKILLS: ToggleItem[] = [
  { name: "fix-bug", description: "diagnose and fix a defect", enabled: true },
  { name: "write-tests", description: "add unit tests", enabled: true },
  { name: "refactor-code", description: "restructure without behaviour change", enabled: false },
];

describe("detectSlashQuery", () => {
  it("detects a / at the start", () => {
    expect(detectSlashQuery("/fix", 4)).toEqual({ start: 0, query: "fix" });
  });

  it("detects a / after leading whitespace", () => {
    expect(detectSlashQuery("  /fix", 6)).toEqual({ start: 2, query: "fix" });
  });

  it("matches the empty query right after /", () => {
    expect(detectSlashQuery("/", 1)).toEqual({ start: 0, query: "" });
  });

  it("returns null once a space follows the command", () => {
    expect(detectSlashQuery("/fix bar", 8)).toBeNull();
  });

  it("returns null for a slash that is not at the start", () => {
    expect(detectSlashQuery("do /fix", 7)).toBeNull();
  });

  it("returns null when caret sits before the slash", () => {
    expect(detectSlashQuery("  /fix", 1)).toBeNull();
  });

  it("uses the caret, not end of text", () => {
    expect(detectSlashQuery("/fixme", 3)).toEqual({ start: 0, query: "fi" });
  });
});

describe("filterSkills", () => {
  it("returns all for an empty query", () => {
    expect(filterSkills(SKILLS, "")).toHaveLength(3);
  });
  it("matches on name", () => {
    expect(filterSkills(SKILLS, "fix").map((s) => s.name)).toEqual(["fix-bug"]);
  });
  it("matches on description", () => {
    expect(filterSkills(SKILLS, "unit").map((s) => s.name)).toEqual(["write-tests"]);
  });
  it("is case-insensitive", () => {
    expect(filterSkills(SKILLS, "REFACTOR").map((s) => s.name)).toEqual(["refactor-code"]);
  });
});

describe("applySlash", () => {
  it("replaces the partial with /name + space", () => {
    const { text, caret } = applySlash("/fi", 0, 3, "fix-bug");
    expect(text).toBe("/fix-bug ");
    expect(caret).toBe(9);
  });

  it("preserves leading whitespace and trailing text", () => {
    const { text, caret } = applySlash("  /w rest", 2, 4, "write-tests");
    expect(text).toBe("  /write-tests  rest");
    expect(caret).toBe(15);
  });
});
