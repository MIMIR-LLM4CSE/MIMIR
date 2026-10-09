import { describe, expect, it } from "vitest";
import { isValidElement } from "react";
import { KIND_ICONS, iconForKind, labelForKind } from "./toolIcons";

// The families the servers declare (servers/_shared/capabilities.py TOOL_KINDS). Kept
// here as a literal rather than imported: this is the UI's end of a wire contract, and
// a family that loses its icon has to fail here and not fall silently back to 🔧.
const TOOL_KINDS = [
  "read", "search", "list", "outline",
  "write", "edit", "delete",
  "bash", "shell", "job", "verdict",
  "eval", "symbolic", "string", "date",
  "slurm", "env", "modules",
  "proxy", "proxy eval",
  "memory", "plan", "skill", "agent", "ask",
  "web", "github", "system",
  "tool",
];

describe("iconForKind", () => {
  it("has an icon for every declared family", () => {
    for (const kind of TOOL_KINDS) {
      expect(KIND_ICONS[kind], `no icon for "${kind}"`).toBeTruthy();
    }
  });

  it("covers no family the servers do not declare", () => {
    expect(Object.keys(KIND_ICONS).sort()).toEqual([...TOOL_KINDS].sort());
  });

  it("gives distinct families distinct icons", () => {
    // What the refonte is for: `evaluate` and `symbolic` both came out 🔧 before.
    const strings = Object.values(KIND_ICONS).filter((i) => typeof i === "string");
    expect(new Set(strings).size).toBe(strings.length);
  });

  it("draws every family with an emoji, bar the GitHub mark", () => {
    // A bare text glyph (❯, ∑) takes the text colour and not the width of its
    // neighbours, which breaks the alignment of the column.
    for (const [kind, icon] of Object.entries(KIND_ICONS)) {
      if (kind === "github") {
        expect(isValidElement(icon)).toBe(true);
        continue;
      }
      expect(typeof icon).toBe("string");
      expect(/\p{Extended_Pictographic}/u.test(icon as string), `"${kind}" is not an emoji`)
        .toBe(true);
    }
  });

  it("writes every family with a capital", () => {
    for (const kind of TOOL_KINDS) {
      expect(labelForKind(kind)[0], kind).toBe(kind[0].toUpperCase());
    }
  });

  it("capitalises the first letter only, not every word", () => {
    // A family is a phrase in sentence case, not a title: `text-transform: capitalize`
    // would render this one "Proxy Eval".
    expect(labelForKind("proxy eval")).toBe("Proxy eval");
  });

  it("keeps the inner capital of a brand the general rule would flatten", () => {
    expect(labelForKind("github")).toBe("GitHub");
  });

  it("falls back to the generic glyph for an unknown or missing family", () => {
    expect(iconForKind("something-a-third-party-named")).toBe(KIND_ICONS.tool);
    expect(iconForKind(undefined)).toBe(KIND_ICONS.tool);
    expect(iconForKind("")).toBe(KIND_ICONS.tool);
  });
});
