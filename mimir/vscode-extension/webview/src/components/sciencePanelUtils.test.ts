import { describe, it, expect } from "vitest";
import { SUBAGENT_LEVEL_OPTIONS } from "./SciencePanel";

describe("SUBAGENT_LEVEL_OPTIONS", () => {
  it("mirrors the ladder in client/config/constants.py, weakest first", () => {
    expect(SUBAGENT_LEVEL_OPTIONS.map((o) => o.value)).toEqual(["explore", "parallel"]);
  });

  it("has no rung for writing in the shared tree", () => {
    // Two sub-agents editing one tree overwrite each other in silence. Writing means
    // a copy of the repository, or it does not happen — so there is nothing between.
    expect(SUBAGENT_LEVEL_OPTIONS).toHaveLength(2);
  });

  it("says what each rung allows, since that is the whole decision", () => {
    for (const option of SUBAGENT_LEVEL_OPTIONS) {
      expect(option.desc.length).toBeGreaterThan(20);
      expect(option.label).toBeTruthy();
    }
  });
});
