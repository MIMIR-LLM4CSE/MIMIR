import { describe, expect, it } from "vitest";
import { withDiff } from "./batchUtils";

describe("withDiff", () => {
  it("lists a file once, with its latest patch", () => {
    const once = withDiff([], "a.py", "@@ -1 +1 @@\n-x\n+y");
    const twice = withDiff(once, "a.py", "@@ -1 +1 @@\n-x\n+z");
    expect(twice).toHaveLength(1);
    expect(twice[0].patch).toContain("+z");
  });

  it("marks a file the patch creates as new", () => {
    expect(withDiff([], "b.py", "--- /dev/null\n+++ b/b.py")[0].is_new).toBe(true);
    expect(withDiff([], "c.py", "")[0].is_new).toBe(true);
    expect(withDiff([], "d.py", "@@ -1 +1 @@")[0].is_new).toBe(false);
  });
});
