import { describe, expect, it } from "vitest";
import { reversePatch } from "./patch";

describe("reversePatch", () => {
  it("restores removed lines and drops added ones", () => {
    const after = "a\nB\nc\nd";
    const patch = "--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c";
    expect(reversePatch(after, patch)).toBe("a\nb\nc\nd");
  });

  it("keeps the lines before a hunk that starts further down", () => {
    const after = "1\n2\n3\nnew\n4";
    const patch = "@@ -3,2 +3,3 @@\n 3\n+new\n 4";
    expect(reversePatch(after, patch)).toBe("1\n2\n3\n4");
  });

  it("recovers an empty file from a patch that created it", () => {
    expect(reversePatch("x\ny", "@@ -0,0 +1,2 @@\n+x\n+y")).toBe("");
  });
});
