import { describe, it, expect } from "vitest";
import {
  HISTORY_IDLE,
  pushHistory,
  stepHistory,
} from "./historyUtils";

const ENTRIES = ["first", "second", "third"];

describe("stepHistory", () => {
  it("walks back from the newest message", () => {
    const a = stepHistory(ENTRIES, HISTORY_IDLE, "draft", "up")!;
    expect(a.text).toBe("third");
    const b = stepHistory(ENTRIES, a.state, a.text, "up")!;
    expect(b.text).toBe("second");
    const c = stepHistory(ENTRIES, b.state, b.text, "up")!;
    expect(c.text).toBe("first");
    expect(stepHistory(ENTRIES, c.state, c.text, "up")).toBeNull();
  });

  it("walks forward and restores the draft", () => {
    const a = stepHistory(ENTRIES, HISTORY_IDLE, "draft", "up")!;
    const b = stepHistory(ENTRIES, a.state, a.text, "up")!;
    const c = stepHistory(ENTRIES, b.state, b.text, "down")!;
    expect(c.text).toBe("third");
    const d = stepHistory(ENTRIES, c.state, c.text, "down")!;
    expect(d.text).toBe("draft");
    expect(d.state).toEqual(HISTORY_IDLE);
  });

  it("does nothing without history or outside a browse", () => {
    expect(stepHistory([], HISTORY_IDLE, "", "up")).toBeNull();
    expect(stepHistory(ENTRIES, HISTORY_IDLE, "", "down")).toBeNull();
  });
});

describe("pushHistory", () => {
  it("skips an immediate repeat", () => {
    expect(pushHistory(["a"], "a")).toEqual(["a"]);
    expect(pushHistory(["a"], "b")).toEqual(["a", "b"]);
  });
});
