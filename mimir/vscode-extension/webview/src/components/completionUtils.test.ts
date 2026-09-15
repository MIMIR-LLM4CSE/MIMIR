import { describe, it, expect } from "vitest";
import { parseCompletion, splitAnswerCompletion } from "./completionUtils";
import { splitAnswerLedger } from "./ledgerUtils";

// A block exactly as client/guardrails/workflow.render_completion_report emits it.
const BLOCK = [
  '<!--mimir:completion status="incomplete" risk="medium" headline="Task is incomplete."-->',
  "Completion report — machine-recorded, not model-authored:",
  "Completed:",
  "- No denied actions",
  "",
  "Remaining issues:",
  "- Declared but never written: solver.py",
  "",
  "Residual risk: medium.",
].join("\n");

const LEDGER = [
  '<!--mimir:ledger status="warn" files="1" summary="1 file"-->',
  "Verification ledger — machine-recorded, not model-authored:",
  "- `solver.py` — **not checked**",
].join("\n");

describe("splitAnswerCompletion", () => {
  it("separates the prose from the completion block", () => {
    const { body, report } = splitAnswerCompletion(`The answer.\n\n${BLOCK}`);
    expect(body).toBe("The answer.");
    expect(report).toBe(BLOCK);
  });

  it("leaves an answer without a report untouched", () => {
    const { body, report } = splitAnswerCompletion("Just prose.");
    expect(body).toBe("Just prose.");
    expect(report).toBeNull();
  });

  it("peels both blocks off in the order they are appended", () => {
    // The ledger is appended last, so it comes off first and the report is the tail
    // of what is left. This is the order ChatMessage relies on.
    const full = `The answer.\n\n${BLOCK}\n\n${LEDGER}`;
    const { body: withReport, ledger } = splitAnswerLedger(full);
    expect(ledger).toBe(LEDGER);
    const { body, report } = splitAnswerCompletion(withReport);
    expect(body).toBe("The answer.");
    expect(report).toBe(BLOCK);
  });
});

describe("parseCompletion", () => {
  it("recovers the header fields and the body", () => {
    const report = parseCompletion(BLOCK);
    expect(report.status).toBe("incomplete");
    expect(report.risk).toBe("medium");
    expect(report.headline).toBe("Task is incomplete.");
    expect(report.body.startsWith("Completed:")).toBe(true);
    expect(report.body.endsWith("Residual risk: medium.")).toBe(true);
    // The framing line belongs to the panel's chrome, not to the body it renders.
    expect(report.body).not.toContain("machine-recorded");
  });

  it("keeps the three statuses apart", () => {
    for (const s of ["incomplete", "handback", "refused-only"]) {
      expect(parseCompletion(BLOCK.replace("incomplete", s)).status).toBe(s);
    }
  });

  it("falls back to incomplete on an unreadable marker", () => {
    // Fail loud rather than quiet: a block nobody can parse still renders, and the
    // status it defaults to is the one that does not claim the work is finished.
    const report = parseCompletion("Completed:\n- nothing");
    expect(report.status).toBe("incomplete");
    expect(report.body).toBe("Completed:\n- nothing");
  });
});
