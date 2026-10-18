import { describe, it, expect } from "vitest";
import { splitAnswerLedger } from "./ledgerUtils";

// Blocks exactly as client/query_engine/verification.render_ledger emits them.
const BLOCK = [
  '<!--mimir:ledger status="warn" files="2" summary="2 files · 1 not checked · 1 step open"-->',
  "Verification ledger — machine-recorded, not model-authored:",
  "- `solver.py` — checked: static",
  "- `test_solver.py` — **not checked**",
  "- Checklist: **1 step unchecked** — add the convergence test",
].join("\n");

describe("splitAnswerLedger", () => {
  it("separates the prose from the ledger block", () => {
    const { body, ledger } = splitAnswerLedger(`The answer.\n\n${BLOCK}`);
    expect(body).toBe("The answer.");
    expect(ledger).toBe(BLOCK);
  });

  it("leaves an answer without a ledger untouched", () => {
    const { body, ledger } = splitAnswerLedger("Just prose.");
    expect(body).toBe("Just prose.");
    expect(ledger).toBeNull();
  });

  it("keeps prose that merely mentions the word ledger", () => {
    expect(splitAnswerLedger("I wrote the ledger module.").ledger).toBeNull();
  });
});
