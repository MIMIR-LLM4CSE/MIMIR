import { describe, it, expect } from "vitest";
import {
  inlineSegments,
  parseLedger,
  rowLevel,
  splitAnswerLedger,
  summaryChips,
} from "./ledgerUtils";

// Blocks exactly as client/query_engine/verification.render_ledger emits them.
const BLOCK = [
  '<!--mimir:ledger status="warn" files="2" summary="2 files · 1 not validated · 1 step open"-->',
  "Verification ledger — machine-recorded, not model-authored:",
  "- `solver.py` — validated: oracle (red→green)",
  "- `test_solver.py` — **not validated**",
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

describe("parseLedger", () => {
  it("reads the header fields off the marker", () => {
    const led = parseLedger(BLOCK);
    expect(led.status).toBe("warn");
    expect(led.files).toBe(2);
    expect(summaryChips(led.summary)).toEqual([
      "2 files",
      "1 not validated",
      "1 step open",
    ]);
  });

  it("collects one row per markdown list item, framing line excluded", () => {
    expect(parseLedger(BLOCK).rows).toHaveLength(3);
    expect(parseLedger(BLOCK).rows[0]).toBe("`solver.py` — validated: oracle (red→green)");
  });

  it("falls back to note status when the marker is absent or unknown", () => {
    expect(parseLedger("- a row").status).toBe("note");
    expect(parseLedger('<!--mimir:ledger status="bogus"-->\n- a row').status).toBe("note");
    expect(parseLedger("- a row").files).toBe(0);
  });
});

describe("rowLevel", () => {
  it("tints emphasised rows as the ones needing action", () => {
    expect(rowLevel("`test_solver.py` — **not validated**")).toBe("warn");
    expect(rowLevel("Checklist: **1 step unchecked** — add it")).toBe("warn");
  });

  it("treats a file row carrying a tier as settled", () => {
    expect(rowLevel("`solver.py` — validated: oracle (red→green)")).toBe("ok");
  });

  it("leaves prose notes neutral", () => {
    expect(rowLevel("Discrimination: none observed — every check…")).toBe("note");
  });
});

describe("inlineSegments", () => {
  it("splits code spans and bold out of a row", () => {
    expect(inlineSegments("`a.py` — **not validated**")).toEqual([
      { kind: "code", text: "a.py" },
      { kind: "text", text: " — " },
      { kind: "strong", text: "not validated" },
    ]);
  });

  it("returns a single text segment for a plain row", () => {
    expect(inlineSegments("Declared but never written")).toEqual([
      { kind: "text", text: "Declared but never written" },
    ]);
  });
});
