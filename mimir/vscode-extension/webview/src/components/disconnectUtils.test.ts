import { describe, it, expect } from "vitest";
import { disconnectOutcome } from "./disconnectUtils";

describe("disconnectOutcome", () => {
  it("promises the server stops only when it is this window's", () => {
    const ours = disconnectOutcome(true);
    expect(ours.note).toContain("ends the server");
    expect(ours.offerStop).toBe(false);
  });

  it("says the run carries on when the server is not this window's", () => {
    // The failure this exists for: the dialog promised the turns ended, the host left
    // a server it had only attached to alone, and the run carried on through a choice
    // the user read as stopping it.
    const theirs = disconnectOutcome(false);
    expect(theirs.note).toContain("leaves it running");
    expect(theirs.note).not.toContain("ends the server");
    expect(theirs.discardLabel).toContain("leave them running");
  });

  it("offers stopping it only where disconnecting would not", () => {
    expect(disconnectOutcome(false).offerStop).toBe(true);
    expect(disconnectOutcome(true).offerStop).toBe(false);
  });

  it("says what stopping a shared server costs", () => {
    // One server per workspace, so another window may be reading it. Ending its turns
    // has to be a stated consequence rather than a surprise.
    expect(disconnectOutcome(false).note).toContain("another window");
  });
});
