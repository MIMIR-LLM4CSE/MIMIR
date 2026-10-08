import { describe, it, expect } from "vitest";
import { disconnectOutcome, workingSessions } from "./disconnectUtils";

const session = (id: string, extra: Partial<{ running: boolean; runs: number }> = {}) =>
  ({ id, ...extra });

describe("workingSessions", () => {
  it("counts a conversation whose background run is still going", () => {
    // The case detaching exists for, and the one that went unasked: the turn that
    // launched the build has answered, so nothing is "running" — and walking away
    // abandoned a two-hour job with no question put.
    const sessions = [session("s1", { running: false, runs: 1 })];
    expect(workingSessions(sessions, null, false).map((s) => s.id)).toEqual(["s1"]);
  });

  it("counts the conversation on screen by its own run, not only by busy", () => {
    const sessions = [session("s1", { runs: 2 })];
    expect(workingSessions(sessions, "s1", false).map((s) => s.id)).toEqual(["s1"]);
  });

  it("still counts a live turn", () => {
    const sessions = [session("s1", { running: true }), session("s2")];
    expect(workingSessions(sessions, null, false).map((s) => s.id)).toEqual(["s1"]);
  });

  it("trusts busy for the conversation on screen", () => {
    // The sessions list is pushed when a turn ends, so it lags the turn happening here.
    const sessions = [session("s1")];
    expect(workingSessions(sessions, "s1", true).map((s) => s.id)).toEqual(["s1"]);
  });

  it("puts the conversation on screen first", () => {
    const sessions = [session("s2", { runs: 1 }), session("s1", { runs: 1 })];
    expect(workingSessions(sessions, "s1", false).map((s) => s.id)).toEqual(["s1", "s2"]);
  });

  it("is empty when nothing is working, so disconnecting asks nothing", () => {
    const sessions = [session("s1", { runs: 0 }), session("s2", { running: false })];
    expect(workingSessions(sessions, "s1", false)).toEqual([]);
  });

  it("does not invent the conversation on screen when the list has no such row", () => {
    expect(workingSessions([session("s2", { runs: 1 })], "s1", true).map((s) => s.id))
      .toEqual(["s2"]);
  });
});

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
