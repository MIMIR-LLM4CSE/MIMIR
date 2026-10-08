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
  it("promises the server stops, because now it does", () => {
    // A server exists claimed — detached, asked to be left running — or owned by the
    // window that started it. There is no third state: one nobody claimed stops when
    // its last client has been gone for the grace, whoever started it. So this dialog
    // has one outcome to describe, and describing it is no longer a guess about who
    // spawned what.
    const outcome = disconnectOutcome();
    expect(outcome.note).toContain("ends the server");
    expect(outcome.note).toContain("their turns with it");
  });

  it("offers keeping them going as the other answer", () => {
    expect(disconnectOutcome().note).toContain("leaves MIMIR running");
    expect(disconnectOutcome().discardLabel).toBe("Disconnect anyway");
  });
});
