import { describe, it, expect } from "vitest";
import { runsInFlight } from "./runDockUtils";
import type { ChatMessage, ToolActivity } from "../types";

function tool(over: Partial<ToolActivity> & { id: string }): ToolActivity {
  return {
    name: "proxy_eval",
    icon: "🖥️",
    label: "Proxy eval: run",
    detail: "",
    status: "running",
    startedAt: 0,
    ...over,
  };
}

const frozen = (tools: ToolActivity[]): ChatMessage =>
  ({ id: "m1", role: "agent", kind: "tools", tools } as ChatMessage);

describe("runsInFlight", () => {
  it("keeps a blocking run that has said what it is doing", () => {
    const runs = runsInFlight([], [tool({ id: "c1", phase: "building (1/2)" })]);
    expect(runs.map((r) => r.id)).toEqual(["c1"]);
  });

  it("ignores a running row that has never reported anything", () => {
    // Most calls are over in milliseconds. A spinner with nothing to say does not
    // earn a card.
    const runs = runsInFlight([], [tool({ id: "c1" })]);
    expect(runs).toEqual([]);
  });

  it("keeps a detached run that its watcher is still reporting on", () => {
    const runs = runsInFlight(
      [frozen([tool({ id: "c1", status: "background", jobKey: "J1", phase: "building" })])],
      [],
    );
    expect(runs.map((r) => r.id)).toEqual(["c1"]);
  });

  it("ignores a detached row nothing is reporting on any more", () => {
    // A session reloaded from disk brings back rows still marked background whose
    // watchers died with the process. A card for one of those would never leave.
    const runs = runsInFlight(
      [frozen([tool({ id: "c1", status: "background", jobKey: "J1" })])],
      [],
    );
    expect(runs).toEqual([]);
  });

  it("ignores a row that has finished", () => {
    const runs = runsInFlight(
      [frozen([tool({ id: "c1", status: "ok", phase: "building" })])],
      [],
    );
    expect(runs).toEqual([]);
  });

  it("does not list one run twice when it appears in both places", () => {
    const row = tool({ id: "c1", status: "background", jobKey: "J1", phase: "building" });
    const runs = runsInFlight([frozen([row])], [row]);
    expect(runs).toHaveLength(1);
  });

  it("reads the frozen transcript before the live step, so order follows the thread", () => {
    const runs = runsInFlight(
      [frozen([tool({ id: "old", status: "background", jobKey: "J1", phase: "building" })])],
      [tool({ id: "new", phase: "building" })],
    );
    expect(runs.map((r) => r.id)).toEqual(["old", "new"]);
  });
});
