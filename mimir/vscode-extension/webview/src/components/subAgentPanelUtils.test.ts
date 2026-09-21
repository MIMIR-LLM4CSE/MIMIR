import { describe, it, expect } from "vitest";
import {
  activityRows,
  anyRunning,
  grantSummary,
  parseTodo,
  stateLabel,
  taskTitle,
  workspaceLine,
} from "./subAgentPanelUtils";
import type { SubAgent } from "../types";

const sub = (over: Partial<SubAgent> = {}): SubAgent =>
  ({ id: "sub-ab12cd34", state: "finished", ...over } as SubAgent);

describe("stateLabel", () => {
  it("separates a run that ended from a task that got done", () => {
    // A sub-agent out of steps still ends cleanly; saying "done" would misreport it.
    expect(stateLabel(sub({ completed: true }))).toBe("done");
    expect(stateLabel(sub({ completed: false }))).toBe("incomplete");
  });

  it("shows a child that is still working, and one that died", () => {
    expect(stateLabel(sub({ state: "running" }))).toBe("running");
    expect(stateLabel(sub({ state: "failed" }))).toBe("failed");
    expect(stateLabel(sub({ state: "abandoned" }))).toBe("abandoned");
    // A caller that gave up outranks a clean finish the caller never saw.
    expect(stateLabel(sub({ state: "abandoned", completed: true }))).toBe("abandoned");
  });

  it("does not invent a state for a card that was never written", () => {
    expect(stateLabel(sub({ state: "unknown" }))).toBe("unknown");
  });
});

describe("grantSummary", () => {
  it("names the tools the child was given", () => {
    expect(grantSummary(sub({ tools: ["read_file_lines", "bash_run"] })))
      .toBe("read_file_lines, bash_run");
  });

  it("says what an empty grant means rather than showing nothing", () => {
    expect(grantSummary(sub({ tools: [] }))).toBe("read-only exploration");
  });

  it("adds the budget in minutes", () => {
    expect(grantSummary(sub({ tools: [], budget_secs: 600 })))
      .toBe("read-only exploration, 10 min");
  });
});

describe("taskTitle", () => {
  it("takes the first line of a task that is a paragraph", () => {
    expect(taskTitle(sub({ task: "Vectorise the inner loop\n\nDetails follow." })))
      .toBe("Vectorise the inner loop");
  });

  it("falls back to the id when the card carries no task", () => {
    expect(taskTitle(sub())).toBe("sub-ab12cd34");
  });
});

describe("parseTodo", () => {
  it("reads the checkbox lines and ignores the rest of the file", () => {
    const md = "# Todo\n\n- [x] measure the baseline\n- [ ] try the blocked variant\n\nnotes\n";
    expect(parseTodo(md)).toEqual([
      { done: true, text: "measure the baseline" },
      { done: false, text: "try the blocked variant" },
    ]);
  });

  it("treats a missing list as empty, not as an error", () => {
    expect(parseTodo(undefined)).toEqual([]);
    expect(parseTodo("")).toEqual([]);
  });
});

describe("activityRows", () => {
  const log = [
    { v: 1, t: "tc", i: "c1", n: "grep_files", l: "Searching: solve", d: "src/" },
    { v: 1, t: "tr", i: "c1", ok: true, s: "3 matches", ms: 41 },
    { v: 1, t: "tc", i: "c2", n: "bash_run", l: "Running: make", d: "" },
  ];

  it("pairs a call with its result and leaves the last one running", () => {
    const rows = activityRows(log);
    expect(rows.map((r) => [r.label, r.status, r.detail])).toEqual([
      ["Searching: solve", "ok", "3 matches"],
      ["Running: make", "running", ""],
    ]);
    expect(rows[0].ms).toBe(41);
  });

  it("marks a failed step as an error", () => {
    const rows = activityRows([log[0], { t: "tr", i: "c1", ok: false, s: "no such file" }]);
    expect(rows[0].status).toBe("error");
  });

  it("drops a result whose call was trimmed off the head of the log", () => {
    expect(activityRows([{ t: "tr", i: "gone", ok: true }])).toEqual([]);
  });

  it("treats a missing log as no activity, not as an error", () => {
    expect(activityRows(undefined)).toEqual([]);
  });
});

describe("anyRunning", () => {
  it("is what a live view polls on", () => {
    expect(anyRunning([{ state: "finished" }, { state: "running" }])).toBe(true);
    expect(anyRunning([{ state: "finished" }])).toBe(false);
  });
});

describe("workspaceLine", () => {
  it("says where the copy is, since keeping it is no longer the exception", () => {
    expect(workspaceLine(sub({
      workspace: { branch: "mimir/sub-ab12cd34", path: "/tmp/wt/sub-ab12cd34", kept: true },
    }))).toBe("branch mimir/sub-ab12cd34 · copy at /tmp/wt/sub-ab12cd34");
  });

  it("shows only the copy when the child wrote nothing and got no branch", () => {
    expect(workspaceLine(sub({ workspace: { branch: "", path: "/tmp/wt/sub-ab12cd34" } })))
      .toBe("copy at /tmp/wt/sub-ab12cd34");
  });

  it("still reads a card written back when copies were removed", () => {
    expect(workspaceLine(sub({
      workspace: { branch: "mimir/old", path: "/tmp/wt/old", kept: false },
    }))).toBe("branch mimir/old · copy removed");
  });

  it("says nothing about a child that never had a copy", () => {
    expect(workspaceLine(sub())).toBe("");
  });
});
