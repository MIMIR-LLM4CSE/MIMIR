/** The logic behind the panel's host tree, read without a browser. */
import { describe, expect, it } from "vitest";
import {
  badgeDetailRows,
  badgeKey,
  badgeLabel,
  elapsedLabel,
  groupKeys,
  originLabel,
  progressLabel,
  toggle,
} from "./panelGroupUtils";
import type { PanelGroup, WatchedRun } from "../types";

const run = (over: Partial<WatchedRun> = {}): WatchedRun =>
  ({ job_key: "k1", ...over }) as WatchedRun;

describe("opening a row", () => {
  it("flips it, and gives React a new object to see", () => {
    const open = {};
    const once = toggle(open, "host:local");
    expect(once["host:local"]).toBe(true);
    expect(open).toEqual({});
    expect(toggle(once, "host:local")).toEqual({});
  });

  it("names a badge's own row apart from any group's", () => {
    // Both live in one map, so a job key that happened to equal a group id would
    // open the wrong thing.
    expect(badgeKey(run({ job_key: "host:local" }))).toBe("badge:host:local");
  });

  it("walks the whole tree when every row is wanted", () => {
    const groups: PanelGroup[] = [
      { id: "a", label: "a", groups: [{ id: "a/1", label: "1" }] },
      { id: "b", label: "b" },
    ];
    expect(groupKeys(groups)).toEqual(["a", "a/1", "b"]);
    expect(groupKeys(undefined)).toEqual([]);
  });
});

describe("what a badge says before it is opened", () => {
  it("names the machine, which is what the row does not already say", () => {
    expect(badgeLabel(run({ nodes: ["gpu007"], job_id: "15306620" }))).toBe("gpu007");
  });

  it("counts them when a job got several", () => {
    expect(badgeLabel(run({ nodes: ["gpu001", "gpu002"] }))).toBe("2 nodes");
  });

  it("falls back to the job id, then to the pid, then to the handle", () => {
    expect(badgeLabel(run({ job_id: "15306620" }))).toBe("15306620");
    expect(badgeLabel(run({ pid: 4242 }))).toBe("pid 4242");
    expect(badgeLabel(run())).toBe("k1");
  });
});

describe("who launched it", () => {
  it("says so plainly for the conversation's own run", () => {
    expect(originLabel(undefined)).toBe("this conversation");
    expect(originLabel({ kind: "main", label: "", session: "s1" })).toBe(
      "this conversation",
    );
  });

  it("names the child and its task", () => {
    expect(
      originLabel({ kind: "subagent", label: "port the solver to AVX-512" }),
    ).toBe("sub-agent: port the solver to AVX-512");
  });

  it("still says a child started it when the task did not travel", () => {
    expect(originLabel({ kind: "subagent" })).toBe("a sub-agent");
  });
});

describe("how far along, and how long", () => {
  it("reports whatever the run's own output said", () => {
    expect(progressLabel(run({ phase: "linking", percent: 61.6 }))).toBe("linking · 62%");
    expect(progressLabel(run({ phase: "linking" }))).toBe("linking");
    expect(progressLabel(run())).toBe("");
  });

  it("says the elapsed time as a person would", () => {
    expect(elapsedLabel(run({ elapsed_s: 42 }))).toBe("42s");
    expect(elapsedLabel(run({ elapsed_s: 125 }))).toBe("2m 5s");
    expect(elapsedLabel(run({ elapsed_s: 4210 }))).toBe("1h 10m");
    expect(elapsedLabel(run())).toBe("");
  });
});

describe("the rows inside an opened badge", () => {
  it("shows a cluster job by its id and a shell run by its pid", () => {
    const slurm = badgeDetailRows(run({ job_id: "15306620", pid: 7 }));
    expect(slurm[0]).toEqual({ label: "job", value: "15306620" });
    const shell = badgeDetailRows(run({ pid: 4242 }));
    expect(shell[0]).toEqual({ label: "pid", value: "4242" });
  });

  it("omits a field the status op did not report, rather than reading 'unknown'", () => {
    const rows = badgeDetailRows(run());
    expect(rows.map((r) => r.label)).toEqual(["launched by"]);
  });

  it("always ends by saying whose run it is", () => {
    const rows = badgeDetailRows(
      run({
        job_id: "15306620",
        command: "mpirun -n 64 ./solve",
        state: "running",
        phase: "linking",
        percent: 62,
        elapsed_s: 4210,
        nodes: ["gpu001", "gpu007"],
        origin: { kind: "subagent", label: "sweep the tile size" },
      }),
    );
    expect(rows.map((r) => r.label)).toEqual([
      "job", "command", "nodes", "state", "progress", "running for", "launched by",
    ]);
    expect(rows[rows.length - 1].value).toBe("sub-agent: sweep the tile size");
  });
});
