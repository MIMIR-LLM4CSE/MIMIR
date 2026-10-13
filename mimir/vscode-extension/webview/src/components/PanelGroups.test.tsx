/** The panel's tree: a scheduler, the machines on it, and what is running there.
 *
 *  The component interprets nothing — it renders whatever a server put in `groups` —
 *  so what these pin is the shape of that rendering: what a closed row shows, what an
 *  open one adds, and that a badge says which machine a run is on and whose it is.
 */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { PanelGroupList } from "./PanelGroups";
import { badgeKey } from "./panelGroupUtils";
import type { PanelGroup, WatchedRun } from "../types";

const kind: PanelGroup = {
  id: "host:hpc1/kind:bb",
  label: "4× a100 · 64 cpus · 503 GB",
  value: "not profiled",
  state: "warn",
  nodes: ["gpu001", "gpu007"],
  lines: [{ label: "arch", value: "x86_64" }],
  action: {
    kind: "probe",
    label: "Profile a node",
    prompt: "Probe one node of the '4× a100' kind on scheduler 'hpc1'",
  },
};

const host: PanelGroup = {
  id: "host:hpc1",
  label: "hpc1",
  host: "hpc1",
  local: false,
  value: "1 node type · x86_64 · a100 · 2 partitions",
  lines: [{ label: "reached", value: "over ssh to hpc1" }],
  groups: [kind],
};

const tree = (open: Record<string, boolean>, groups: PanelGroup[] = [host]) =>
  renderToStaticMarkup(
    <PanelGroupList groups={groups} open={open} onToggle={() => {}} onPrompt={() => {}} />,
  );

describe("a closed tree", () => {
  it("shows the scheduler and its summary, and nothing of what is inside", () => {
    const html = tree({});
    expect(html).toContain("hpc1");
    expect(html).toContain("1 node type · x86_64 · a100 · 2 partitions");
    expect(html).not.toContain("over ssh to hpc1");
    expect(html).not.toContain("4× a100");
  });

  it("says it is closed, and is a button a keyboard can reach", () => {
    const html = tree({});
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain("<button");
    expect(html).toContain("▸");
  });
});

describe("an opened scheduler", () => {
  it("adds its own lines and the kinds of machine on it", () => {
    const html = tree({ "host:hpc1": true });
    expect(html).toContain('aria-expanded="true"');
    expect(html).toContain("▾");
    expect(html).toContain("over ssh to hpc1");
    expect(html).toContain("4× a100 · 64 cpus · 503 GB");
    // One level at a time: the kind is listed, not unfolded.
    expect(html).not.toContain("x86_64</span>");
  });

  it("unfolds a kind of machine on its own", () => {
    const html = tree({ "host:hpc1": true, "host:hpc1/kind:bb": true });
    expect(html).toContain("gpu001, gpu007");
    expect(html).toContain("Profile a node");
  });

  it("marks a kind nothing has been probed on", () => {
    expect(tree({ "host:hpc1": true })).toContain("is-warn");
  });
});

describe("the runs on a machine", () => {
  const run: WatchedRun = {
    job_key: "15306620",
    kind: "slurm-batch",
    state: "running",
    job_id: "15306620",
    command: "mpirun -n 64 ./solve",
    nodes: ["gpu007"],
    elapsed_s: 4210,
    origin: { kind: "subagent", label: "sweep the tile size", session: "p/subagents/s1" },
  };

  it("sits on the row even while it is closed", () => {
    // What is running is the one thing a reader should not have to open anything to see.
    const html = tree({}, [{ ...host, badges: [run] }]);
    expect(html).toContain("science-badge");
    expect(html).toContain("gpu007");
    expect(html).toContain("is-running");
  });

  it("opens onto the job, the command and whose run it is", () => {
    const html = tree({ [badgeKey(run)]: true }, [{ ...host, badges: [run] }]);
    expect(html).toContain("mpirun -n 64 ./solve");
    expect(html).toContain("1h 10m");
    expect(html).toContain("sub-agent: sweep the tile size");
  });

  it("names a shell run by its pid instead", () => {
    const shell: WatchedRun = {
      job_key: "k1", kind: "shell-command", state: "running",
      pid: 4242, command: "make -j64",
    };
    const html = tree({ [badgeKey(shell)]: true }, [{ ...host, badges: [shell] }]);
    expect(html).toContain("pid 4242");
    expect(html).toContain("make -j64");
    expect(html).toContain("this conversation");
  });
});

describe("what the tree asks for", () => {
  it("offers the request before making it, never the request itself", () => {
    // The prompt is the server's own sentence, and it reaches the model as a user turn
    // — so whatever tool the model reaches for keeps the approval it already requires.
    const html = tree({ "host:hpc1": true, "host:hpc1/kind:bb": true });
    expect(html).toContain("Profile a node");
    expect(html).not.toContain("Ask for it");
  });

  it("offers nothing for a kind that is already profiled", () => {
    const profiled: PanelGroup = {
      ...kind, value: undefined, state: undefined, action: undefined,
      detail: "Profiled on gpu007, 2026-10-02",
    };
    const html = tree(
      { "host:hpc1": true, "host:hpc1/kind:bb": true },
      [{ ...host, groups: [profiled] }],
    );
    expect(html).toContain("Profiled on gpu007");
    expect(html).not.toContain("Profile a node");
  });
});
