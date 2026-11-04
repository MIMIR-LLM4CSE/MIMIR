import { describe, expect, it } from "vitest";
import { gainOf, mergeGraph, outcome, type MergeState } from "./merge";
import { mergeable, mergePrompt, pickerHtml, pickLabel, ticked, type ProxyMeta } from "./picker";

const sources = [
  { proxy: "pa", relative: 0.8, files: ["a.py"] },
  { proxy: "pc", relative: 0.85, files: ["a.py"] },
  { proxy: "pb", relative: 0.9, files: ["b.py"] },
];
const pairs = [
  { a: "pa", b: "pc", shared: ["a.py"], conflicts: ["a.py"] },
  { a: "pa", b: "pb", shared: [], conflicts: [] },
  { a: "pc", b: "pb", shared: [], conflicts: [] },
];
const chain: MergeState = {
  phase: "done", sources, pairs, expected: 0.612,
  all: { applied: ["pa", "pb"], conflicts: [{ proxy: "pc", files: ["a.py"] }], expected: 0.72, relative: 0.95, holds: false },
  steps: [
    { proxy: "pa", kept: true, marginal: 0.8, interaction: 1 },
    { proxy: "pc", kept: false, why: "conflict", files: ["a.py"] },
    { proxy: "pb", kept: false, why: "interference", marginal: 1.0625, interaction: 1.18 },
  ],
  final: { from: "chain", relative: 0.8, kept: ["pa"],
    set_aside: [{ proxy: "pc", why: "conflict" }, { proxy: "pb", why: "interference" }] },
};

describe("a merge's outcome per source", () => {
  it("says what was kept and why the others were set aside", () => {
    expect(outcome(chain, "pa").state).toBe("kept");
    expect(outcome(chain, "pc").state).toBe("conflict");
    expect(outcome(chain, "pb")).toEqual({ state: "interference", marginal: 1.0625 });
  });

  it("marks the sources being measured all at once", () => {
    const running: MergeState = { ...chain, phase: "all", steps: [], final: null,
      all: { applied: ["pa", "pb"], conflicts: [{ proxy: "pc", files: ["a.py"] }], expected: 0.72 } };
    expect(outcome(running, "pa").state).toBe("applied");
    expect(outcome(running, "pc").state).toBe("conflict");
  });

  it("is a forecast without a merge", () => {
    expect(outcome(null, "pa").state).toBe("forecast");
  });
});

describe("the merge graph", () => {
  it("draws the conflict as an arc and the gains expected and measured", () => {
    const html = mergeGraph(chain);
    expect(html).toContain('class="arc conflict"');
    expect(html).not.toContain('class="arc shared"');
    expect(html).toContain("expected 39%");
    expect(html).toContain("measured 20%");
    expect(html).toContain("set aside: slowed the whole");
    expect(html).toMatch(/class="mlink interference intf"/);
  });

  it("is a forecast for proxies ticked, not merged", () => {
    const html = mergeGraph({ sources, pairs, expected: 0.612, refused: [{ proxy: "pd", why: "no run accepted beyond its baseline" }] }, 600);
    expect(html).toContain("Merge forecast");
    expect(html).toContain("never measured");
    expect(html).toContain("left out: pd");
  });

  it("says why a forecast cannot be measured", () => {
    const html = mergeGraph({ sources, pairs, expected: 0.612, refusal: "pa and pb have a different metric" });
    expect(html).toContain("pa and pb have a different metric");
    expect(html).not.toContain("expected together");
  });

  it("writes a gain from a relative", () => {
    expect(gainOf(0.8)).toBe("20%");
    expect(gainOf(1.05)).toBe("−5.0%");
    expect(gainOf(null)).toBe("–");
  });
});

const meta: ProxyMeta[] = [
  { name: "pa", branch: "mimir/sub-1", metric: "time_s", goal: "min", steps: 4, gain: 0.2, active: true, frozen: "", merge: false, sources: [], mergeable: true },
  { name: "pb", branch: "mimir/sub-2", metric: "time_s", goal: "min", steps: 2, gain: 0.1, active: false, frozen: "", merge: false, sources: [], mergeable: true },
  { name: "pz", branch: "", metric: "time_s", goal: "min", steps: 3, gain: null, active: false, frozen: "", merge: false, sources: [], mergeable: false },
  { name: "merge-1", branch: "mimir/sub-3", metric: "time_s", goal: "min", steps: 3, gain: 0.25, active: false, frozen: "", merge: true, sources: ["pa", "pb"], mergeable: true },
  { name: "pa@20261104-101500", branch: "", metric: "time_s", goal: "min", steps: 6, gain: 0.3, active: false, frozen: "cleaned", merge: false, sources: [] },
];

describe("the proxy picker", () => {
  it("groups the proxies and says each one's gain", () => {
    const html = pickerHtml(meta, "", "pa", true, new Set());
    expect(html.indexOf("Running")).toBeLessThan(html.indexOf("Merges"));
    expect(html.indexOf("Merges")).toBeLessThan(html.indexOf("Frozen"));
    expect(html).toContain("20% better");
    expect(html).toContain('<span class="badge on">active</span>');
    expect(html).toContain('<span class="badge">cleaned</span>');
    // The active proxy is the empty pick, and is the one shown.
    expect(html).toMatch(/pick-row is-cur"><input type="checkbox" data-tick="pa"/);
    expect(html).toMatch(/data-merge disabled/);
  });

  it("merges only the ticked proxies a merge can take", () => {
    expect(mergeable(meta, ["pa", "pz", "merge-1", "pa@20261104-101500", "pb"])).toEqual(["pa", "pb"]);
    const html = pickerHtml(meta, "pa,pb", "pa", true, new Set(["pa", "pb"]));
    expect(html).toContain("Merge 2");
    expect(html).not.toMatch(/data-merge disabled/);
  });

  it("names the pick on its button", () => {
    expect(ticked("pa,pb")).toEqual(["pa", "pb"]);
    expect(ticked("pa")).toEqual([]);
    expect(pickLabel("pa,pb", "pa")).toBe("pa + pb");
    expect(pickLabel("*", "pa")).toBe("Every proxy");
    expect(pickLabel("", "pa")).toBe("pa");
  });

  it("asks the agent to plan the merge", () => {
    expect(mergePrompt(["pa", "pb"])).toContain("proxy_merge(op='plan', proxies=['pa', 'pb'])");
  });
});
