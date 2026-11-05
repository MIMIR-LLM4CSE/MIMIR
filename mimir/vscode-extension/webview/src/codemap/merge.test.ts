import { describe, expect, it } from "vitest";
import { gainOf, mergeGraph, outcome, type MergeState } from "./merge";
import { mergePrompt, picked, pickerHtml, pickLabel, reachable, ticked, type ProxyMeta } from "./picker";

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
const done: MergeState = {
  phase: "done", sources, pairs,
  merged: [
    { proxy: "pa", conflicts: [], resolution: "" },
    { proxy: "pc", conflicts: ["a.py"], resolution: "pa's tiling kept, as the user chose" },
    { proxy: "pb", conflicts: [], resolution: "" },
  ],
  dropped: [], conflict: null, question: null,
  kernels: [
    { proxy: "pa", case: "a", alone: 0.8, in_merge: 0.8, interaction: 1, flagged: false },
    { proxy: "pc", case: "a", alone: 0.85, in_merge: 1.0, interaction: 1.18, flagged: true },
    { proxy: "pb", case: "a", alone: 0.9, in_merge: 0.85, interaction: 0.94, flagged: false },
  ],
  whole: { relative: 0.88 },
};
const stopped: MergeState = {
  phase: "merging", sources, pairs, merged: [{ proxy: "pa", conflicts: [], resolution: "" }],
  dropped: [], conflict: { proxy: "pc", files: ["a.py"] }, question: null, kernels: null, whole: null,
};

describe("a merge's outcome per source", () => {
  it("says which kernel kept its gain and which lost it", () => {
    expect(outcome(done, "pa").state).toBe("kept");
    expect(outcome(done, "pc").state).toBe("flagged");
    expect(outcome(done, "pb").kernel?.in_merge).toBe(0.85);
  });

  it("follows the merge under way", () => {
    expect(outcome(stopped, "pa").state).toBe("merged");
    expect(outcome(stopped, "pc").state).toBe("conflict");
    expect(outcome(stopped, "pb").state).toBe("pending");
    const waiting = { ...stopped, conflict: null, question: { proxy: "pc", question: "pa's tiling or pc's?" } };
    expect(outcome(waiting, "pc").state).toBe("waiting");
    expect(outcome({ ...done, dropped: ["pc"] }, "pc").state).toBe("dropped");
  });

  it("is a forecast without a merge", () => {
    expect(outcome(null, "pa").state).toBe("forecast");
  });
});

describe("the merge graph", () => {
  it("draws a resolved conflict, each kernel against alone and the whole", () => {
    const html = mergeGraph(done);
    expect(html).toContain('class="arc resolved"');
    expect(html).toContain("pa's tiling kept");
    expect(html).toContain("measured 12%");
    expect(html).toContain("lost gain: pc");
    expect(html).toMatch(/class="mlink flagged intf"/);
    expect(html).toMatch(/class="mlink kept syn"/);
    expect(html).toContain("conflict resolved");
    expect(html).not.toContain("conflict: to resolve");
  });

  it("says where a merge stopped and what it waits for", () => {
    expect(mergeGraph(stopped)).toContain("stopped on a conflict: pc on a.py");
    expect(mergeGraph(stopped)).toContain('class="arc conflict"');
    const waiting = { ...stopped, conflict: null, question: { proxy: "pc", question: "pa's tiling or pc's?" } };
    expect(mergeGraph(waiting)).toContain("waiting for your answer: pa's tiling or pc's?");
  });

  it("is a forecast for proxies ticked, with its conflicts to resolve", () => {
    const html = mergeGraph({ sources, pairs, refused: [{ proxy: "pd", why: "no run accepted beyond its baseline" }] }, 600);
    expect(html).toContain("Merge forecast");
    expect(html).toContain("1</b> conflict to resolve");
    expect(html).toContain("left out: pd");
    expect(html).toContain("conflict: to resolve when merged");
    expect(html).not.toContain("better in the merge");      // nothing measured, no such link
    // Its Merge button asks the agent for every proxy ticked; the agent says what cannot be merged.
    expect(html).toContain(">Merge 4</button>");
    expect(html).toContain("proxy_merge(op='plan', proxies=['pa', 'pc', 'pb', 'pd'])");
  });

  it("offers Merge even when the server would refuse, and draws no legend for nothing", () => {
    const html = mergeGraph({ sources: [sources[2]], pairs: [], refused: [{ proxy: "pz", why: "no run accepted beyond its baseline" }],
      refusal: "A merge needs two proxies at least that kept an accepted run." });
    expect(html).toContain(">Merge 2</button>");
    expect(html).not.toContain("mlegend");
  });

  it("warns of proxies that ran together in one tree", () => {
    const html = mergeGraph({ sources, pairs: [{ ...pairs[0], together: true }, pairs[1], pairs[2]] });
    expect(html).toContain("ran together: pa + pc");
    expect(html).toContain("ran at the same time in one tree");
    expect(mergeGraph({ sources, pairs })).not.toContain("ran together");
  });

  it("has no Merge button once the merge runs", () => {
    expect(mergeGraph(done)).not.toContain("merge-go");
  });

  it("writes a gain from a relative", () => {
    expect(gainOf(0.8)).toBe("20%");
    expect(gainOf(1.05)).toBe("−5.0%");
    expect(gainOf(null)).toBe("–");
  });
});

const meta: ProxyMeta[] = [
  { name: "pa", branch: "mimir/sub-1", metric: "time_s", goal: "min", steps: 4, gain: 0.2, active: true, frozen: "", merge: false, sources: [] },
  { name: "pb", branch: "mimir/sub-2", metric: "time_s", goal: "min", steps: 2, gain: 0.1, active: false, frozen: "", merge: false, sources: [] },
  { name: "pz", branch: "", metric: "time_s", goal: "min", steps: 3, gain: null, active: false, frozen: "", merge: false, sources: [] },
  { name: "merge-1", branch: "mimir/sub-3", metric: "time_s", goal: "min", steps: 3, gain: 0.25, active: false, frozen: "", merge: true, sources: ["pa", "pb"] },
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

  it("offers Merge as soon as two proxies are ticked", () => {
    const html = pickerHtml(meta, "pa,pb", "pa", true, new Set(["pa", "pb"]));
    expect(html).toContain("Merge 2");
    expect(html).not.toMatch(/data-merge disabled/);
    expect(pickerHtml(meta, "", "pa", true, new Set(["pa"]))).toMatch(/data-merge disabled/);
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

describe("the pick and the answer", () => {
  it("reads whose steps an answer holds as a pick writes it", () => {
    expect(picked({ selected: ["pa", "pb"] })).toBe("pa,pb");
    expect(picked({ combined: true })).toBe("*");
    expect(picked({ proxy: "pa", active: "pa" })).toBe("");
    expect(picked({ proxy: "pb", active: "pa" })).toBe("pb");
  });

  it("asks only for what the server still offers", () => {
    expect(reachable("pa", ["pa", "pb"], "pa")).toBe("");
    expect(reachable("gone", ["pa", "pb"], "pa")).toBe("");
    expect(reachable("pa,gone,pb", ["pa", "pb"], "")).toBe("pa,pb");
    expect(reachable("pb,gone", ["pa", "pb"], "pa")).toBe("pb");
    expect(reachable("*", ["pa"], "pa")).toBe("");
  });
});
