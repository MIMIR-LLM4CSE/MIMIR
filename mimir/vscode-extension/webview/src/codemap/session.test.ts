import { describe, expect, it } from "vitest";
import { countersBlock, fnParts, functionsBlock, gpuBlock, kernelBlock, kernelName, megabytes, memoryBlock, stepCard, strip, timeAt, topFunctions, type Session } from "./session";

const step = (n: number, verdict: string, profile: Session["steps"][number]["profile"]) => ({
  n, run_id: "r" + n, axis: "", verdict, value: 1, incumbent_value: 1, delta: 0,
  changed: [], added: [], removed: [], profile,
});

const session: Session = {
  proxy: "solver", metric: "time_s", goal: "min",
  baseline: {
    run_id: "r0", value: 10,
    profile: { tool: "py-spy", files: { "a.cu": 0.5 }, gpu: { tool: "nsys", files: { "a.cu": 0.6 }, busy: 0.6 } },
  },
  steps: [
    step(1, "accept", { tool: "py-spy", files: { "a.cu": 0.4 }, gpu: { tool: "nsys", files: { "a.cu": 0.3 }, busy: 0.3 } }),
    step(2, "accept", { tool: "py-spy", files: { "a.cu": 0.4 }, gpu: { tool: "nsys", unavailable: "nsys is not on PATH" } }),
  ],
};

describe("timeAt with a GPU", () => {
  it("shows a kept step's GPU time beside the one it improved on", () => {
    const t = timeAt(session, 1, "step")!;
    expect(t.gpuFiles).toEqual({ "a.cu": 0.3 });
    expect(t.gpuBefore).toEqual({ "a.cu": 0.6 });
  });

  it("has no GPU bars when the step's GPU could not be traced", () => {
    const t = timeAt(session, 2, "step")!;
    expect(t.gpuFiles).toBeNull();
    expect(t.gpuBefore).toBeNull();
    expect(t.files).toEqual({ "a.cu": 0.4 });
  });

  it("compares nothing since the baseline", () => {
    expect(timeAt(session, 1, "all")!.gpuBefore).toBeNull();
  });
});

describe("a profile whose CPU side could not be sampled", () => {
  const gpuOnly: Session = {
    ...session,
    steps: [step(1, "accept", { tool: "", unavailable: "perf is not allowed here",
                                gpu: { tool: "nsys", files: { "a.cu": 0.3 }, busy: 0.3 },
                                memory: { tool: "heaptrack", peak_mb: 12, files: { "a.cu": 1 } } })],
  };

  it("draws its GPU alone", () => {
    const t = timeAt(gpuOnly, 1, "step")!;
    expect(t.from).toBe("#1");
    expect(t.cpuUnavailable).toBe("perf is not allowed here");
    expect(t.files).toEqual({});
    expect(t.before).toBeNull();
    expect(t.gpuFiles).toEqual({ "a.cu": 0.3 });
    expect(t.gpuBefore).toEqual({ "a.cu": 0.6 });
    expect(t.memory?.peak_mb).toBe(12);
  });

  it("says so in the step's card, with what was taken", () => {
    const card = stepCard(gpuOnly, 1, (f, r) => f + " " + r);
    expect(card).toContain("CPU not profiled");
    expect(card).toContain("perf is not allowed here");
    expect(card).toContain("On the GPU (nsys)");
    expect(card).toContain("heap peak 12 MB");
  });
});

describe("counters in a step's card", () => {
  it("reads the run's hardware counters and DRAM traffic", () => {
    const html = countersBlock({ tool: "perf stat", ipc: 0.82, cache_miss_rate: 0.31, branch_miss_rate: 0.004,
                                 dram_gbytes_per_s: 18.4, bandwidth_fraction: 0.76 });
    expect(html).toContain("IPC 0.82 · cache misses 31% · branch misses 0.4%");
    expect(html).toContain("DRAM 18.4 GB/s (76% of peak)");
    expect(countersBlock({ unavailable: "perf is not allowed here" })).toContain("perf is not allowed here");
  });

  it("reads one kernel's throughput and what limits it", () => {
    const html = kernelBlock({ tool: "ncu", name: "void relax(float *)", sm_pct: 21, dram_pct: 84.6, occupancy_pct: 50, bound: "memory" });
    expect(html).toContain("Kernel relax (ncu)");
    expect(html).toContain("compute 21% · memory 85% · occupancy 50% of peak");
    expect(html).toContain("memory-bound");
  });

  it("shows them in the card of a step that took them", () => {
    const counted: Session = { ...session, steps: [step(1, "accept", { tool: "perf", files: { "a.cu": 0.4 },
                                                                     counters: { tool: "perf stat", ipc: 1.5 } })] };
    expect(stepCard(counted, 1, (f, r) => f + " " + r)).toContain("IPC 1.50");
    expect(timeAt(counted, 1, "step")!.counters?.ipc).toBe(1.5);
  });
});

describe("the GPU in a step's card", () => {
  it("says why it was not traced, or what its kernels took", () => {
    expect(gpuBlock({ tool: "nsys", unavailable: "nsys is not on PATH" })).toContain("nsys is not on PATH");
    const html = gpuBlock({ tool: "nsys", busy: 0.5, idle: 0.3, memcpy: { HtoD: 0.2 },
                            kernels: { "void stencil<float>(float *)": { share: 0.5 } } });
    expect(html).toContain("kernels 50% · copies 20% · idle 30%");
    expect(html).toContain("stencil&lt;float&gt;");
    expect(kernelName("void k::relax(float *, int)")).toBe("k::relax");
  });
});

describe("memory in a step's card", () => {
  it("names the step's peak against its incumbent's, red when it grew", () => {
    const grown: Session = { ...session, steps: [{ ...session.steps[0], peak_rss_mb: 1536, incumbent_peak_rss_mb: 1024 }] };
    const card = stepCard(grown, 1, (f, r) => f + " " + r);
    expect(card).toContain("1.5 GB (+50%)");
    expect(card).toContain('class="is-danger">1.5 GB');
  });

  it("sums an MPI run's ranks and names the largest", () => {
    const mpi: Session = { ...session, steps: [{ ...session.steps[0], peak_rss_mb: 900, incumbent_peak_rss_mb: 1000,
                                                 peak_rss_total_mb: 7200, incumbent_peak_rss_total_mb: 4000 }] };
    const card = stepCard(mpi, 1, (f, r) => f + " " + r);
    expect(card).toContain('class="is-danger">7.0 GB summed, largest process 900 MB (+80%)');
  });

  it("lists where the heap's peak was allocated", () => {
    const html = memoryBlock({ tool: "massif", peak_mb: 28.6, files: { "src/solver.f90": 0.67 } }, (f, r) => f + " " + r);
    expect(html).toContain("heap peak 29 MB");
    expect(html).toContain("src/solver.f90 67%");
    expect(megabytes(512)).toBe("512 MB");
  });

  it("shows the GPU's peak memory and what was copied", () => {
    const html = gpuBlock({ tool: "nsys", busy: 0.5, peak_device_mb: 3072, memcpy_mb: { HtoD: 2048 } });
    expect(html).toContain("peak device memory 3.0 GB · copied HtoD 2.0 GB");
  });
});

describe("every proxy at once", () => {
  const all: Session = {
    proxy: "solver, mesh", metric: "vs baseline", goal: "min", combined: true,
    baseline: { run_id: "", value: 1, profile: null },
    steps: [
      { ...step(1, "accept", null), proxy: "solver", metric: "time_s", value: 0.8, incumbent_value: 1, delta: -0.2 },
      { ...step(2, "accept", null), proxy: "mesh", metric: "gflops", value: 0.5, incumbent_value: 1, delta: -0.5 },
      { ...step(3, "reject", null), proxy: "solver", metric: "time_s", value: 0.9, incumbent_value: 0.8, delta: 0.125 },
    ],
  };

  it("sums up each proxy against its own baseline, and draws no single best line", () => {
    const html = strip(all, 3, "step");
    expect(html).toContain("solver <b>−20%</b> · mesh <b>−50%</b> since their baselines");
    expect(html).not.toContain('class="best"');
    expect(html).toContain("mesh · −50%");
  });

  it("names a step's proxy and its metric, and profiles the proxy it ran on", () => {
    const card = stepCard(all, 2, (f, r) => f + " " + r);
    expect(card).toContain("<span>proxy</span>");
    expect(card).toContain("gflops × baseline");
    expect(card).toContain("proxy_name='mesh'");
  });

  it("offers a profile on each proxy's latest step only", () => {
    expect(stepCard(all, 1, (f, r) => f + " " + r)).not.toContain("data-prompt");
    expect(stepCard(all, 3, (f, r) => f + " " + r)).toContain("proxy_name='solver'");
  });
});

describe("the session's chart", () => {
  const run: Session = {
    proxy: "solver", metric: "time_s", goal: "min",
    baseline: { run_id: "r0", value: 10, profile: null },
    steps: [
      { ...step(1, "accept", null), value: 8, incumbent_value: 10, delta: -0.2 },
      { ...step(2, "reject", null), value: 9, incumbent_value: 8, delta: 0.125 },
    ],
  };

  it("draws the baseline, the best as it went, and kept and ruled-out steps apart", () => {
    const html = strip(run, 1, "step", 600);
    expect(html).toContain('class="base-line"');
    expect(html).toContain('class="best"');
    expect(html).toContain('class="dot kept is-cur" data-step="1"');
    expect(html).toContain('class="dot rej is-after" data-step="2"');
  });

  it("has a chip per step, the one shown current", () => {
    const html = strip(run, 2, "step");
    expect(html).toContain('data-step="2" aria-current="true"');
    expect(html).toContain("#2 ✕ <b>+13%</b>");
    expect(html).toContain("#1 <b>−20%</b>");
  });
});

describe("functions", () => {
  const fns = {
    "src/solver.f90::solver::relax": { self: 0.3, total: 0.5, line: 12 },
    "src/grid.cpp::build": { self: 0.1, total: 0.1 },
    "src/solver.f90::residual": { self: 0.2, total: 0.2, line: 40 },
  };

  it("keeps a Fortran name whole: the file ends at the first ::", () => {
    expect(fnParts("src/solver.f90::solver::relax")).toEqual({ file: "src/solver.f90", name: "solver::relax" });
  });

  it("lists the costliest by their own time, of the files asked for", () => {
    expect(topFunctions(fns, 2).map(([k]) => k)).toEqual(["src/solver.f90::solver::relax", "src/solver.f90::residual"]);
    expect(topFunctions(fns, 5, (f) => f === "src/grid.cpp").map(([k]) => k)).toEqual(["src/grid.cpp::build"]);
  });

  it("gives each row its file, name and line", () => {
    const html = functionsBlock(fns, (f, r, line, label) => [f, r, line, label].join("|"), 1);
    expect(html).toContain("src/solver.f90|30%|12|solver::relax");
  });
});
