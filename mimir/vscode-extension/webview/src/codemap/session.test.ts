import { describe, expect, it } from "vitest";
import { gpuBlock, kernelName, megabytes, memoryBlock, stepCard, strip, timeAt, type Session } from "./session";

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
    expect(html).not.toContain('class="line"');
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
