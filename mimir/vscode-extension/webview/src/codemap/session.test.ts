import { describe, expect, it } from "vitest";
import { gpuBlock, kernelName, megabytes, memoryBlock, stepCard, timeAt, type Session } from "./session";

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
