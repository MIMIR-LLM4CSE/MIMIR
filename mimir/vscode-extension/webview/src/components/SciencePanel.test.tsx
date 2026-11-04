/** The drawer renders what the servers said; a section that offers a view gets the
 *  button that opens it, and only that section. */
import { beforeAll, describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { SciencePanel } from "./SciencePanel";
import type { PanelSection } from "../types";

beforeAll(() => {
  // The drawer reads the window's width for its own; the static render has no window.
  (globalThis as { window?: unknown }).window ??= { innerWidth: 1200 };
});

const architecture: PanelSection = {
  section: "Architecture",
  title: "Architecture",
  lines: [{ label: "code", value: "271 files · 56 modules" }],
  open: "code_map",
  map: {
    components: [
      { id: "src/solver", name: "solver", tier: 1, files: 52, breaks: 0, cycles: 1 },
      { id: "src/core", name: "core", tier: 0, files: 6, breaks: 1, cycles: 0 },
    ],
    links: [{ from: "src/solver", to: "src/core", n: 12, breaks: 0 }],
  },
};
const machine: PanelSection = {
  section: "Machine",
  title: "Machine",
  lines: [{ label: "cpu", value: "x86_64" }],
};

function render(sections: PanelSection[]): string {
  const none = () => undefined;
  return renderToStaticMarkup(
    <SciencePanel sections={sections} runs={[]} onCancelRun={none} onCall={none} onRefresh={none}
      onReload={none} refreshing={false} agentReady onPrompt={none} onOpenView={none} />,
  );
}

describe("a section that offers a view", () => {
  it("shows the button that opens the code map", () => {
    expect(render([architecture])).toContain(">Open map</button>");
  });

  it("draws its components, highest tier first, with their flags", () => {
    const html = render([architecture]);
    expect(html.indexOf("solver")).toBeLessThan(html.indexOf(">core<"));
    expect(html).toContain("↻ 1");
    expect(html).toContain("✕ 1");
  });

  it("is the only one that does", () => {
    expect(render([machine])).not.toContain("Open map");
  });
});

describe("the optimisation's curve", () => {
  const optimisation: PanelSection = {
    section: "Optimisation",
    title: "Optimisation",
    lines: [{ label: "proxy", value: "solver" }],
    open: "code_map",
    chart: {
      metric: "time_s", goal: "min", baseline: 10, best: 8, tried: 3, kept: 1,
      points: [
        { n: 1, run: "r1", axis: "tiling", value: 8, kind: "kept", best: 8 },
        { n: 2, run: "r2", axis: "simd", value: 9, kind: "not kept", best: 8 },
        { n: 3, run: "r3", axis: "omp", value: null, kind: "failed", best: 8 },
      ],
    },
  };

  it("heads the section with the best and what it won over the baseline", () => {
    const html = render([optimisation]);
    expect(html).toContain("best time_s");
    expect(html).toContain(">8</span>");
    expect(html).toContain("−20% · 1.25×");
    expect(html).toContain("3 tried · 1 kept");
  });

  it("draws every run, the best so far and the baseline", () => {
    const html = render([optimisation]);
    expect(html.match(/science-curve-point/g)).toHaveLength(3);
    expect(html).toContain("science-curve-best-line");
    expect(html).toContain("science-curve-baseline");
    expect(html).toContain("is-failed");
  });

  it("reads out the latest run until another is hovered", () => {
    expect(render([optimisation])).toContain("#3 · omp");
  });

  it("draws nothing for a section without a curve", () => {
    expect(render([machine])).not.toContain("science-curve");
  });
});
