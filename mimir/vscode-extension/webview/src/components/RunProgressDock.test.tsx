/** The dock card for a run whose row has scrolled out of the thread.
 *
 *  It names the same call its row does, so it has to say the same thing: a card reading
 *  "Running shell command" three lines under a row reading "Bash · running the
 *  row-display tests" is the same run described two ways.
 */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { RunProgressDock } from "./RunProgressDock";
import type { ToolActivity } from "../types";

function run(over: Partial<ToolActivity>): ToolActivity {
  return {
    id: "c1", name: "bash_run", label: "Running shell command", detail: "",
    status: "running", startedAt: 0, ...over,
  } as ToolActivity;
}

const dock = (tool: ToolActivity) =>
  renderToStaticMarkup(<RunProgressDock runs={[tool]} onFocus={() => {}} />);

describe("the run dock", () => {
  it("says what the row says", () => {
    const html = dock(run({ kind: "bash", doing: "running the row-display tests" }));
    expect(html).toContain("Bash · running the row-display tests");
    expect(html).not.toContain(">Running shell command<");
    // The precise name of what was called stays in the tooltip.
    expect(html).toContain('title="Running shell command');
  });

  it("falls back to the derived label for a run with no description", () => {
    expect(dock(run({}))).toContain(">Running shell command<");
  });

  it("keeps showing the live phase beside it", () => {
    const html = dock(run({ kind: "proxy eval", doing: "running the kernel", phase: "building solver" }));
    expect(html).toContain("Proxy eval · running the kernel");
    expect(html).toContain("building solver");
  });
});
