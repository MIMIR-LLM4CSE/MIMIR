/** What a tool row actually puts on screen.
 *
 *  Rendered rather than asserted piecemeal, because the thing being pinned is a
 *  layout decision: which of the four slots — icon, family, description, arg preview —
 *  appear together, and which yield to each other. The rules only make sense as a row.
 */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { ToolActivityList } from "./ToolActivityList";
import type { ChatMessage, ToolActivity } from "../types";
import { pruneForStorage } from "./transcriptUtils";

function row(p: Partial<ToolActivity>): ToolActivity {
  return {
    id: "c1", name: "x", label: "Derived label",
    detail: "", status: "ok", startedAt: 0, durationMs: 12, ...p,
  } as ToolActivity;
}

/** The row as a reader sees it: markup stripped, the GitHub mark named. */
function asRead(tools: ToolActivity[]): string {
  return renderToStaticMarkup(<ToolActivityList tools={tools} />)
    .replace(/<svg[\s\S]*?<\/svg>/g, "[github-mark]")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const TARGET = { path: "/w/dispatch.py", name: "dispatch.py" };

describe("the tool row", () => {
  it("shows the family, then what the call is doing", () => {
    const html = renderToStaticMarkup(
      <ToolActivityList tools={[row({ kind: "read", doing: "reading the dispatch loop" })]} />
    );
    // The family at full strength, the description dim — the two classes carry that.
    expect(html).toContain('<span class="tool-kind">Read</span>');
    expect(html).toContain('<span class="tool-doing">reading the dispatch loop</span>');
    // The derived label is no longer said, but it is still the tooltip.
    expect(html).toContain('title="Derived label"');
    expect(html).not.toContain(">Derived label<");
  });

  it("keeps the file name as a link beside a description that omits it", () => {
    const read = asRead([row({ kind: "read", doing: "reading the loop", target: TARGET })]);
    expect(read).toBe("📖 Read reading the loop dispatch.py 12ms");
  });

  it("does not say the file name twice when the description names it", () => {
    const read = asRead([
      row({ kind: "edit", doing: "fixing the bound in dispatch.py", target: TARGET }),
    ]);
    expect(read).toBe("✏️ Edit fixing the bound in dispatch.py 12ms");
  });

  it("shows the salient argument beside the description", () => {
    // What the call is for, and what it is for *on*. These reached the row through the
    // server's label template until the row stopped showing the label, and the verdict,
    // the url and the job id went off screen with it.
    expect(asRead([row({ kind: "verdict", doing: "recording what the suite showed",
                         detail: "pass" })]))
      .toBe("⚖️ Verdict recording what the suite showed pass 12ms");
    expect(asRead([row({ kind: "web", doing: "fetching the solver docs",
                         detail: "api.github.com/repos/foo/bar/x.py" })]))
      .toBe("🌐 Web fetching the solver docs api.github.com/repos/foo/bar/x.py 12ms");
    expect(asRead([row({ kind: "slurm", doing: "cancelling the stuck allocation",
                         detail: "12345" })]))
      .toBe("🛰️ Slurm cancelling the stuck allocation 12345 12ms");
  });

  it("never says on the line what the panel below carries in full", () => {
    const read = asRead([row({
      kind: "bash", doing: "running the row-display tests", detail: "pytest -q",
      exec: { command: "pytest -q", stdout: "14 passed", stderr: "", returncode: 0 },
    })]);
    expect(read).toBe("💻 Bash running the row-display tests 12ms ▾ IN $ pytest -q OUT 14 passed");
  });

  it("keeps the command off a failed row too, where the panel is closed", () => {
    // The IN half is sent with the `tool_call` event, so the panel exists from the
    // moment the call is made — a failed row is collapsed, not empty. Cropped at a
    // share of the line, the command was a fragment pretending to be one.
    const read = asRead([row({
      kind: "bash", status: "error", error: "boom", doing: "running the whole suite",
      detail: "pytest -q mimir/tests/test_very_long_name.py",
      exec: { command: "pytest -q mimir/tests/test_very_long_name.py", stdout: "", stderr: "" },
    })]);
    expect(read).toBe("✕ 💻 Bash running the whole suite 12ms ▸");
  });

  it("gives a preview its own tooltip, since it is the slot that gets cut", () => {
    // Capped at a share of the line and ellipsised there, and the end of a url or a
    // repository path is what the row was for.
    const html = renderToStaticMarkup(
      <ToolActivityList tools={[row({
        kind: "github", doing: "seeing which versions CI covers",
        detail: "MIMIR-LLM4CSE/MIMIR/.github/workflows/ci.yml",
      })]} />
    );
    expect(html).toContain('title="MIMIR-LLM4CSE/MIMIR/.github/workflows/ci.yml"');
  });

  it("falls back to the derived label when the model wrote no description", () => {
    // The family alone is too coarse to read: four tools answer to "Todo", so a bare
    // `Todo` row left no way to tell a checklist being written from one being read.
    const read = asRead([row({
      kind: "todo", doing: undefined, label: "Updating checklist step 2",
    })]);
    expect(read).toBe("📋 Todo Updating checklist step 2 12ms");
  });

  it("keeps a saved plan and the checklist apart", () => {
    // One server holds both, and they are different objects.
    const read = asRead([
      row({ id: "1", kind: "plan", label: "Recording the plan: the refonte" }),
      row({ id: "2", kind: "todo", label: "Writing the checklist", detail: "4 steps" }),
    ]);
    expect(read).toBe(
      "🗺️ Plan Recording the plan: the refonte 12ms 📋 Todo Writing the checklist 4 steps 12ms"
    );
  });

  it("renders a row recorded before the family existed", () => {
    const read = asRead([row({ kind: undefined, doing: undefined, detail: "legacy.py" })]);
    expect(read).toBe("🔧 Tool Derived label legacy.py 12ms");
  });

  it("draws the GitHub mark in the icon slot", () => {
    const read = asRead([row({ kind: "github", doing: "fetching the workflow file" })]);
    expect(read).toBe("[github-mark] GitHub fetching the workflow file 12ms");
  });

  it("survives the round trip through the stored transcript", () => {
    // The transcript is stored as JSON and restored rows are rendered as they come
    // back, never rebuilt. So nothing on a row may be unserialisable — which is why
    // the icon is derived from the family at render time instead of carried: the
    // GitHub mark is an element, and JSON would hand it back as a dead object React
    // refuses to render, taking the whole transcript down with it.
    const stored = JSON.parse(JSON.stringify(
      pruneForStorage([{
        id: "m1", role: "agent", kind: "tools",
        tools: [row({ kind: "github", doing: "fetching the workflow file" })],
      }] as ChatMessage[])
    )) as ChatMessage[];
    expect(asRead(stored[0].tools!)).toBe("[github-mark] GitHub fetching the workflow file 12ms");
  });

  it("tells families apart that the old name-keyed table collapsed into one glyph", () => {
    const read = asRead([
      row({ id: "1", kind: "eval", doing: "computing the residual norm" }),
      row({ id: "2", kind: "symbolic", doing: "solving for the steady state" }),
    ]);
    expect(read).toBe(
      "🧮 Eval computing the residual norm 12ms 📐 Symbolic solving for the steady state 12ms"
    );
  });
});
