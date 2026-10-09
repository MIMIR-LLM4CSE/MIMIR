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
    expect(html).toContain('<span class="tool-kind">read</span>');
    expect(html).toContain('<span class="tool-doing">reading the dispatch loop</span>');
    // The derived label is no longer said, but it is still the tooltip.
    expect(html).toContain('title="Derived label"');
    expect(html).not.toContain(">Derived label<");
  });

  it("keeps the file name as a link beside a description that omits it", () => {
    const read = asRead([row({ kind: "read", doing: "reading the loop", target: TARGET })]);
    expect(read).toBe("📖 read reading the loop dispatch.py 12ms");
  });

  it("does not say the file name twice when the description names it", () => {
    const read = asRead([
      row({ kind: "edit", doing: "fixing the bound in dispatch.py", target: TARGET }),
    ]);
    expect(read).toBe("✏️ edit fixing the bound in dispatch.py 12ms");
  });

  it("drops the command preview when a description says what the run is for", () => {
    // The command is in the IN pane below, which opens itself on a successful run.
    const read = asRead([row({
      kind: "shell", doing: "running the row-display tests", detail: "pytest -q",
      exec: { command: "pytest -q", stdout: "14 passed", stderr: "", returncode: 0 },
    })]);
    expect(read).toBe("💻 shell running the row-display tests 12ms ▾ IN $ pytest -q OUT 14 passed");
  });

  it("brings the command preview back on a row with no description", () => {
    // A failed row is collapsed: without the preview it would say nothing but "shell".
    const read = asRead([row({
      kind: "shell", status: "error", detail: "pytest -q", error: "boom",
    })]);
    expect(read).toBe("✕ 💻 shell pytest -q 12ms ▸");
  });

  it("renders a row recorded before the family existed", () => {
    const read = asRead([row({ kind: undefined, doing: undefined, detail: "legacy.py" })]);
    expect(read).toBe("🔧 tool legacy.py 12ms");
  });

  it("draws the GitHub mark in the icon slot", () => {
    const read = asRead([row({ kind: "github", doing: "fetching the workflow file" })]);
    expect(read).toBe("[github-mark] github fetching the workflow file 12ms");
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
    expect(asRead(stored[0].tools!)).toBe("[github-mark] github fetching the workflow file 12ms");
  });

  it("tells families apart that the old name-keyed table collapsed into one glyph", () => {
    const read = asRead([
      row({ id: "1", kind: "eval", doing: "computing the residual norm" }),
      row({ id: "2", kind: "symbolic", doing: "solving for the steady state" }),
    ]);
    expect(read).toBe(
      "🧮 eval computing the residual norm 12ms 📐 symbolic solving for the steady state 12ms"
    );
  });
});
