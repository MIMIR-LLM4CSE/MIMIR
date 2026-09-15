import { describe, it, expect } from "vitest";
import { orderLiveStream } from "./liveStreamUtils";
import type { ToolActivity } from "../types";

function tool(id: string, seq: number, parentId?: string): ToolActivity {
  return {
    id, name: "bash_run", icon: "›", label: id, detail: "",
    status: "running", startedAt: 0, seq, parentId,
  };
}

const block = (seq: number) => ({ id: `b${seq}`, seq });

/** Compact shape of the result: the kind of each entry, in order. */
const shape = (entries: ReturnType<typeof orderLiveStream>[number][]) =>
  entries.map((e) => (e.kind === "tools" ? `tools(${e.tools.map((t) => t.id).join(",")})` : e.kind));

describe("orderLiveStream", () => {
  it("keeps prose above a tool row that arrived after it", () => {
    // The reported symptom: the card rendered above the paragraph while the step
    // was live, then jumped below it when the step ended.
    const entries = orderLiveStream({
      thinking: [], tools: [tool("t1", 5)], draftSeq: 3,
    });
    expect(shape(entries)).toEqual(["draft", "tools(t1)"]);
  });

  it("keeps prose below a tool row that ran before it", () => {
    const entries = orderLiveStream({
      thinking: [], tools: [tool("t1", 2)], draftSeq: 7,
    });
    expect(shape(entries)).toEqual(["tools(t1)", "draft"]);
  });

  it("groups consecutive tool rows into one card", () => {
    const entries = orderLiveStream({
      thinking: [], tools: [tool("t1", 2), tool("t2", 3), tool("t3", 4)], draftSeq: null,
    });
    expect(shape(entries)).toEqual(["tools(t1,t2,t3)"]);
  });

  it("splits a tool group around the reasoning that interrupted it", () => {
    const entries = orderLiveStream({
      thinking: [block(3)], tools: [tool("t1", 2), tool("t2", 4)], draftSeq: null,
    });
    expect(shape(entries)).toEqual(["tools(t1)", "thinking", "tools(t2)"]);
  });

  it("interleaves reasoning, tools and prose in arrival order", () => {
    const entries = orderLiveStream({
      thinking: [block(1), block(5)], tools: [tool("t1", 2)], draftSeq: 3,
    });
    expect(shape(entries)).toEqual(["thinking", "tools(t1)", "draft", "thinking"]);
  });

  it("keeps a sub-agent row with its parent, reasoning in between or not", () => {
    // The child inherits the parent's stamp and is spliced in after it, so the
    // family cannot be split across two cards by a block that arrived meanwhile.
    const entries = orderLiveStream({
      thinking: [block(4)],
      tools: [tool("p", 2), tool("c", 2, "p"), tool("t2", 6)],
      draftSeq: null,
    });
    expect(shape(entries)).toEqual(["tools(p,c)", "thinking", "tools(t2)"]);
  });

  it("places an unstamped row first rather than dropping it", () => {
    // Rows restored from a saved session carry no stamp.
    const { seq: _drop, ...unstamped } = tool("old", 0);
    const entries = orderLiveStream({
      thinking: [block(1)], tools: [unstamped as ToolActivity], draftSeq: null,
    });
    expect(shape(entries)).toEqual(["tools(old)", "thinking"]);
  });

  it("returns nothing when the step produced nothing", () => {
    expect(orderLiveStream({ thinking: [], tools: [], draftSeq: null })).toEqual([]);
  });
});
