/** A long history is rendered from its tail, with the way back to the rest. */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { ChatThread, THREAD_WINDOW_STEP } from "./ChatThread";
import type { ChatMessage, ToolActivity } from "../types";

const msgs = (n: number): ChatMessage[] =>
  Array.from({ length: n }, (_, i) => ({
    id: `m${i}`, role: i % 2 ? "assistant" : "user", kind: "text", text: `message ${i}`,
  }) as ChatMessage);

type Extra = Partial<Parameters<typeof ChatThread>[0]>;

const thread = (messages: ChatMessage[], extra: Extra = {}) =>
  renderToStaticMarkup(
    <ChatThread
      messages={messages}
      busy={false}
      chatThreadRef={React.createRef()}
      bottomRef={React.createRef()}
      lastUserMsgRef={React.createRef()}
      onScroll={() => {}}
      onApprovalResponse={() => {}}
      {...extra}
    />,
  );

describe("the thread window", () => {
  it("renders a short conversation whole", () => {
    const html = thread(msgs(10));
    expect(html).toContain("message 0");
    expect(html).not.toContain("Show earlier messages");
  });

  it("renders only the tail of a long one, and says how much is hidden", () => {
    const total = THREAD_WINDOW_STEP + 30;
    const html = thread(msgs(total));
    expect(html).toContain(`message ${total - 1}`);
    expect(html).toContain(`message ${30}`);
    expect(html).not.toContain(">message 29<");
    expect(html).toContain("Show earlier messages (30 hidden)");
  });
});

// The answers-only view: the conversation proper stays, the machinery goes.
// Tool rows, reasoning panels and edit cards are hidden whether frozen in the
// transcript or still in flight; the draft — the answer being written — does not.
describe("the answers-only view", () => {
  const conversation: ChatMessage[] = [
    { id: "q", role: "user", kind: "text", text: "What is 2+2?" } as ChatMessage,
    { id: "th", role: "agent", kind: "thinking", text: "adding it up" } as ChatMessage,
    {
      id: "t", role: "agent", kind: "tools",
      tools: [{ id: "c1", name: "evaluate", label: "Evaluating", detail: "", status: "ok", startedAt: 0, doing: "searching for the off-by-one bound" } as ToolActivity],
    } as ChatMessage,
    { id: "ed", role: "agent", kind: "editing", diffs: [{ file: "solver.py", patch: "+fixed" }] } as ChatMessage,
    { id: "a", role: "agent", kind: "text", text: "It is 4." } as ChatMessage,
  ];

  it("renders everything by default", () => {
    const html = thread(conversation);
    expect(html).toContain("What is 2+2?");
    expect(html).toContain("searching for the off-by-one bound");
    expect(html).toContain("adding it up");
    expect(html).toContain("solver.py");
    expect(html).toContain("It is 4.");
  });

  it("leaves only the prompts and answers when answersOnly is set", () => {
    const html = thread(conversation, { answersOnly: true });
    expect(html).toContain("What is 2+2?");
    expect(html).toContain("It is 4.");
    expect(html).not.toContain("searching for the off-by-one bound");
    expect(html).not.toContain("adding it up");
    expect(html).not.toContain("solver.py");
  });

  it("keeps the draft, and hides the live machinery of the step in flight", () => {
    const liveTool = {
      id: "c1", name: "evaluate", label: "Evaluating", detail: "",
      status: "running", startedAt: 0, seq: 1, doing: "running the row-display tests",
    } as ToolActivity;
    const liveThinking = [{ id: "b1", text: "wondering aloud", done: false, collapsed: false, startedAt: 0, seq: 1 }];
    const html = thread([], {
      answersOnly: true,
      draft: "The answer, so far",
      draftSeq: 1,
      liveToolCalls: [liveTool],
      liveThinkingBlocks: liveThinking,
    });
    expect(html).toContain("The answer, so far");
    expect(html).not.toContain("running the row-display tests");
    expect(html).not.toContain("wondering aloud");
  });

  it("does not disturb a long history's window count", () => {
    // The window is computed over the filtered list, so hiding machinery can only
    // shrink what is rendered — never push live rows past the window edge.
    const html = thread(msgs(THREAD_WINDOW_STEP + 30), { answersOnly: true });
    expect(html).toContain(`message ${THREAD_WINDOW_STEP + 29}`);
    expect(html).toContain("Show earlier messages");
  });
});
