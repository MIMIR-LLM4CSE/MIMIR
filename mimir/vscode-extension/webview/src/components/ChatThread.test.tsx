/** A long history is rendered from its tail, with the way back to the rest. */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { ChatThread, THREAD_WINDOW_STEP } from "./ChatThread";
import type { ChatMessage } from "../types";

const msgs = (n: number): ChatMessage[] =>
  Array.from({ length: n }, (_, i) => ({
    id: `m${i}`, role: i % 2 ? "assistant" : "user", kind: "text", text: `message ${i}`,
  }) as ChatMessage);

const thread = (messages: ChatMessage[]) =>
  renderToStaticMarkup(
    <ChatThread
      messages={messages}
      busy={false}
      chatThreadRef={React.createRef()}
      bottomRef={React.createRef()}
      lastUserMsgRef={React.createRef()}
      onScroll={() => {}}
      onApprovalResponse={() => {}}
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
