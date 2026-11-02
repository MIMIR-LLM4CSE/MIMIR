/** Editing a sent message: which bubbles offer it, and what the confirm card says. */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { ChatThread } from "./ChatThread";
import { RewindConfirm } from "./MessageEditor";
import { planIsEmpty, type Rewind } from "../hooks/useRewind";
import { createChatReducer, initialChatState } from "../state/chatReducer";
import type { ChatMessage, RewindPlanMessage } from "../types";

const thread = (messages: ChatMessage[], rewind?: Partial<Rewind>) =>
  renderToStaticMarkup(
    <ChatThread
      messages={messages}
      busy={false}
      chatThreadRef={React.createRef()}
      bottomRef={React.createRef()}
      lastUserMsgRef={React.createRef()}
      onScroll={() => {}}
      onApprovalResponse={() => {}}
      rewind={{ editingId: null, start: () => {}, ...rewind } as Rewind}
    />,
  );

const plan = (extra: Partial<RewindPlanMessage> = {}): RewindPlanMessage =>
  ({ type: "rewind_plan", message_id: "m1", available: true, ...extra });

describe("editing a sent message", () => {
  it("is offered, by a double-click, on a message that opened a turn, not on a steer or an answer", () => {
    const html = thread([
      { id: "m1", role: "user", kind: "text", text: "first", opensTurn: true },
      { id: "a1", role: "assistant", kind: "text", text: "answer" },
      { id: "s1", role: "user", kind: "text", text: "steer" },
    ] as ChatMessage[]);
    expect(html.match(/message-body--editable/g)?.length).toBe(1);
    expect(html).toContain("Double-click to edit");
  });

  it("opens the editor in place of the bubble being edited", () => {
    const html = thread(
      [{ id: "m1", role: "user", kind: "text", text: "first", opensTurn: true }] as ChatMessage[],
      { editingId: "m1", plan: null, pendingText: null, working: false },
    );
    expect(html).toContain("message-editor__input");
    expect(html).not.toContain("message-body--editable");
  });

  it("the bubble a query opens carries the query's id", () => {
    const reduce = createChatReducer(() => "made");
    const state = reduce(initialChatState, { type: "submit_query", text: "go", id: "q1" });
    expect(state.messages.at(-1)).toMatchObject({ id: "q1", opensTurn: true });
  });

  it("names what is put back, stopped and left before anything happens", () => {
    const html = renderToStaticMarkup(
      <RewindConfirm
        plan={plan({
          turn_running: true,
          files: [{ path: "a.py", action: "restore" }, { path: "new.txt", action: "delete" }],
          runs: [{ job_key: "j1", kind: "shell", cancellable: true }],
          conflicts: ["b.py"],
        })}
        onConfirm={() => {}}
        onBack={() => {}}
      />,
    );
    expect(html).toContain("restore · a.py");
    expect(html).toContain("remove · new.txt");
    expect(html).toContain("j1");
    expect(html).toContain("b.py");
    expect(html).toContain("turn in progress is stopped");
  });

  it("goes without asking only when there is nothing to undo", () => {
    expect(planIsEmpty(plan())).toBe(true);
    expect(planIsEmpty(plan({ turn_running: true }))).toBe(false);
    expect(planIsEmpty(plan({ files: [{ path: "a", action: "restore" }] }))).toBe(false);
  });
});
