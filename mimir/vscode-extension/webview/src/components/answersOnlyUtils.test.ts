import { describe, it, expect } from "vitest";
import { answersOnlyView, keepInAnswersOnlyView } from "./answersOnlyUtils";
import type { ChatMessage } from "../types";

const msg = (over: Partial<ChatMessage> & Pick<ChatMessage, "id" | "kind">): ChatMessage =>
  ({ role: "agent", ...over } as ChatMessage);

describe("keepInAnswersOnlyView", () => {
  it("keeps every user message, whatever its kind", () => {
    // A queued steer is as much the user's voice as the prompt that opened the turn.
    expect(keepInAnswersOnlyView(msg({ id: "u1", role: "user", kind: "text" }))).toBe(true);
    expect(keepInAnswersOnlyView(msg({ id: "u2", role: "user", kind: "text", queued: true }))).toBe(true);
  });

  it("keeps agent prose — a committed answer and a provisional one", () => {
    expect(keepInAnswersOnlyView(msg({ id: "a1", kind: "text" }))).toBe(true);
    expect(keepInAnswersOnlyView(msg({ id: "a2", kind: "text", provisional: true }))).toBe(true);
  });

  it("keeps an error, a pending approval and a command reply", () => {
    // A failed turn's outcome, a question the agent is asking, an answer to a typed
    // command: all three address the user, so hiding one would lose information or
    // strand the turn.
    expect(keepInAnswersOnlyView(msg({ id: "e1", kind: "error" }))).toBe(true);
    expect(keepInAnswersOnlyView(msg({ id: "p1", kind: "approval" }))).toBe(true);
    expect(keepInAnswersOnlyView(msg({ id: "c1", kind: "command" }))).toBe(true);
  });

  it("hides the machinery: tool activity, reasoning and edit cards", () => {
    expect(keepInAnswersOnlyView(msg({ id: "t1", kind: "tools" }))).toBe(false);
    expect(keepInAnswersOnlyView(msg({ id: "th1", kind: "thinking" }))).toBe(false);
    expect(keepInAnswersOnlyView(msg({ id: "ed1", kind: "editing" }))).toBe(false);
  });
});

describe("answersOnlyView", () => {
  it("reduces a turn to its prompt and its answer, in order", () => {
    const conversation = [
      msg({ id: "q", role: "user", kind: "text", text: "Question" }),
      msg({ id: "th", kind: "thinking" }),
      msg({ id: "t", kind: "tools" }),
      msg({ id: "ed", kind: "editing" }),
      msg({ id: "a", kind: "text", text: "Answer" }),
    ];
    expect(answersOnlyView(conversation).map((m) => m.id)).toEqual(["q", "a"]);
  });

  it("keeps a failed turn's error, and a turn waiting on an approval", () => {
    const conversation = [
      msg({ id: "q", role: "user", kind: "text" }),
      msg({ id: "t", kind: "tools" }),
      msg({ id: "e", kind: "error", text: "boom" }),
      msg({ id: "q2", role: "user", kind: "text" }),
      msg({ id: "p", kind: "approval" }),
    ];
    expect(answersOnlyView(conversation).map((m) => m.id)).toEqual(["q", "e", "q2", "p"]);
  });
});
