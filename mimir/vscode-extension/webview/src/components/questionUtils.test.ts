import { describe, expect, it } from "vitest";
import { pendingQuestion, questionAfterLoad } from "./questionUtils";

const asked = { type: "user_question" as const, id: "q1", questions: [] };

describe("pendingQuestion", () => {
  it("is attributed to the conversation on screen unless the message names one", () => {
    expect(pendingQuestion(asked, "s1", 0).sessionId).toBe("s1");
    expect(pendingQuestion({ ...asked, session_id: "s2" }, "s1", 0).sessionId).toBe("s2");
  });

  it("carries a deadline only when the question has a wall", () => {
    expect(pendingQuestion(asked, "s1", 1000).expiresAt).toBeUndefined();
    expect(pendingQuestion({ ...asked, timeout_secs: 30 }, "s1", 1000).expiresAt).toBe(31000);
  });
});

describe("questionAfterLoad", () => {
  const card = pendingQuestion(asked, "s1", 0);

  it("keeps the card on a reconnect to the conversation that asked", () => {
    expect(questionAfterLoad(card, "s1")).toBe(card);
  });

  it("drops it when another conversation comes on screen", () => {
    expect(questionAfterLoad(card, "s2")).toBeNull();
  });
});
