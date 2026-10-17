/**
 * The answers-only view: which rows of the transcript are the conversation itself.
 *
 * Every message the user addresses the agent, and every message that answers them,
 * stays; the working machinery in between — tool activity, reasoning, edit cards —
 * does not. The choice is by kind, not by content, so it holds for a restored
 * transcript exactly as it does for a live one.
 */
import type { ChatMessage } from "../types";

/**
 * Whether *m* belongs in the answers-only view.
 *
 * Kept:
 * - every user message — prompts, steers, queued steers (``role`` decides, so a
 *   future user-side kind is kept without a change here);
 * - agent prose (``kind: "text"``) — a turn's committed answer, and the
 *   ``provisional`` answer of a turn parked on a question or a dropped line;
 * - error cards (``kind: "error"``) — a failed turn's only outcome: hiding one
 *   would make the turn disappear rather than read as failed;
 * - approval cards (``kind: "approval"``) — a pending approval is a question the
 *   agent asked, and hiding it would strand a turn waiting for an answer;
 * - command replies (``kind: "command"``) — the direct answer to a session
 *   command the user typed.
 *
 * Hidden: ``tools``, ``thinking`` and ``editing`` — the step's work, not its words.
 */
export function keepInAnswersOnlyView(m: ChatMessage): boolean {
  if (m.role === "user") return true;
  switch (m.kind) {
    case "tools":
    case "thinking":
    case "editing":
      return false;
    default:
      return true;
  }
}

/** *messages* reduced to the conversation proper. */
export function answersOnlyView(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter(keepInAnswersOnlyView);
}
