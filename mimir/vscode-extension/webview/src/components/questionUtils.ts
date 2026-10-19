// The question card a turn parks on, as the panel holds it. Free of React, so
// unit-testable.

import type { PromptOrigin, QuestionSpec, UserQuestionMessage } from "../types";

export interface PendingQuestion {
  id: string;
  questions: QuestionSpec[];
  /** Set when the card was raised by something other than the turn on screen — a
   *  sub-agent working alongside it. Rendered as a badge: an unattributed question
   *  is one the user cannot weigh. */
  origin?: PromptOrigin;
  /** Which conversation asked, so the answer reaches the agent parked on it. */
  sessionId?: string;
  /** Epoch ms the agent stops waiting at, when the question carries a wall. */
  expiresAt?: number;
}

/** The card for *msg*, asked in *activeSessionId* unless the message names its own. */
export function pendingQuestion(
  msg: UserQuestionMessage, activeSessionId: string | null, now: number,
): PendingQuestion {
  return {
    id: msg.id,
    questions: msg.questions,
    origin: msg.origin,
    sessionId: msg.session_id ?? activeSessionId ?? undefined,
    expiresAt: msg.timeout_secs ? now + msg.timeout_secs * 1000 : undefined,
  };
}

/**
 * The card still standing once conversation *loadedSessionId* is on screen: the one it
 * asked keeps it (a reconnect, whose worker will not send it again); any other drops
 * it, and the server sends it again when the user goes back.
 */
export function questionAfterLoad(
  card: PendingQuestion | null, loadedSessionId: string,
): PendingQuestion | null {
  return card && card.sessionId !== loadedSessionId ? null : card;
}
