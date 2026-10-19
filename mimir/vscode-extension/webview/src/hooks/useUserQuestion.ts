// The question the turn on screen is parked on: a clarification batch or a plan up for
// approval. Answering it, or its wall running out, sets the turn producing again.

import { useCallback, useMemo, useState } from "react";
import type { ApprovalMessage, ClientMessage, QuestionAnswer, UserQuestionMessage } from "../types";
import type { ChatAction } from "../state/chatReducer";
import type { ServerHandlers } from "../state/serverMessages";
import {
  pendingQuestion, questionAfterLoad, type PendingQuestion,
} from "../components/questionUtils";

interface Deps {
  send: (msg: ClientMessage) => void;
  dispatch: (action: ChatAction) => void;
  activeSessionIdRef: React.MutableRefObject<string | null>;
  isForeign: (msg: { session_id?: string }) => boolean;
  rememberForeign: (msg: ApprovalMessage | UserQuestionMessage) => void;
  forgetForeign: (sessionId: string) => void;
  scrollToBottom: () => void;
}

export function useUserQuestion({
  send, dispatch, activeSessionIdRef, isForeign, rememberForeign, forgetForeign, scrollToBottom,
}: Deps) {
  const [userQuestion, setUserQuestion] = useState<PendingQuestion | null>(null);

  const clear = useCallback(() => setUserQuestion(null), []);

  const answer = useCallback((answers: QuestionAnswer[]) => {
    if (!userQuestion) return;
    send({
      type: "user_question_response",
      id: userQuestion.id,
      session_id: userQuestion.sessionId,
      answers,
    });
    // Whatever was chosen, the agent goes back to work: even a rejected plan is the
    // loop resuming to write the answer that says so.
    dispatch({ type: "prompt_answered", resumes: true });
    setUserQuestion(null);
  }, [userQuestion, send, dispatch]);

  const on: ServerHandlers = useMemo(() => ({
    // Parks the agent on a person: the card is held here, and the reducer is told the
    // turn stopped producing (see parkOnPrompt) — which is also what hands the
    // transcript, plan included, back to the server.
    user_question: (msg) => {
      if (isForeign(msg)) {
        // Another conversation's: it belongs in the strip, not this thread, and must
        // not clear `busy` (parkOnPrompt does) while this conversation still works.
        rememberForeign(msg);
        return;
      }
      setUserQuestion(pendingQuestion(msg, activeSessionIdRef.current, Date.now()));
      dispatch(msg);
      scrollToBottom();
    },
    // The wait behind a card ended with nothing answered. Nothing is sent back — the
    // agent has gone on with the option it recommended — so this takes the card down,
    // wherever it shows, and marks the turn as producing again.
    prompt_expired: (msg) => {
      if (isForeign(msg)) {
        forgetForeign(msg.session_id as string);
        return;
      }
      setUserQuestion((prev) => (prev && prev.id !== msg.id ? prev : null));
      dispatch({ type: "prompt_answered", resumes: true });
    },
    session_loaded: (msg) => setUserQuestion((prev) => questionAfterLoad(prev, msg.session_id)),
  }), [dispatch, activeSessionIdRef, isForeign, rememberForeign, forgetForeign, scrollToBottom]);

  return { userQuestion, answer, clear, on };
}
