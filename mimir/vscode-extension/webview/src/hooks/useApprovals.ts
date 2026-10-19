// Answering the approval cards a turn parks on: one answer for every call a card
// merged, sent to the conversation that asked.

import { useCallback } from "react";
import type { ApprovalMode, ChatMessage, ClientMessage } from "../types";
import type { ChatAction, ChatState } from "../state/chatReducer";

type Choice = "y" | "n" | "a";

interface Deps {
  send: (msg: ClientMessage) => void;
  dispatch: (action: ChatAction) => void;
  chatStateRef: React.MutableRefObject<ChatState>;
  activeSessionIdRef: React.MutableRefObject<string | null>;
  changeApprovalMode: (mode: ApprovalMode) => void;
}

/** The newest approval card in *messages*, if one is standing. */
function newestCard(messages: ChatMessage[]): ChatMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].kind === "approval" && messages[i].approval !== undefined) return messages[i];
  }
  return undefined;
}

export function useApprovals({
  send, dispatch, chatStateRef, activeSessionIdRef, changeApprovalMode,
}: Deps) {
  const respond = useCallback((id: string, choice: Choice, approvedFiles?: string[]) => {
    // Every merged id, so all the blocked calls behind the card unblock.
    const card = chatStateRef.current.messages.find(
      (m) => m.kind === "approval" && m.approval?.id === id
    );
    const allIds = card?.approval?.ids ?? [id];
    // The conversation that asked, read off the card. The server drops an answer
    // carrying none rather than guessing; the session on screen is exactly the wrong
    // guess for a card raised elsewhere.
    const sessionId = card?.approval?.session_id ?? activeSessionIdRef.current ?? undefined;
    for (const aid of allIds) {
      const payload: {
        type: "approval_response"; id: string; choice: Choice;
        session_id?: string; approved_files?: string[];
      } = { type: "approval_response", id: aid, choice, session_id: sessionId };
      if (approvedFiles !== undefined) payload.approved_files = approvedFiles;
      send(payload);
    }
    // The card(s) go, and the turn is producing again.
    dispatch({ type: "approval_response", id, choice });
  }, [send, dispatch, chatStateRef, activeSessionIdRef]);

  /**
   * Switch the approval mode, answering the card already standing when the new mode
   * covers it. The switch has to work mid-run, which is when a card is on screen with
   * the agent parked on it; leaving it standing would make "auto" mean "auto, starting
   * after you clear this one".
   */
  const changeMode = useCallback((mode: ApprovalMode) => {
    changeApprovalMode(mode);
    const card = newestCard(chatStateRef.current.messages);
    if (!card?.approval) return;
    const leavesWorkspace =
      (card.approval.oow_paths?.length ?? 0) > 0 || Boolean(card.approval.oow_path);
    const covered = mode === "auto_all" || (mode === "auto" && !leavesWorkspace);
    if (covered) respond(card.approval.id, "y");
  }, [changeApprovalMode, chatStateRef, respond]);

  return { respond, changeMode };
}
