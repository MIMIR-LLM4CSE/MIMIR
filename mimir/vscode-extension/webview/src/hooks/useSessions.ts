// The conversations: the list, the one on screen, the cards other ones raised, and how
// far into each one's journal this window has rendered.

import { useCallback, useMemo, useRef, useState } from "react";
import type {
  ApprovalMessage, ServerMessage, SessionMeta, UserQuestionMessage,
} from "../types";
import type { ChatAction, ChatState } from "../state/chatReducer";
import type { ServerHandlers, ServerMessageOf } from "../state/serverMessages";
import { chooseRestoredMessages } from "../components/transcriptUtils";

interface Deps {
  dispatch: (action: ChatAction) => void;
  chatStateRef: React.MutableRefObject<ChatState>;
  scrollToBottom: () => void;
  /** Run one event of a replay through the panel's own message handler. */
  redeliver: (msg: ServerMessage) => void;
  /**
   * The rest of the panel's share of a loaded conversation. *switched* is false for a
   * reconnect to the one already on screen; *pushTranscript* when the screen kept a
   * richer copy than the server's.
   */
  onLoaded: (msg: ServerMessageOf<"session_loaded">, switched: boolean, pushTranscript: boolean) => void;
}

type ForeignPrompt = ApprovalMessage | UserQuestionMessage;

export function useSessions({ dispatch, chatStateRef, scrollToBottom, redeliver, onLoaded }: Deps) {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  // Read by message handlers, which must not see a stale session.
  const activeSessionIdRef = useRef<string | null>(null);
  // True between a switch request and its session_loaded reply.
  const [sessionLoading, setSessionLoading] = useState(false);
  const [showSessionsPanel, setShowSessionsPanel] = useState(false);
  // Cards raised by conversations the user is not reading, newest per conversation.
  // They do not belong in this thread — that is another transcript — but they cannot
  // be hidden either: each is a turn parked on a person with no timeout. Shown in a
  // strip above the thread; answering one sends its own session_id.
  const [foreignPrompts, setForeignPrompts] = useState<Record<string, ForeignPrompt>>({});
  // How far into each session's journal this window has rendered. Per session: one
  // number would carry a session's high-water mark onto another, and tell the server
  // this window had seen events it never got.
  const renderedSeqRef = useRef<Record<string, number>>({});

  /** Whether a card belongs to a conversation other than the one on screen. A card
   *  with no session_id comes from a server that does not attribute them, and counts as
   *  this conversation's — which is what such a server means by it. */
  const isForeign = useCallback(
    (msg: { session_id?: string }) =>
      !!msg.session_id && msg.session_id !== activeSessionIdRef.current,
    [],
  );
  // One per conversation: the newest card is the one its turn is parked on.
  const rememberForeign = useCallback((msg: ForeignPrompt) => {
    setForeignPrompts((prev) => ({ ...prev, [msg.session_id as string]: msg }));
  }, []);
  const forgetForeign = useCallback((sessionId: string) => {
    setForeignPrompts((prev) => {
      if (!(sessionId in prev)) return prev;
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
  }, []);

  /** Remember the journal position of *msg*, so a later attach replays only the rest. */
  const noteSeq = useCallback((msg: ServerMessage) => {
    const seq = (msg as { seq?: unknown }).seq;
    const owner = (msg as { session_id?: unknown }).session_id;
    if (typeof seq === "number" && typeof owner === "string") {
      const seen = renderedSeqRef.current[owner] ?? 0;
      if (seq > seen) renderedSeqRef.current[owner] = seq;
    }
  }, []);

  const reset = useCallback(() => {
    setActiveSessionId(null);
    setSessions([]);
  }, []);

  const on: ServerHandlers = useMemo(() => ({
    sessions_list: (msg) => setSessions(msg.sessions),

    // What this conversation produced while nobody was attached. Each event goes
    // through the panel's own handler, which rebuilds the rich transcript — tool rows,
    // diff cards, reasoning panels — for a stretch this window was not there for.
    // Streamed deltas are not in it: a resumed turn arrives as the aggregates that
    // closed its blocks.
    replay: (msg) => {
      for (const ev of msg.events ?? []) redeliver(ev);
      const seen = renderedSeqRef.current[msg.session_id] ?? 0;
      if (msg.through_seq > seen) renderedSeqRef.current[msg.session_id] = msg.through_seq;
      if (!msg.more) scrollToBottom();
    },

    // A recorded prose block, which only a replay carries. Fed as a token, so it is
    // placed among the tool cards that followed it the way the live stream was.
    assistant_text: (msg) => dispatch({ type: "token", text: msg.text }),

    session_loaded: (msg) => {
      const prevSessionId = activeSessionIdRef.current;
      activeSessionIdRef.current = msg.session_id;
      setActiveSessionId(msg.session_id);
      // This conversation's card belongs in the thread now, not the strip: if its turn
      // is still parked the server resends the card on arrival.
      forgetForeign(msg.session_id);
      // Reconnecting to the session on screen keeps the live transcript when the
      // stored copy is behind it (chooseRestoredMessages).
      const restore = chooseRestoredMessages(
        msg.display_messages, chatStateRef.current.messages, prevSessionId === msg.session_id,
      );
      dispatch({ type: "session_loaded_messages", messages: restore.messages });
      // After the restore, which resets the turn state this window arrives with.
      if (msg.turn_running) dispatch({ type: "turn_resumed" });
      onLoaded(msg, prevSessionId !== msg.session_id, restore.push);
      setSessionLoading(false);
      scrollToBottom();
    },
  }), [dispatch, chatStateRef, scrollToBottom, redeliver, onLoaded, forgetForeign]);

  return {
    sessions, activeSessionId, activeSessionIdRef, renderedSeqRef,
    sessionLoading, setSessionLoading, showSessionsPanel, setShowSessionsPanel,
    foreignPrompts, isForeign, rememberForeign, forgetForeign,
    noteSeq, reset, on,
  };
}
