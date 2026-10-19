// Handing the rendered chat to the server, so a reload can show it again. Tool rows,
// reasoning panels and diff cards are assembled by the reducer and live nowhere else.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage, ClientMessage } from "../types";
import type { ChatState } from "../state/chatReducer";
import type { ServerHandlers } from "../state/serverMessages";
import { makeId } from "../state/ids";
import { planTranscript, type TranscriptSync } from "../components/transcriptUtils";

/** How often the rendered transcript is checkpointed to the server mid-turn. */
const TRANSCRIPT_CHECKPOINT_MS = 5000;

interface Deps {
  send: (msg: ClientMessage) => void;
  busy: boolean;
  chatStateRef: React.MutableRefObject<ChatState>;
  activeSessionIdRef: React.MutableRefObject<string | null>;
  renderedSeqRef: React.MutableRefObject<Record<string, number>>;
}

export function useTranscriptSync({ send, busy, chatStateRef, activeSessionIdRef, renderedSeqRef }: Deps) {
  // The list last sent: skipped when unchanged, so a checkpoint costs nothing on a
  // quiet turn. The reducer never mutates, so identity is the whole test.
  const lastSentMessagesRef = useRef<ChatMessage[] | null>(null);
  // What the server was last sent, so the next send carries only what changed.
  const transcriptSyncRef = useRef<TranscriptSync | null>(null);
  // Bumped when the transcript must reach the server now rather than at the next end
  // of turn: a reconnect that kept the richer copy.
  const [pushes, setPushes] = useState(0);

  const sendTranscript = useCallback((force = false) => {
    const sessionId = activeSessionIdRef.current;
    const messages = chatStateRef.current.messages;
    if (!sessionId || messages.length === 0) return;
    if (!force && messages === lastSentMessagesRef.current) return;
    lastSentMessagesRef.current = messages;
    const { body, next } = planTranscript(
      messages, sessionId, transcriptSyncRef.current, force, makeId);
    transcriptSyncRef.current = next;
    send({
      type: "transcript",
      session_id: sessionId,
      ...body,
      // How far these messages reach. The server replays from it, so a window that has
      // rendered everything is sent nothing on its next attach.
      through_seq: renderedSeqRef.current[sessionId] ?? 0,
    });
  }, [send, chatStateRef, activeSessionIdRef, renderedSeqRef]);

  const requestPush = useCallback(() => setPushes((n) => n + 1), []);

  // At the end of the turn. The delay lets the last few events (a trailing verdict,
  // the batch status) land in the transcript before it is sent.
  const wasBusyRef = useRef(false);
  useEffect(() => {
    const wasBusy = wasBusyRef.current;
    wasBusyRef.current = busy;
    if (busy || !wasBusy) return;
    const timer = setTimeout(sendTranscript, 1000);
    return () => clearTimeout(timer);
  }, [busy, sendTranscript]);

  // And during it, at step boundaries (streaming prose is held in `draft`), so a drop
  // or reload mid-turn does not lose the work on screen.
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(sendTranscript, TRANSCRIPT_CHECKPOINT_MS);
    return () => clearInterval(timer);
  }, [busy, sendTranscript]);

  // A push asked for out of band. Forced: the list may be the very one the last send
  // was handed, and that send is what the closed socket swallowed.
  useEffect(() => {
    if (pushes === 0) return;
    sendTranscript(true);
  }, [pushes, sendTranscript]);

  const on: ServerHandlers = useMemo(() => ({
    // The server has made itself survivable, and wants the screen's copy now.
    transcript_resync: (msg) => {
      if (msg.session_id === activeSessionIdRef.current) sendTranscript(true);
    },
  }), [activeSessionIdRef, sendTranscript]);

  return { requestPush, on };
}
