// Leaving the server: asked first when it would end work still running, with the option
// of leaving that work going (detaching) instead.

import { useCallback, useMemo, useState } from "react";
import type { ApprovalMode, ClientMessage, SessionMeta } from "../types";
import type { ServerHandlers } from "../state/serverMessages";
import { workingSessions } from "../components/disconnectUtils";

interface Deps {
  send: (msg: ClientMessage) => void;
  sessions: SessionMeta[];
  activeSessionId: string | null;
  busy: boolean;
  approvalMode: ApprovalMode;
  /** Close the socket and clear what this window showed of the server. */
  leave: () => void;
}

export function useDisconnect({ send, sessions, activeSessionId, busy, approvalMode, leave }: Deps) {
  // Set while the server says it has detached: closing the window leaves it working.
  const [detached, setDetached] = useState(false);
  // True while the disconnect waits on an answer about the running work. Not a "show
  // the dialog" flag but the disconnect itself, held: cancelling means it never happened.
  const [disconnectPending, setDisconnectPending] = useState(false);

  // The conversations still working. Two sources, because neither is enough alone:
  // `running` in the session list is the server's answer for the conversations this
  // window is not reading, but it is pushed when a turn ends, not when one starts; for
  // the turn on screen, the client's own `busy` is authoritative and immediate.
  const runningSessions = useMemo(
    () => workingSessions(sessions, activeSessionId, busy),
    [sessions, activeSessionId, busy],
  );

  /**
   * Disconnect, asking first when it would end work that is still running. Nothing is
   * torn down when there is something to ask: dismissing the question cancels the
   * disconnect, so Escape is never a decision that ends a two-hour build.
   */
  const requestDisconnect = useCallback(() => {
    if (runningSessions.length > 0 && !detached) {
      setDisconnectPending(true);
      return;
    }
    leave();
  }, [runningSessions.length, detached, leave]);

  const detachThenDisconnect = useCallback(() => {
    // Only the conversations that are working: detaching everything would give an
    // idle conversation an autonomy level nobody chose for it.
    send({
      type: "detach",
      autonomy: approvalMode,
      session_ids: runningSessions.map((s) => s.id),
    });
    setDisconnectPending(false);
    setDetached(true);
    // After the message, so it leaves before the socket does.
    setTimeout(leave, 150);
  }, [send, approvalMode, runningSessions, leave]);

  const disconnectAnyway = useCallback(() => {
    setDisconnectPending(false);
    leave();
  }, [leave]);

  const cancelDisconnect = useCallback(() => setDisconnectPending(false), []);

  // The autonomy a detached run works under is the approval mode already set: the
  // button carries no picker of its own, so the two cannot disagree.
  const detach = useCallback(
    () => send({ type: "detach", autonomy: approvalMode }), [send, approvalMode]);
  const reattach = useCallback(
    () => send({ type: "detach", autonomy: approvalMode, enabled: false }), [send, approvalMode]);

  const on: ServerHandlers = useMemo(() => ({
    // Absent means detached: an older server sends no flag, and the message existed
    // only to announce one.
    detached: (msg) => setDetached(msg.detached !== false),
  }), []);

  return {
    detached, disconnectPending, runningSessions,
    requestDisconnect, detachThenDisconnect, disconnectAnyway, cancelDisconnect,
    detach, reattach, on,
  };
}
