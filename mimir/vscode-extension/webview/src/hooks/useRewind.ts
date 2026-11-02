// Editing a message already sent: the bubble being edited, what the edit would undo,
// and the rewind that sends the edited text once the user confirms.

import { useCallback, useMemo, useRef, useState } from "react";
import type { ClientMessage, RewindPlanMessage, TodoItem } from "../types";
import type { ChatAction } from "../state/chatReducer";
import type { ServerHandlers } from "../state/serverMessages";

interface Deps {
  send: (msg: ClientMessage) => void;
  dispatch: (action: ChatAction) => void;
  activeSessionIdRef: React.MutableRefObject<string | null>;
  /** Start a turn with *text*, whatever the panel thinks is running: the rewind stopped it. */
  startQuery: (text: string) => void;
  onTodos: (todos: TodoItem[]) => void;
}

/** True when the plan has nothing to undo or show: the rewind can go without asking. */
export function planIsEmpty(plan: RewindPlanMessage): boolean {
  return !plan.turn_running
    && !(plan.files?.length) && !(plan.conflicts?.length)
    && !(plan.unrestorable?.length) && !(plan.runs?.length);
}

export function useRewind({ send, dispatch, activeSessionIdRef, startQuery, onTodos }: Deps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [plan, setPlan] = useState<RewindPlanMessage | null>(null);
  // The edited text, once submitted: the confirm card is up, or the plan is awaited.
  const [pendingText, setPendingText] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const editingRef = useRef<string | null>(null);
  const pendingRef = useRef<string | null>(null);

  const reset = useCallback(() => {
    editingRef.current = null;
    pendingRef.current = null;
    setEditingId(null);
    setPlan(null);
    setPendingText(null);
    setWorking(false);
  }, []);

  const go = useCallback((id: string, text: string) => {
    setWorking(true);
    send({ type: "rewind", session_id: activeSessionIdRef.current, message_id: id, text });
  }, [send, activeSessionIdRef]);

  // The preview is asked for as the editor opens, so it is there by the time the
  // edited text is.
  const start = useCallback((id: string) => {
    editingRef.current = id;
    pendingRef.current = null;
    setEditingId(id);
    setPlan(null);
    setPendingText(null);
    setWorking(false);
    send({ type: "rewind_preview", session_id: activeSessionIdRef.current, message_id: id });
  }, [send, activeSessionIdRef]);

  const submit = useCallback((text: string) => {
    const id = editingRef.current;
    if (!id || !text.trim()) return;
    pendingRef.current = text;
    setPendingText(text);
    if (plan && plan.message_id === id && plan.available && planIsEmpty(plan)) go(id, text);
  }, [plan, go]);

  const confirm = useCallback(() => {
    const id = editingRef.current;
    if (id && pendingRef.current) go(id, pendingRef.current);
  }, [go]);

  // Back to the editor from the confirm card.
  const back = useCallback(() => {
    pendingRef.current = null;
    setPendingText(null);
  }, []);

  const on: ServerHandlers = useMemo(() => ({
    rewind_plan: (msg) => {
      if (msg.message_id !== editingRef.current) return;
      setPlan(msg);
      // A refusal answers the rewind itself: nothing was changed, say so.
      if (!msg.available) setWorking(false);
      const text = pendingRef.current;
      if (text && msg.available && planIsEmpty(msg)) go(msg.message_id, text);
    },
    rewound: (msg) => {
      if (msg.session_id !== activeSessionIdRef.current) return;
      dispatch({ type: "session_loaded_messages", messages: msg.display_messages });
      onTodos(msg.todos ?? []);
      reset();
      if (msg.text.trim()) startQuery(msg.text);
    },
  }), [dispatch, activeSessionIdRef, onTodos, reset, startQuery, go]);

  return { editingId, plan, pendingText, working, start, submit, confirm, back, cancel: reset, on };
}

export type Rewind = ReturnType<typeof useRewind>;
