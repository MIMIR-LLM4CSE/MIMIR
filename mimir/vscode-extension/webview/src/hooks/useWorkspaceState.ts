// What the session's work has left in view: the checklist, the edits waiting in the
// batch review, the servers and skills switched on or off, and the context window's
// usage. Each is the server's to report; the panel shows the last report.

import { useCallback, useMemo, useState } from "react";
import type { ClientMessage, DiffEntry, TodoItem, ToggleItem } from "../types";
import type { ContextUsage } from "../components/ContextBar";
import { withDiff } from "../components/batchUtils";
import type { ServerHandlers } from "../state/serverMessages";

export function useWorkspaceState(send: (msg: ClientMessage) => void) {
  const [todos, setTodos] = useState<TodoItem[]>([]);
  // Accumulated across queries; kept until the user accepts or reverts.
  const [batchFiles, setBatchFiles] = useState<DiffEntry[]>([]);
  const [serverToggles, setServerToggles] = useState<ToggleItem[]>([]);
  const [skillToggles, setSkillToggles] = useState<ToggleItem[]>([]);
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);

  const clearTodos = useCallback(() => {
    send({ type: "clear_todos" });
    setTodos([]);
  }, [send]);

  const batch = useMemo(() => ({
    acceptAll: () => {
      setBatchFiles([]);
      send({ type: "batch_review_accept" });
    },
    revertAll: () => {
      setBatchFiles([]);
      send({ type: "batch_review_revert" });
    },
    acceptFile: (file: string) => {
      setBatchFiles((prev) => prev.filter((f) => f.file !== file));
      send({ type: "batch_review_accept_file", file });
    },
    revertFile: (file: string) => {
      setBatchFiles((prev) => prev.filter((f) => f.file !== file));
      send({ type: "batch_review_revert_file", file });
    },
  }), [send]);

  /**
   * A conversation was loaded: its checklist replaces the old one and the usage is
   * unknown until reported. The batch belongs to the conversation left, so a switch
   * drops it; a reconnect to the same one keeps it.
   */
  const onSessionLoaded = useCallback((items: TodoItem[] | undefined, switched: boolean) => {
    setTodos(items ?? []);
    if (switched) setBatchFiles([]);
    setContextUsage(null);
  }, []);

  const reset = useCallback(() => setTodos([]), []);

  const on: ServerHandlers = useMemo(() => ({
    todo: (msg) => setTodos(msg.items),
    batch_status: (msg) => setBatchFiles(msg.files ?? []),
    // The reducer draws the edit card; the batch keeps the file.
    diff: (msg) => setBatchFiles((prev) => withDiff(prev, msg.file, msg.patch)),
    toggles_list: (msg) => {
      setServerToggles(msg.servers ?? []);
      setSkillToggles(msg.skills ?? []);
    },
    context_usage: (msg) => setContextUsage({
      used_tokens: msg.used_tokens,
      total_tokens: msg.total_tokens,
      reserved_tokens: msg.reserved_tokens,
      overhead_tokens: msg.overhead_tokens,
      overhead_measured: msg.overhead_measured,
      // Carried through, or the bar reads a floor as a verdict: a resumed session
      // comes back with no agent, so `used_tokens` is missing the system prompt and
      // tools and the figure must not be judged against the limit.
      provisional: msg.provisional,
      history_messages: msg.history_messages,
      history_messages_full: msg.history_messages_full,
    }),
  }), []);

  return {
    todos, clearTodos, batchFiles, batch, serverToggles, skillToggles, contextUsage,
    onSessionLoaded, reset, on,
  };
}
