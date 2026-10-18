/**
 * What disconnecting does, kept apart from the dialog so it is testable without a DOM.
 */

/**
 * The conversations that disconnecting now would abandon: a running turn, or a
 * background run the conversation is waiting on (the case detaching exists for).
 *
 * *busy* is authoritative for the conversation on screen: the sessions list lags it.
 */
export function workingSessions<T extends {
  id: string; running?: boolean; runs?: number;
}>(sessions: T[], activeSessionId: string | null, busy: boolean): T[] {
  const working = (s: T | undefined) =>
    !!s && (!!s.running || (s.runs ?? 0) > 0);
  const others = sessions.filter((s) => working(s) && s.id !== activeSessionId);
  if (!activeSessionId) return others;
  const mine = sessions.find((s) => s.id === activeSessionId);
  if (!busy && !working(mine)) return others;
  return mine ? [mine, ...others] : others;
}

/** Which actions a disconnect dialog offers, and what it may truthfully say. */
export interface DisconnectOutcome {
  /** What disconnecting does to the runs, in the dialog's own voice. */
  note: string;
  /** Label for the plain disconnect. */
  discardLabel: string;
}

/**
 * What disconnecting does: a server nobody claimed stops once its last client is gone,
 * turns included. A claimed one survives, and raises no dialog.
 */
export function disconnectOutcome(): DisconnectOutcome {
  return {
    note:
      "Disconnecting ends the server, and their turns with it. Keeping them going " +
      "leaves MIMIR running without this window — reopening the workspace comes " +
      "back to it.",
    discardLabel: "Disconnect anyway",
  };
}
