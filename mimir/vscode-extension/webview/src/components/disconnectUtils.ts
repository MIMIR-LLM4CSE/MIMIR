/**
 * What disconnecting actually does, which is not the same answer in both cases.
 *
 * A workspace has one server, and a window either started it or attached to one that
 * was already serving. Only the first is this window's to stop: the host kills the
 * process it spawned and deliberately leaves alone one it merely connected to, since
 * a second window of the same workspace may be reading it and a detached one was kept
 * going on purpose. So "disconnect" ends the run in one case and leaves it running in
 * the other — and a dialog that promises the first in both tells half its users the
 * opposite of what will happen, at the one moment they are deciding.
 *
 * Pure, and separate from the dialog, so what it claims can be tested without a DOM.
 */

/**
 * The conversations that would be abandoned by disconnecting now.
 *
 * "Working" is not "a turn is in flight". A conversation that launched a two-hour
 * build and answered has no turn at all, and is precisely the one worth asking about
 * before walking away — it is the case detaching exists for. Asking only about live
 * turns meant the question was never put for it: the job was left behind silently,
 * which from the outside looks exactly like MIMIR having decided to detach on its own.
 *
 * *busy* is the client's own flag for the conversation on screen, and it is
 * authoritative for that one: the sessions list is pushed when a turn *ends*, so it
 * lags the turn happening right here.
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
 * What disconnecting does, now that it does one thing.
 *
 * A server exists claimed — detached, asked to be left running — or owned by the window
 * that started it. There is no third state: one nobody claimed stops when its last
 * client has been gone for the grace, whoever started it. So disconnecting ends it and
 * the turns with it, and this dialog is only ever raised in that case: a claimed server
 * survives a disconnect by definition, which is nothing to ask about.
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
