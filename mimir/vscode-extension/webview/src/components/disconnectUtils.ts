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

/** Which actions a disconnect dialog offers, and what it may truthfully say. */
export interface DisconnectOutcome {
  /** What disconnecting does to the runs, in the dialog's own voice. */
  note: string;
  /** Label for the plain disconnect. */
  discardLabel: string;
  /**
   * Whether to offer stopping the server outright.
   *
   * Only when it is not this window's to kill: there, disconnecting leaves it running,
   * so stopping it has to be asked for separately. When it *is* this window's,
   * disconnecting already stops it and a second button would mean the same thing.
   */
  offerStop: boolean;
}

export function disconnectOutcome(serverIsOurs: boolean): DisconnectOutcome {
  if (serverIsOurs) {
    return {
      note:
        "Disconnecting ends the server, and their turns with it. Keeping them going " +
        "leaves MIMIR running without this window — reopening the workspace comes " +
        "back to it.",
      discardLabel: "Disconnect anyway",
      offerStop: false,
    };
  }
  return {
    note:
      "This server was not started by this window, so disconnecting leaves it " +
      "running and their turns carry on — reopening the workspace comes back to " +
      "them. Stopping it ends every turn, including any in another window of this " +
      "workspace.",
    discardLabel: "Disconnect, leave them running",
    offerStop: true,
  };
}
