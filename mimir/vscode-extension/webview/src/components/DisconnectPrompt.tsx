import React, { useEffect, useRef } from "react";
import type { SessionMeta } from "../types";
import { disconnectOutcome } from "./disconnectUtils";

interface Props {
  /** The conversations still working — what the user is about to walk away from. */
  running: SessionMeta[];
  /**
   * Whether this window started the server it is talking to.
   *
   * It decides what disconnecting *does*: a server this window spawned is killed with
   * it, one it merely attached to is left running. Both outcomes were described as the
   * first, which told half the users the opposite of what would happen.
   */
  serverIsOurs: boolean;
  /** Keep them going: detach, then disconnect. */
  onDetach: () => void;
  /** Disconnect, whatever that does to the runs here — see {@link disconnectOutcome}. */
  onDiscard: () => void;
  /** Stop the server outright. Only offered where disconnecting would not. */
  onStop: () => void;
  /** Neither: stay connected. Dismissing is this, which is why it is the safe one. */
  onCancel: () => void;
}

/**
 * Asked when disconnecting would end work that is still running.
 *
 * Three answers, and the mapping is deliberate: dismissing the dialog cancels the
 * disconnect rather than choosing one of the outcomes. A dialog that defaults to an
 * action on being dismissed makes Escape a decision, and the decision here can end a
 * two-hour build.
 *
 * It names the conversations rather than counting them. "2 conversations are running"
 * tells the user to go and look; their titles let them answer from here.
 *
 * Only shown when something is actually running — disconnecting from an idle agent is
 * not a decision and must not be made into one.
 */
export const DisconnectPrompt: React.FC<Props> = ({
  running, serverIsOurs, onDetach, onDiscard, onStop, onCancel,
}) => {
  const outcome = disconnectOutcome(serverIsOurs);
  const detachRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    detachRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCancel();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onCancel]);

  return (
    <div
      className="disconnect-backdrop"
      // The backdrop is a dismissal, so it cancels — same as Escape.
      onClick={onCancel}
      role="presentation"
    >
      <div
        className="disconnect-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="disconnect-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div id="disconnect-title" className="disconnect-title">
          {running.length === 1
            ? "A conversation is still working"
            : `${running.length} conversations are still working`}
        </div>

        <ul className="disconnect-list">
          {running.map((s) => (
            <li key={s.id}>{s.title || "Untitled conversation"}</li>
          ))}
        </ul>

        <p className="disconnect-note">{outcome.note}</p>

        <div className="disconnect-actions">
          <button ref={detachRef} className="disconnect-primary" onClick={onDetach}>
            Keep them going
          </button>
          <button className="disconnect-secondary" onClick={onDiscard}>
            {outcome.discardLabel}
          </button>
          {outcome.offerStop && (
            <button className="disconnect-secondary" onClick={onStop}>
              Stop the server
            </button>
          )}
        </div>
        <button className="disconnect-close" onClick={onCancel} aria-label="Stay connected">
          ×
        </button>
      </div>
    </div>
  );
};
