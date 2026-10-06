import React, { useEffect, useRef } from "react";
import type { SessionMeta } from "../types";

interface Props {
  /** The conversations still working — what the user is about to walk away from. */
  running: SessionMeta[];
  /** Keep them going: detach, then disconnect. */
  onDetach: () => void;
  /** Disconnect anyway — the server stops and their turns end with it. */
  onDiscard: () => void;
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
  running, onDetach, onDiscard, onCancel,
}) => {
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

        <p className="disconnect-note">
          Disconnecting ends the server, and their turns with it. Keeping them going
          leaves MIMIR running without this window — reopening the workspace comes back
          to it.
        </p>

        <div className="disconnect-actions">
          <button ref={detachRef} className="disconnect-primary" onClick={onDetach}>
            Keep them going
          </button>
          <button className="disconnect-secondary" onClick={onDiscard}>
            Disconnect anyway
          </button>
        </div>
        <button className="disconnect-close" onClick={onCancel} aria-label="Stay connected">
          ×
        </button>
      </div>
    </div>
  );
};
