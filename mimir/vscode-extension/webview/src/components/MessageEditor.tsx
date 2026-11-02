import React, { useEffect, useRef, useState } from "react";
import type { Rewind } from "../hooks/useRewind";

interface Props {
  initialText: string;
  rewind: Rewind;
}

/**
 * A sent message reopened for editing, in its own bubble. Sending it shows what going
 * back to it undoes — files put back or removed, runs stopped, the turn in flight — and
 * waits for a confirmation; with nothing to undo it goes at once.
 */
export const MessageEditor: React.FC<Props> = ({ initialText, rewind }) => {
  const [text, setText] = useState(initialText);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const { plan, pendingText, working } = rewind;

  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    area.focus();
    area.setSelectionRange(area.value.length, area.value.length);
  }, []);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      rewind.cancel();
    } else if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      rewind.submit(text);
    }
  };

  if (pendingText !== null) {
    if (working || !plan) {
      return <div className="rewind-card rewind-card--busy">{working ? "Going back…" : "Checking what this undoes…"}</div>;
    }
    if (!plan.available) {
      return (
        <div className="rewind-card" role="alert">
          <div className="rewind-card__title">
            {plan.reason ? "Could not go back" : "This message cannot be edited"}
          </div>
          <div className="rewind-card__note">
            {plan.reason || "No checkpoint was kept for it, so going back to it is not possible."}
          </div>
          <div className="rewind-card__actions">
            <button className="rewind-btn" onClick={rewind.cancel}>Close</button>
          </div>
        </div>
      );
    }
    return <RewindConfirm plan={plan} onConfirm={rewind.confirm} onBack={rewind.back} />;
  }

  return (
    <div className="message-editor">
      <textarea
        ref={areaRef}
        className="message-editor__input"
        value={text}
        rows={Math.min(12, Math.max(2, text.split("\n").length))}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        aria-label="Edit message"
      />
      <div className="rewind-card__actions">
        <button className="rewind-btn" onClick={rewind.cancel}>Cancel</button>
        <button className="rewind-btn rewind-btn--primary" disabled={!text.trim()}
                onClick={() => rewind.submit(text)}>
          Send
        </button>
      </div>
    </div>
  );
};

interface ConfirmProps {
  plan: NonNullable<Rewind["plan"]>;
  onConfirm: () => void;
  onBack: () => void;
}

/** What going back undoes, before anything is touched. */
export const RewindConfirm: React.FC<ConfirmProps> = ({ plan, onConfirm, onBack }) => {
  const files = plan.files ?? [];
  const runs = plan.runs ?? [];
  const conflicts = plan.conflicts ?? [];
  const unrestorable = plan.unrestorable ?? [];
  return (
    <div className="rewind-card">
      <div className="rewind-card__title">Go back to this message?</div>
      <div className="rewind-card__note">Everything after it is removed from the conversation.</div>
      {plan.turn_running && <div className="rewind-card__row">The turn in progress is stopped.</div>}
      {files.length > 0 && (
        <Section title={`Files put back (${files.length})`}
                 items={files.map((f) => `${f.action === "delete" ? "remove" : "restore"} · ${f.path}`)} />
      )}
      {runs.length > 0 && (
        <Section title={`Runs stopped (${runs.length})`}
                 items={runs.map((r) => r.cancellable ? r.job_key : `${r.job_key} (cannot be stopped from here)`)} />
      )}
      {conflicts.length > 0 && (
        <Section warn title="Left as they are — changed since by someone else" items={conflicts} />
      )}
      {unrestorable.length > 0 && (
        <Section warn title="Not put back — no copy was kept" items={unrestorable} />
      )}
      <div className="rewind-card__actions">
        <button className="rewind-btn" onClick={onBack}>Back</button>
        <button className="rewind-btn rewind-btn--primary" onClick={onConfirm}>Go back and send</button>
      </div>
    </div>
  );
};

const Section: React.FC<{ title: string; items: string[]; warn?: boolean }> = ({ title, items, warn }) => (
  <div className={`rewind-card__section${warn ? " rewind-card__section--warn" : ""}`}>
    <div className="rewind-card__section-title">{title}</div>
    <ul>
      {items.map((item) => <li key={item}>{item}</li>)}
    </ul>
  </div>
);
