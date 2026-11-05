import React from "react";
import type { ToolActivity } from "../types";
import { useElapsed, formatDuration } from "../hooks/useElapsed";
import { progressStyle } from "./ToolActivityList";
import { labelForKind } from "./toolIcons";

interface Props {
  /** Runs still in flight whose row has scrolled out of the thread. */
  runs: ToolActivity[];
  /** Bring the run's row back into view. */
  onFocus: (id: string) => void;
}

/** At most this many cards; the rest are counted. Two is what fits without the dock
 *  becoming the thing the reader has to scroll past. */
const MAX_CARDS = 2;

function RunCard({ run, onFocus }: { run: ToolActivity; onFocus: (id: string) => void }) {
  const elapsed = useElapsed(run.startedAt, true);
  // The card names the same run its row does, so it says what the row says: the family
  // and the model's sentence about the call. Left on the derived label, the dock for a
  // running shell call read "Running shell command" while its own row three lines up
  // read "Bash · running the row-display tests". The label stays in the tooltip, where
  // it is the precise name of what was called.
  const name = run.doing
    ? `${labelForKind(run.kind || "tool")} · ${run.doing}`
    : run.label;
  const hasPercent = typeof run.percent === "number";
  const pct = hasPercent ? Math.max(0, Math.min(100, run.percent as number)) : 0;

  return (
    <button
      className="run-dock-card"
      onClick={() => onFocus(run.id)}
      title={`${run.label}${run.phase ? ` — ${run.phase}` : ""} · click to scroll to it`}
      style={hasPercent ? progressStyle(pct) : undefined}
      role={hasPercent ? "progressbar" : undefined}
      aria-valuenow={hasPercent ? pct : undefined}
      aria-valuemin={hasPercent ? 0 : undefined}
      aria-valuemax={hasPercent ? 100 : undefined}
    >
      {!hasPercent && <span className="tb-spinner" aria-hidden="true" />}
      <span className="run-dock-body">
        <span className="run-dock-label">{name}</span>
        {run.phase && (
          <span className="run-dock-phase">
            {run.phase}
            <span className="streaming-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          </span>
        )}
      </span>
      {hasPercent && <span className="tool-percent">{Math.round(pct)}%</span>}
      <span className="run-dock-time">{formatDuration(elapsed)}</span>
    </button>
  );
}

interface ViewProps extends Props {
  /** Folded to one line that counts the runs. */
  folded: boolean;
  onFold: (folded: boolean) => void;
}

/** The dock as it is drawn, open or folded. */
export function RunDockView({ runs, onFocus, folded, onFold }: ViewProps) {
  if (runs.length === 0) return null;
  if (folded) {
    return (
      <div className="run-dock" aria-live="polite">
        <button
          className="run-dock-fold"
          onClick={() => onFold(false)}
          aria-expanded="false"
          title="Show the cards of the runs still going"
        >
          <span className="tb-spinner" aria-hidden="true" />
          {`${runs.length} running · Show`}
        </button>
      </div>
    );
  }
  const shown = runs.slice(0, MAX_CARDS);
  const hidden = runs.length - shown.length;

  return (
    <div className="run-dock" aria-live="polite">
      {shown.map((run) => (
        <RunCard key={run.id} run={run} onFocus={onFocus} />
      ))}
      <span className="run-dock-foot">
        {hidden > 0 && <span className="run-dock-more">{`+${hidden} more running`}</span>}
        <button
          className="run-dock-fold"
          onClick={() => onFold(true)}
          aria-expanded="true"
          title="Fold the cards to one line; the runs go on"
        >
          Hide
        </button>
      </span>
    </div>
  );
}

/**
 * Runs still going whose rows have scrolled off the top of the thread, one click from
 * their place in it. A card leaves when its run ends or its row is visible again; Hide
 * folds them all to one line, which stays folded for the runs that come after until
 * it is opened again.
 */
export function RunProgressDock(props: Props) {
  const [folded, setFolded] = React.useState(false);
  return <RunDockView {...props} folded={folded} onFold={setFolded} />;
}
