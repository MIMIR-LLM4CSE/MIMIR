import React from "react";
import type { CSSProperties } from "react";
import type { ToolActivity } from "../types";
import { useElapsed, formatDuration } from "../hooks/useElapsed";

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
  const hasPercent = typeof run.percent === "number";
  const pct = hasPercent ? Math.max(0, Math.min(100, run.percent as number)) : 0;

  return (
    <button
      className="run-dock-card"
      onClick={() => onFocus(run.id)}
      title={`${run.label}${run.phase ? ` — ${run.phase}` : ""} · click to scroll to it`}
      style={hasPercent ? ({ "--tool-progress": `${pct}%` } as CSSProperties) : undefined}
      role={hasPercent ? "progressbar" : undefined}
      aria-valuenow={hasPercent ? pct : undefined}
      aria-valuemin={hasPercent ? 0 : undefined}
      aria-valuemax={hasPercent ? 100 : undefined}
    >
      <span className="tb-spinner" aria-hidden="true" />
      <span className="run-dock-body">
        <span className="run-dock-label">{run.label}</span>
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
      <span className="run-dock-time">{formatDuration(elapsed)}</span>
    </button>
  );
}

/**
 * Runs still going, whose rows have scrolled off the top of the thread.
 *
 * A build takes twenty minutes and the agent does not wait for it: the thread grows,
 * the row that carries the progress leaves the screen, and checking on it means
 * scrolling back and finding it again. This keeps it where it can be read, out of the
 * way, and one click from its place in the conversation.
 *
 * It has no close button on purpose. There is nothing to dismiss: a card is here
 * because a run is going and its row is not visible, and it leaves the moment either
 * stops being true.
 */
export function RunProgressDock({ runs, onFocus }: Props) {
  if (runs.length === 0) return null;
  const shown = runs.slice(0, MAX_CARDS);
  const hidden = runs.length - shown.length;

  return (
    <div className="run-dock" aria-live="polite">
      {shown.map((run) => (
        <RunCard key={run.id} run={run} onFocus={onFocus} />
      ))}
      {hidden > 0 && (
        <span className="run-dock-more">
          +{hidden} more running
        </span>
      )}
    </div>
  );
}
