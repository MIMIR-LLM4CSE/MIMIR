import React, { useCallback, useEffect, useState } from "react";
import type { PanelSection, WatchedRun } from "../types";
import { PanelGroupList } from "./PanelGroups";
import { PanelResizer, usePanelWidth } from "./PanelResizer";
import { toggle } from "./panelGroupUtils";

/** While the drawer is open, ask again this often. */
const PANEL_POLL_MS = 20000;

interface Props {
  runs: WatchedRun[];
  sections: PanelSection[];
  /** Ask again, serving whatever the servers have cached. The timer's call. */
  onRefresh: () => void;
  /** Read it all again, cache bypassed, building this conversation's agent if it has
   *  none yet. What opening the drawer and pressing ⟳ do. */
  onReload: () => void;
  /** An ask is in flight: the glyph turns for as long as it is. */
  refreshing: boolean;
  /** Whether the servers that fill the sections are up. */
  agentReady: boolean;
  /** Put a request to the model, as the user's own turn. */
  onPrompt: (text: string) => void;
}

/**
 * What is specific to scientific computing, in one drawer.
 *
 * Almost none of it is the client's own. The optimisation in progress, the machine, the
 * clusters this workspace reaches — each is filled by the server that owns those facts,
 * asked by capability rather than by name, so a plugin server adds its own section
 * without a line changing here. What stays client-side is the runs still going outside
 * the current turn, because that is this process's own watcher state.
 *
 * The sub-agent rung is deliberately not here: how far delegation may go is an agent
 * setting, set once alongside thinking depth and enforcement, not a fact about the
 * machine. It lives in the settings popover, and the sub-agents at work have their own
 * drawer under the session that spawned them.
 *
 * A server's section may answer with a tree as well as with lines — the clusters, the
 * kinds of machine on each, what a probe found on one — and the runs in flight are hung
 * on the machines they are running on. Which rows are open is held here, because it
 * belongs to the reader rather than to any one section.
 */
export const SciencePanel: React.FC<Props> = ({
  runs,
  sections,
  onRefresh,
  onReload,
  refreshing,
  agentReady,
  onPrompt,
}) => {
  // Which rows are open, keyed on the servers' own ids so the answer survives a
  // refresh. Everything starts closed: a workspace that reaches three clusters would
  // otherwise open onto a page of machines.
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const toggleGroup = useCallback(
    (key: string) => setOpenGroups((open) => toggle(open, key)),
    [],
  );
  const { width, handleProps } = usePanelWidth("science");

  // Opening the drawer is the one ask that must not be answered from a cache: it is
  // also the ask that may have to start the agent whose servers hold the answers.
  useEffect(() => {
    onReload();
  }, [onReload]);

  // And while it stays open. A queue moves, a job ends, a partition fills — none of
  // which happens inside a turn, so the panel cannot be left to the turn's own
  // refreshes. Cheap by construction: every section is served from the servers' caches
  // unless the reader asks for a re-read.
  useEffect(() => {
    const timer = setInterval(onRefresh, PANEL_POLL_MS);
    return () => clearInterval(timer);
  }, [onRefresh]);

  return (
    <div className="sessions-panel science-panel" style={{ width }}>
      <div className="sessions-panel-header">
        <span className="sessions-panel-title">Scientific computing</span>
        <button
          className={`sessions-toggle-btn ${refreshing ? "is-busy" : ""}`}
          title="Read it all again, past the servers' caches"
          aria-label="Refresh panel"
          aria-busy={refreshing}
          onClick={() => onReload()}
        >
          ⟳
        </button>
      </div>

      <div className="sessions-list">
        {!agentReady && sections.length === 0 && (
          <div className="sessions-empty">
            Starting this conversation's agent — the servers that fill this panel are
            coming up.
          </div>
        )}

        {sections.map((section) => (
          <div className="science-section" key={section.section}>
            <div className="science-section-title">{section.title}</div>
            {section.lines.map((line, i) => (
              <div className="science-line" key={i}>
                <span className="science-line-label">{line.label}</span>
                <span className={`science-line-value ${line.state ? `is-${line.state}` : ""}`}>
                  {line.value}
                </span>
              </div>
            ))}
            {section.groups && section.groups.length > 0 && (
              <PanelGroupList
                groups={section.groups}
                open={openGroups}
                onToggle={toggleGroup}
                onPrompt={onPrompt}
              />
            )}
            {section.detail && <div className="science-detail">{section.detail}</div>}
          </div>
        ))}

        <div className="science-section">
          <div className="science-section-title">Runs outside this turn</div>
          {/* The ones no machine claimed. A run on a scheduler this panel knows is a
              badge on that machine's row instead, where it means more. */}
          {runs.length === 0 && <div className="sessions-empty">Nothing running.</div>}
          {runs.map((run) => (
            <div className="science-line" key={run.job_key}>
              <span className="science-line-label">{run.kind || run.server || "run"}</span>
              <span className="science-line-value">{run.job_key}</span>
            </div>
          ))}
        </div>
      </div>

      <PanelResizer label="scientific computing" handleProps={handleProps} />
    </div>
  );
};
