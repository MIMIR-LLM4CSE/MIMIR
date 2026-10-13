import React, { useCallback, useEffect, useState } from "react";
import type { PanelSection } from "../types";
import { PanelGroupList } from "./PanelGroups";
import { PanelResizer, usePanelWidth } from "./PanelResizer";
import { toggle } from "./panelGroupUtils";

/** While the drawer is open, ask again this often. */
const PANEL_POLL_MS = 20000;

/** Accent for the sections MIMIR ships; a plugin's own section gets the fallback. */
const SECTION_ACCENT: Record<string, string> = {
  machine: "var(--mimir-cyan)",
  cluster: "var(--mimir-blue)",
  optimisation: "var(--mimir-coral)",
};
const FALLBACK_ACCENT = "var(--mimir-violet)";

interface Props {
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
 * None of it is the client's own. The machine, the clusters this workspace reaches, the
 * optimisation in progress — each is filled by the server that owns those facts, asked
 * by capability rather than by name, so a plugin server adds its own section without a
 * line changing here. Nearest first: the host, then the clusters it reaches, then what
 * is being optimised on them.
 *
 * The sub-agent rung is deliberately not here: how far delegation may go is an agent
 * setting, set once alongside thinking depth and enforcement, not a fact about the
 * machine. It lives in the settings popover, and the sub-agents at work have their own
 * drawer under the session that spawned them.
 *
 * A server's section may answer with a tree as well as with lines — the clusters, the
 * kinds of machine on each, what a probe found on one — and the runs in flight are hung
 * on the machines they are running on, as badges. That is the only place they appear: a
 * flat list of them under the sections said where a run was not, which is the one thing
 * a reader of this drawer never asks. Which rows are open is held here, because it
 * belongs to the reader rather than to any one section.
 */
export const SciencePanel: React.FC<Props> = ({
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

  // Opening the drawer serves whatever the servers have cached, and starts this
  // conversation's agent if it has none. It is not a re-read: a drawer that probed
  // every cluster each time it was opened took seconds to say what it already knew.
  useEffect(() => {
    onRefresh();
  }, [onRefresh]);

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
          <div>
            <div className="science-skeleton" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div className="sessions-empty">
              Starting this conversation's agent — the servers that fill this panel are
              coming up.
            </div>
          </div>
        )}

        {sections.map((section) => {
          const accent = SECTION_ACCENT[section.section.toLowerCase()] ?? FALLBACK_ACCENT;
          return (
          <div
            className="science-section"
            key={section.section}
            style={{ ["--section-accent" as any]: accent }}
          >
            <div className="science-section-title">{section.title}</div>
            {section.lines.map((line, i) => (
              <div className="science-line" key={i}>
                <span className="science-line-label" title={line.label}>{line.label}</span>
                <span
                  className={`science-line-value ${line.state ? `is-${line.state}` : ""}`}
                  title={line.value}
                >
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
          );
        })}
      </div>

      <PanelResizer label="scientific computing" handleProps={handleProps} />
    </div>
  );
};
