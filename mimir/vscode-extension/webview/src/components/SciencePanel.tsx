import React, { useCallback, useEffect, useState } from "react";
import type { PanelSection, WatchedRun } from "../types";
import { PanelGroupList, RunBadge } from "./PanelGroups";
import { PanelMiniMap } from "./PanelMiniMap";
import { PanelResizer, usePanelWidth } from "./PanelResizer";
import { badgeKey, isFilled, toggle } from "./panelGroupUtils";

/** While the drawer is open, ask again this often. */
const PANEL_POLL_MS = 10000;

/** Accent for the sections MIMIR ships; a plugin's own section gets the fallback. */
const SECTION_ACCENT: Record<string, string> = {
  machine: "var(--mimir-cyan)",
  cluster: "var(--mimir-blue)",
  optimisation: "var(--mimir-coral)",
  architecture: "var(--mimir-violet)",
};
const FALLBACK_ACCENT = "var(--mimir-violet)";

/** A section as the drawer draws it; the runs one carries its badges itself. */
type ShownSection = PanelSection & { unplaced?: WatchedRun[] };

interface Props {
  sections: PanelSection[];
  /** Runs the watcher holds that no machine row claimed. */
  runs: WatchedRun[];
  /** Stop a run with the cancel op its server named. */
  onCancelRun: (jobKey: string) => void;
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
  /** Open a view a section offers, by the name the section gave it ("code_map"), on
   *  *focus* when given (a component of the map). */
  onOpenView: (view: string, focus?: string) => void;
}

/**
 * What is specific to scientific computing, in one drawer: the host, the clusters it
 * reaches, the optimisation in progress, each section filled by the server that owns
 * it, asked by capability.
 *
 * Runs in flight appear as badges on the machines they run on. Which rows are open
 * belongs to the reader, so it is held here.
 */
export const SciencePanel: React.FC<Props> = ({
  sections,
  runs,
  onCancelRun,
  onRefresh,
  onReload,
  refreshing,
  agentReady,
  onPrompt,
  onOpenView,
}) => {
  // Which rows are open, keyed on the servers' own ids so the answer survives a
  // refresh. Everything starts closed: a workspace that reaches three clusters would
  // otherwise open onto a page of machines.
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const toggleGroup = useCallback(
    (key: string) => setOpenGroups((open) => toggle(open, key)),
    [],
  );
  // Which sections are folded, keyed on the section's id. Everything starts open: a
  // section is shown only once it has something in it, and that is what to read.
  const [foldedSections, setFoldedSections] = useState<Record<string, boolean>>({});
  const toggleSection = useCallback(
    (key: string) => setFoldedSections((folded) => toggle(folded, key)),
    [],
  );
  const { width, panelRef, handleProps } = usePanelWidth("science");
  // The runs no machine claimed, as a section of their own: a run the user cannot see
  // is one they cannot stop either.
  const shown = [
    ...sections.filter(isFilled),
    ...(runs.length > 0
      ? [{ section: "runs", title: "Runs", lines: [], unplaced: runs }]
      : []),
  ] as ShownSection[];

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
    <div className="sessions-panel science-panel" ref={panelRef} style={{ width }}>
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
        {!agentReady && shown.length === 0 && (
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

        {agentReady && shown.length === 0 && (
          <div className="sessions-empty">Nothing to report for this workspace yet.</div>
        )}

        {shown.map((section) => {
          const accent = SECTION_ACCENT[section.section.toLowerCase()] ?? FALLBACK_ACCENT;
          const isOpen = !foldedSections[section.section];
          const bodyId = `ps-${section.section}`;
          return (
          <div
            className={`science-section${isOpen ? "" : " is-folded"}`}
            key={section.section}
            style={{ ["--section-accent" as any]: accent }}
          >
            <button
              className="science-section-title"
              aria-expanded={isOpen}
              aria-controls={bodyId}
              onClick={() => toggleSection(section.section)}
            >
              <span className="science-section-label">{section.title}</span>
              <span className="science-section-chevron" aria-hidden="true">
                {isOpen ? "▾" : "▸"}
              </span>
            </button>
            {isOpen && (
              <div id={bodyId} role="group">
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
                    onCancelRun={onCancelRun}
                  />
                )}
                {section.unplaced && (
                  <div className="science-badges is-flat">
                    {section.unplaced.map((run) => (
                      <RunBadge
                        key={run.job_key}
                        run={run}
                        open={!!openGroups[badgeKey(run)]}
                        onToggle={() => toggleGroup(badgeKey(run))}
                        onCancel={onCancelRun}
                      />
                    ))}
                  </div>
                )}
                {section.detail && <div className="science-detail">{section.detail}</div>}
                {section.map && section.open && section.map.components.length > 0 && (
                  <PanelMiniMap map={section.map} onOpen={(id) => onOpenView(section.open!, id)} />
                )}
                {section.open === "code_map" && (
                  <button className="science-open" onClick={() => onOpenView("code_map")}>
                    Open map
                  </button>
                )}
              </div>
            )}
          </div>
          );
        })}
      </div>

      <PanelResizer label="scientific computing" handleProps={handleProps} />
    </div>
  );
};
