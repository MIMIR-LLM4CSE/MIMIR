import React, { useEffect } from "react";
import type { PanelSection, SubAgent, SubAgentLevel, WatchedRun } from "../types";
import { SubAgentsPanel } from "./SubAgentsPanel";

export const SUBAGENT_LEVEL_OPTIONS: {
  value: SubAgentLevel;
  icon: string;
  label: string;
  desc: string;
}[] = [
  {
    value: "explore",
    icon: "🔍",
    label: "explore",
    desc: "Sub-agents read, search and navigate. They change nothing",
  },
  {
    value: "parallel",
    icon: "🌿",
    label: "parallel",
    desc: "They may also edit and run — each in its own copy of the repository, on its own branch",
  },
];

interface Props {
  level: SubAgentLevel;
  onLevelChange: (level: SubAgentLevel) => void;
  subAgents: SubAgent[];
  openedSubAgent: SubAgent | null;
  runs: WatchedRun[];
  sections: PanelSection[];
  onRefresh: () => void;
  onOpenSubAgent: (id: string) => void;
  onCloseSubAgent: () => void;
}

/**
 * What is specific to scientific computing, in one drawer.
 *
 * Four sections, and only two of them are the client's own: how far sub-agents may go
 * with the ones this session started, and the runs still going outside the current
 * turn. The other two — the optimisation in progress, the machine — are filled by the
 * servers that own those facts, asked by capability rather than by name, so a plugin
 * server adds its own section without a line changing here.
 */
export const SciencePanel: React.FC<Props> = ({
  level,
  onLevelChange,
  subAgents,
  openedSubAgent,
  runs,
  sections,
  onRefresh,
  onOpenSubAgent,
  onCloseSubAgent,
}) => {
  useEffect(() => {
    onRefresh();
  }, [onRefresh]);

  return (
    <div className="sessions-panel science-panel">
      <div className="sessions-panel-header">
        <span className="sessions-panel-title">Scientific computing</span>
        <button
          className="sessions-toggle-btn"
          title="Refresh"
          aria-label="Refresh panel"
          onClick={() => onRefresh()}
        >
          ⟳
        </button>
      </div>

      <div className="sessions-list">
        <div className="science-section">
          <div className="science-section-title">Sub-agents</div>
          <div className="science-levels">
            {SUBAGENT_LEVEL_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                className={`science-level ${level === opt.value ? "active" : ""}`}
                title={opt.desc}
                aria-pressed={level === opt.value}
                onClick={() => onLevelChange(opt.value)}
              >
                {opt.icon} {opt.label}
              </button>
            ))}
          </div>
          <div className="science-level-desc">
            {SUBAGENT_LEVEL_OPTIONS.find((o) => o.value === level)?.desc}
          </div>
          <SubAgentsPanel
            subAgents={subAgents}
            opened={openedSubAgent}
            onRefresh={onRefresh}
            onOpen={onOpenSubAgent}
            onClose={onCloseSubAgent}
            embedded
          />
        </div>

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
            {section.detail && <div className="science-detail">{section.detail}</div>}
          </div>
        ))}

        <div className="science-section">
          <div className="science-section-title">Runs outside this turn</div>
          {runs.length === 0 && <div className="sessions-empty">Nothing running.</div>}
          {runs.map((run) => (
            <div className="science-line" key={run.job_key}>
              <span className="science-line-label">{run.kind || run.server || "run"}</span>
              <span className="science-line-value">{run.job_key}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
