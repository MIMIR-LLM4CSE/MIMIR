import React, { useEffect, useState } from "react";
import type { SubAgent } from "../types";
import {
  activityRows,
  anyRunning,
  grantSummary,
  parseTodo,
  stateLabel,
  taskTitle,
  workspaceLine,
} from "./subAgentPanelUtils";

interface Props {
  subAgents: SubAgent[];
  /** The one whose card was fetched in full (task, todo, handoff), if any. */
  opened: SubAgent | null;
  onRefresh: () => void;
  onOpen: (id: string) => void;
  onClose: () => void;
  /** Inside the scientific-computing panel: no frame and no header of its own. */
  embedded?: boolean;
}

/**
 * The sub-agents of the session in view.
 *
 * Deliberately not in the session list: a sub-agent is not a conversation the user can
 * return to, it is work one of their sessions sent out. So it lives here, under the
 * session that spawned it, and the panel only reads — relaunching one from here would
 * be a second way to delegate, behind the orchestrator's back.
 */
export const SubAgentsPanel: React.FC<Props> = ({
  subAgents,
  opened,
  onRefresh,
  onOpen,
  onClose,
  embedded,
}) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // The cards are files another process writes; ask again on open, and after a run
  // ends the list is refreshed by the caller.
  useEffect(() => {
    onRefresh();
  }, [onRefresh]);

  // While something is working, keep the OPEN card live. The list itself is refreshed
  // by App, which polls for as long as a sub-agent is working whether this drawer is
  // open or not — the count on the button has to come down too, and a panel that owns
  // the only interval cannot make that happen while it is closed.
  const watching = anyRunning(subAgents);
  useEffect(() => {
    if (!watching || !expandedId) return;
    const timer = setInterval(() => onOpen(expandedId), 2000);
    return () => clearInterval(timer);
  }, [watching, expandedId, onOpen]);

  const toggle = (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      onClose();
      return;
    }
    setExpandedId(id);
    onOpen(id);
  };

  return (
    <div className={embedded ? "subagents-embedded" : "sessions-panel subagents-panel"}>
      {!embedded && (
        <div className="sessions-panel-header">
          <span className="sessions-panel-title">Sub-agents</span>
          <button
            className="sessions-toggle-btn"
            title="Refresh"
            aria-label="Refresh sub-agents"
            onClick={() => onRefresh()}
          >
            ⟳
          </button>
        </div>
      )}

      <div className={embedded ? "" : "sessions-list"}>
        {subAgents.length === 0 && (
          <div className="sessions-empty">
            Nothing delegated in this session yet.
          </div>
        )}

        {subAgents.map((sub) => {
          const isOpen = expandedId === sub.id;
          const detail = isOpen && opened?.id === sub.id ? opened : null;
          const todo = parseTodo(detail?.todo);
          return (
            <div key={sub.id} className={`subagent-row ${isOpen ? "open" : ""}`}>
              <button
                className="subagent-head"
                onClick={() => toggle(sub.id)}
                aria-expanded={isOpen}
              >
                <span className={`subagent-state state-${sub.state}`}>{stateLabel(sub)}</span>
                <span className="subagent-task">{taskTitle(sub)}</span>
              </button>
              <div className="subagent-grant">{grantSummary(sub)}</div>

              {isOpen && (
                <div className="subagent-detail">
                  {sub.model && <div className="subagent-model">{sub.model}</div>}
                  {detail && workspaceLine(detail) && (
                    <div className="subagent-model">{workspaceLine(detail)}</div>
                  )}
                  {/* Indexed key, not row.id: a status row has no id, and several in
                      a row would otherwise share the empty one. */}
                  {activityRows(detail?.activity).slice(-40).map((row, i) => (
                    <div key={`${row.id}-${i}`} className={`subagent-step step-${row.status}`}>
                      <span className="subagent-step-mark">
                        {row.status === "running"
                          ? "…"
                          : row.status === "ok"
                            ? "✓"
                            : row.status === "note"
                              ? "·"
                              : "✗"}
                      </span>
                      <span className="subagent-step-label">{row.label || row.name}</span>
                      {row.detail && <span className="subagent-step-detail">{row.detail}</span>}
                    </div>
                  ))}
                  {todo.length > 0 && (
                    <ul className="subagent-todo">
                      {todo.map((item, i) => (
                        <li key={i} className={item.done ? "done" : ""}>
                          {item.done ? "✓" : "○"} {item.text}
                        </li>
                      ))}
                    </ul>
                  )}
                  {detail?.answer && <div className="subagent-answer">{detail.answer}</div>}
                  {detail && !detail.answer && todo.length === 0 && (
                    <div className="sessions-empty">
                      It left no checklist and no handoff.
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
