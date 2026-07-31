import React, { useState } from "react";
import type { TodoItem } from "../types";

interface Props {
  items: TodoItem[];
  busy: boolean;
  onClear?: () => void;
}

export const PlanBar: React.FC<Props> = ({ items, busy, onClear }) => {
  const [expanded, setExpanded] = useState(false);

  if (items.length === 0) return null;

  const doneCount = items.filter((i) => i.done).length;
  const total = items.length;
  const allDone = doneCount === total;

  // Current step: first pending item, or last item when all done.
  const currentIdx = allDone
    ? total - 1
    : items.findIndex((i) => !i.done);
  const currentItem = items[currentIdx];

  const progressPct = total > 0 ? Math.round((doneCount / total) * 100) : 0;

  return (
    <div className={`plan-bar ${expanded ? "plan-bar--expanded" : ""} ${allDone ? "plan-bar--done" : ""}`}>
      {/* ── Collapsed header (always visible) ──────────────────────── */}
      <div
        className="plan-bar-header"
        onClick={() => setExpanded((v) => !v)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && setExpanded((v) => !v)}
      >
        {/* Progress bar */}
        <div className="plan-bar-progress-track">
          <div
            className="plan-bar-progress-fill"
            style={{ width: `${progressPct}%` }}
          />
        </div>

        {/* Current step label */}
        <div className="plan-bar-current">
          {busy && !allDone && (
            <span className="plan-bar-spinner" aria-hidden>⟳</span>
          )}
          {allDone ? (
            <span className="plan-bar-done-label">✔ Plan complete</span>
          ) : (
            <span className="plan-bar-step-label">
              <span className="plan-bar-step-counter">{doneCount + 1}/{total}</span>
              {currentItem?.text}
            </span>
          )}
        </div>

        {/* Clear button */}
        {onClear && (
          <button
            className="plan-bar-clear"
            title="Clear plan"
            onClick={(e) => { e.stopPropagation(); onClear(); }}
            tabIndex={-1}
          >
            ✕
          </button>
        )}

        {/* Expand chevron */}
        <button
          className="plan-bar-chevron"
          title={expanded ? "Collapse plan" : "Show all steps"}
          onClick={(e) => { e.stopPropagation(); setExpanded((v) => !v); }}
          tabIndex={-1}
        >
          {expanded ? "▲" : "▼"}
        </button>
      </div>

      {/* ── Expanded step list ──────────────────────────────────────── */}
      {expanded && (
        <ul className="plan-bar-list">
          {items.map((item, idx) => {
            const isCurrent = idx === currentIdx && !allDone;
            return (
              <li
                key={idx}
                className={`plan-bar-item ${item.done ? "done" : ""} ${isCurrent ? "current" : ""}`}
              >
                <span className="plan-bar-item-icon">
                  {item.done ? "✔" : isCurrent ? (busy ? "⟳" : "▶") : "○"}
                </span>
                <span className="plan-bar-item-text">{item.text}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};
