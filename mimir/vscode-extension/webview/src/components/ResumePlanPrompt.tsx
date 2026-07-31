import React from "react";
import type { TodoItem } from "../types";

interface Props {
  items: TodoItem[];
  onChoice: (resume: boolean) => void;
}

export const ResumePlanPrompt: React.FC<Props> = ({ items, onChoice }) => {
  const pending = items.filter((i) => !i.done);
  const done    = items.filter((i) =>  i.done);

  return (
    <div className="resume-overlay">
      <div className="resume-modal">
        <div className="resume-title">📋 Previous session plan</div>
        <div className="resume-subtitle">
          {pending.length} task{pending.length !== 1 ? "s" : ""} remaining
          {done.length > 0 ? `, ${done.length} completed` : ""}
        </div>
        <ul className="resume-list">
          {items.map((item, i) => (
            <li key={i} className={`resume-item ${item.done ? "resume-item--done" : ""}`}>
              <span className="resume-item-check">{item.done ? "✔" : "○"}</span>
              {item.text}
            </li>
          ))}
        </ul>
        <div className="resume-actions">
          <button className="resume-btn resume-btn--yes" onClick={() => onChoice(true)}>
            Resume plan
          </button>
          <button className="resume-btn resume-btn--no" onClick={() => onChoice(false)}>
            Start fresh
          </button>
        </div>
      </div>
    </div>
  );
};
