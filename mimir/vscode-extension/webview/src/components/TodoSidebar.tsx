import React from "react";
import type { TodoItem } from "../types";

interface Props {
  items: TodoItem[];
}

export const TodoSidebar: React.FC<Props> = ({ items }) => {
  if (items.length === 0) return null;
  const pending = items.filter((i) => !i.done).length;

  return (
    <aside className="todo-sidebar">
      <div className="todo-header">
        <span className="todo-title">Tasks</span>
        {pending > 0 && (
          <span className="todo-badge">{pending} pending</span>
        )}
      </div>
      <ul className="todo-list">
        {items.map((item, idx) => (
          <li key={idx} className={`todo-item ${item.done ? "done" : ""}`}>
            <span className="todo-check">{item.done ? "✔" : "○"}</span>
            <span className="todo-text">{item.text}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
};
