import React, { useEffect, useRef, useState } from "react";
import { MimirMark } from "./MimirMark";
import type { AgentMode } from "../types";

interface Props {
  mode: AgentMode;
  onModeChange: (mode: AgentMode) => void;
}

// Mirrors VALID_MODES in client/config/models.py. Each mode carries its own
// description because the labels alone no longer say what the mode will do.
// The colour is the mode's identity across the whole chat (see --mode-accent in
// main.css): blue = agent, red = plan, green = ask.
export const MODE_OPTIONS: {
  value: AgentMode;
  icon: React.ReactNode;
  label: string;
  desc: string;
}[] = [
  {
    value: "agent",
    icon: <MimirMark />,
    label: "agent",
    desc: "Full tool use — explores, edits, runs, and validates",
  },
  {
    value: "plan",
    icon: "📋",
    label: "plan",
    desc: "Read-only research, then a plan you approve before any work starts",
  },
  {
    value: "ask",
    icon: "💬",
    label: "ask",
    desc: "Read-only Q&A about the codebase — no edits, no plan",
  },
];

/** Mode picker as its own toolbar control, separate from agent settings. */
export const ModeSwitcher: React.FC<Props> = ({ mode, onModeChange }) => {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const active = MODE_OPTIONS.find((o) => o.value === mode) ?? MODE_OPTIONS[0];

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="mode-wrapper" ref={wrapRef}>
      <button
        className="settings-btn mode-btn"
        title={`Mode: ${active.label} — ${active.desc}`}
        aria-label={`Agent mode: ${active.label}`}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="mode-btn-icon">{active.icon}</span>
        <span className="mode-btn-label">{active.label}</span>
        <span className="mode-btn-caret">▾</span>
      </button>
      {open && (
        <div className="settings-popover mode-popover">
          <div className="settings-section-label">Mode</div>
          <div className="settings-mode-list">
            {MODE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                data-mode={opt.value}
                className={`settings-option-btn settings-mode-btn ${mode === opt.value ? "active" : ""}`}
                onClick={() => {
                  onModeChange(opt.value);
                  setOpen(false);
                }}
              >
                <span className="settings-mode-btn-label">
                  {opt.icon} {opt.label}
                </span>
                <span className="settings-mode-btn-desc">{opt.desc}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
