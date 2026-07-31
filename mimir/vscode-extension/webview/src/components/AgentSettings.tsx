import React, { useEffect, useRef } from "react";

// Mode lives in its own toolbar control (ModeSwitcher), not in this popover.
interface Props {
  thinkingLevel: number;
  streaming: boolean;
  contextMode: "compact" | "full";
  enforcement: "strict" | "light" | "off";
  onThinkingLevelChange: (level: number) => void;
  onStreamingToggle: (val: boolean) => void;
  onContextModeChange: (val: "compact" | "full") => void;
  onEnforcementChange: (val: "strict" | "light" | "off") => void;
  onClose: () => void;
}

// Mirrors THINKING_DEPTH_LABELS / THINKING_DEPTH_BUDGETS in client/config/constants.py.
const DEPTH_LABELS = ["off", "auto", "quick", "medium", "deep", "max"] as const;
const DEPTH_BUDGETS = ["disabled", "model-chosen", "~500 tok", "~4K tok", "~16K tok", "unlimited"] as const;
const DEPTH_HINTS = [
  "No reasoning block at all",
  "The model sets its own depth per turn — near-zero on trivial work, long where it matters",
  "Fixed ~500-token reasoning budget",
  "Fixed ~4K-token reasoning budget",
  "Fixed ~16K-token reasoning budget",
  "Unbudgeted reasoning on every turn",
] as const;

const ENFORCEMENT_OPTIONS = [
  { value: "strict", label: "🛡 strict", title: "All guidance nudges on (discovery, doc, state, etc.) — best for smaller models" },
  { value: "light", label: "⚖ light", title: "Drop the chatty discovery nudge, keep the rest of the guidance layer" },
  { value: "off", label: "🚀 off", title: "No guidance nudges — verification & safety guards still on; best for strong models" },
] as const;

const ENFORCEMENT_HINTS: Record<"strict" | "light" | "off", string> = {
  strict: "All guidance nudges on — best for smaller models",
  light: "Discovery nudge off, rest of guidance kept",
  off: "Guidance off — verification & safety always on",
};

export const AgentSettings: React.FC<Props> = ({
  thinkingLevel,
  streaming,
  contextMode,
  enforcement,
  onThinkingLevelChange,
  onStreamingToggle,
  onContextModeChange,
  onEnforcementChange,
  onClose,
}) => {
  const ref = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.closest(".settings-wrapper")?.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div className="settings-popover" ref={ref}>
      {/* Context window strategy */}
      <div className="settings-section-label">Context memory</div>
      <div className="settings-row">
        <button
          className={`settings-option-btn ${contextMode === "compact" ? "active" : ""}`}
          title="Compact: summarise history after each write — efficient for small models"
          onClick={() => { onContextModeChange("compact"); onClose(); }}
        >
          ⚡ compact
        </button>
        <button
          className={`settings-option-btn settings-option-btn--full ${contextMode === "full" ? "active" : ""}`}
          title="Full: keep all tool messages in history — best on large-context models"
          onClick={() => { onContextModeChange("full"); onClose(); }}
        >
          🧠 full
        </button>
      </div>
      <div className="settings-context-hint">
        {contextMode === "compact"
          ? "History compacted after writes — best for small models"
          : "All tool results kept in history — best on large-context models"}
      </div>

      <div className="settings-divider" />

      {/* Enforcement (guidance-nudge tiering) */}
      <div className="settings-section-label">Enforcement</div>
      <div className="settings-row">
        {ENFORCEMENT_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            className={`settings-option-btn ${enforcement === opt.value ? "active" : ""}`}
            title={opt.title}
            onClick={() => { onEnforcementChange(opt.value); onClose(); }}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <div className="settings-context-hint">{ENFORCEMENT_HINTS[enforcement]}</div>

      <div className="settings-divider" />

      {/* Thinking depth slider */}
      <div className="settings-section-label">Thinking depth</div>
      <div className="settings-depth-wrap">
        <div className="settings-depth-header">
          <span className={`settings-depth-badge ${thinkingLevel === 0 ? "off" : "on"}`}>
            💭 {DEPTH_LABELS[thinkingLevel]}
          </span>
          <span className="settings-depth-budget">{DEPTH_BUDGETS[thinkingLevel]}</span>
        </div>
        <input
          type="range"
          min={0}
          max={DEPTH_LABELS.length - 1}
          step={1}
          value={thinkingLevel}
          className="settings-depth-slider"
          style={{ "--pct": `${(thinkingLevel * 100) / (DEPTH_LABELS.length - 1)}%` } as React.CSSProperties}
          onChange={(e) => onThinkingLevelChange(Number(e.target.value))}
        />
        <div className="settings-depth-ticks">
          {DEPTH_LABELS.map((label, i) => (
            <span
              key={i}
              className={`settings-depth-tick ${i === thinkingLevel ? "active" : ""}`}
              title={DEPTH_HINTS[i]}
              onClick={() => onThinkingLevelChange(i)}
            >
              {label}
            </span>
          ))}
        </div>
        <div className="settings-context-hint">{DEPTH_HINTS[thinkingLevel]}</div>
      </div>

      <div className="settings-divider" />

      {/* Streaming toggle */}
      <div className="settings-section-label">Capabilities</div>
      <div className="settings-toggle-row" onClick={() => onStreamingToggle(!streaming)}>
        <span className="settings-toggle-label">⚡ streaming</span>
        <span className={`settings-toggle ${streaming ? "on" : "off"}`}>
          {streaming ? "on" : "off"}
        </span>
      </div>
    </div>
  );
};
