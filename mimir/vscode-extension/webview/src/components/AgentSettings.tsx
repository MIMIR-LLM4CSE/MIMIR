import React, { useEffect, useRef } from "react";
import type { ThinkingProfile } from "../types";

// Mode lives in its own toolbar control (ModeSwitcher), not in this popover.
interface Props {
  thinkingLevel: number;
  thinkingProfile?: ThinkingProfile;
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

/* ── Which rungs this model can actually honour ──────────────────────────────
 * The ladder above assumes the model takes a token budget. Some families take a
 * named effort rung instead, on their own scale — low/high/max for one, OpenAI's
 * low/medium/high for another — and some cannot stop reasoning at all. Offering the
 * full slider there would be six settings with two effects, so the scale is built
 * from what the server reported. The value handed back is always a real
 * THINKING_DEPTH index, so nothing downstream has to know about any of this.     */
interface DepthScale {
  depths: number[];
  labels: string[];
  budgets: string[];
  hints: string[];
  note?: string;
}

// Depths an effort ladder maps onto, cheapest first: quick / medium / deep / max.
const EFFORT_DEPTHS = [2, 3, 4, 5];

function buildScale(profile: ThinkingProfile | undefined): DepthScale {
  const mechanism = profile?.mechanism ?? "kwarg";
  const canDisable = profile?.can_disable ?? true;

  if (mechanism === "effort") {
    const levels = profile?.levels?.length ? profile.levels : ["low", "medium", "high"];
    const scale: DepthScale = {
      depths: levels.map((_, i) => EFFORT_DEPTHS[Math.min(i, EFFORT_DEPTHS.length - 1)]),
      labels: [...levels],
      budgets: levels.map((l) => `${l} effort`),
      hints: levels.map((l, i) =>
        i === 0 ? `Least reasoning this model will do (${l})`
        : i === levels.length - 1 ? `Most reasoning effort (${l})`
        : `Reasoning effort: ${l}`),
      note: canDisable
        ? "This model names its reasoning effort rather than budgeting it in tokens."
        : "This model always reasons — its effort scale has no \"off\".",
    };
    if (canDisable) {
      scale.depths.unshift(0);
      scale.labels.unshift("off");
      scale.budgets.unshift("disabled");
      scale.hints.unshift("No reasoning block at all");
    }
    return scale;
  }

  if (mechanism === "directive") {
    return {
      depths: [0, 1],
      labels: ["off", "on"],
      budgets: ["disabled", "model-chosen"],
      hints: [
        "No reasoning block at all",
        "Reasoning on, at whatever depth the model chooses",
      ],
      note: "This model is steered by a system-prompt directive, so it takes no token budget.",
    };
  }

  // enable_thinking + thinking_budget: the full ladder.
  return {
    depths: [0, 1, 2, 3, 4, 5],
    labels: [...DEPTH_LABELS],
    budgets: [...DEPTH_BUDGETS],
    hints: [...DEPTH_HINTS],
  };
}

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

// Approvals live in their own toolbar control (ApprovalSwitcher), not in this
// popover: the mode answers cards on the user's behalf and is switched mid-run, so
// its state has to be readable without opening anything.
export const AgentSettings: React.FC<Props> = ({
  thinkingLevel,
  thinkingProfile,
  streaming,
  contextMode,
  enforcement,
  onThinkingLevelChange,
  onStreamingToggle,
  onContextModeChange,
  onEnforcementChange,
  onClose,
}) => {
  // The depths this model can express, and where the current level sits on them.
  // An unavailable depth (e.g. "off" on a model that always reasons) falls back to
  // the nearest rung rather than showing an empty selection.
  const scale = buildScale(thinkingProfile);
  const exact = scale.depths.indexOf(thinkingLevel);
  const rung = exact >= 0
    ? exact
    : scale.depths.reduce(
        (best, d, i) => (Math.abs(d - thinkingLevel) < Math.abs(scale.depths[best] - thinkingLevel) ? i : best),
        0,
      );

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

      {/* Thinking depth slider — rungs come from the model's mechanism */}
      <div className="settings-section-label">Thinking depth</div>
      <div className="settings-depth-wrap">
        <div className="settings-depth-header">
          <span className={`settings-depth-badge ${scale.depths[rung] === 0 ? "off" : "on"}`}>
            💭 {scale.labels[rung]}
          </span>
          <span className="settings-depth-budget">{scale.budgets[rung]}</span>
        </div>
        <input
          type="range"
          min={0}
          max={scale.depths.length - 1}
          step={1}
          value={rung}
          className="settings-depth-slider"
          style={{ "--pct": `${(rung * 100) / Math.max(1, scale.depths.length - 1)}%` } as React.CSSProperties}
          onChange={(e) => onThinkingLevelChange(scale.depths[Number(e.target.value)])}
        />
        <div className="settings-depth-ticks">
          {scale.labels.map((label, i) => (
            <span
              key={i}
              className={`settings-depth-tick ${i === rung ? "active" : ""}`}
              title={scale.hints[i]}
              onClick={() => onThinkingLevelChange(scale.depths[i])}
            >
              {label}
            </span>
          ))}
        </div>
        <div className="settings-context-hint">{scale.hints[rung]}</div>
        {scale.note && <div className="settings-context-hint settings-depth-note">{scale.note}</div>}
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
