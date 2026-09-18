import React, { useEffect, useRef, useState } from "react";
import type { TemperatureState, ThinkingProfile } from "../types";

// Mode lives in its own toolbar control (ModeSwitcher), not in this popover.
interface Props {
  thinkingLevel: number;
  thinkingProfile?: ThinkingProfile;
  streaming: boolean;
  contextMode: "compact" | "full";
  enforcement: "strict" | "light" | "off";
  temperature: TemperatureState;
  onTemperatureChange: (value: number | null) => void;
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

/* ── Temperature ─────────────────────────────────────────────────────────────
 * Mirrors TEMPERATURE_MIN / TEMPERATURE_MAX in client/config/constants.py. The
 * hints are tendencies, not thresholds: the right value differs per model, which is
 * why "Model default" (nothing sent) is the starting state.                      */
const TEMP_MIN = 0;
const TEMP_MAX = 2;
const TEMP_STEP = 0.05;
// Where unticking "Model default" puts the slider: neutral for most families.
const TEMP_START = 1.0;

function temperatureHint(value: number, thinkingOn: boolean): { text: string; warn: boolean } {
  if (value < 0.4) {
    return thinkingOn
      ? { text: "⚠ Low with reasoning on: risk of repetition loops in the thinking", warn: true }
      : { text: "Near-deterministic: little variation between attempts", warn: false };
  }
  if (value <= 1.0) return { text: "Usual range for reasoning models", warn: false };
  if (value <= 1.3) return { text: "More varied output; occasional drift", warn: false };
  return { text: "⚠ Incoherent output and malformed tool calls likely", warn: true };
}

// Approvals live in their own toolbar control (ApprovalSwitcher), not in this
// popover: the mode answers cards on the user's behalf and is switched mid-run, so
// its state has to be readable without opening anything.
export const AgentSettings: React.FC<Props> = ({
  thinkingLevel,
  thinkingProfile,
  streaming,
  contextMode,
  enforcement,
  temperature,
  onTemperatureChange,
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

  // The slider moves a local draft; the value is sent on release only, since each
  // send is a write to the preferences file.
  const [tempDraft, setTempDraft] = useState<number>(temperature.value ?? TEMP_START);
  useEffect(() => {
    setTempDraft(temperature.value ?? TEMP_START);
  }, [temperature.value]);
  const commitTemp = () => {
    if (temperature.value !== null && tempDraft !== temperature.value) onTemperatureChange(tempDraft);
  };
  const tempHint = temperatureHint(tempDraft, thinkingLevel > 0);

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

      {temperature.supported && (
        <>
          <div className="settings-divider" />

          {/* Sampling temperature — per model; the default sends none */}
          <div className="settings-section-label">Temperature</div>
          <div
            className="settings-toggle-row"
            title="Send no temperature: the server applies the model's own recommended sampling"
            onClick={() => onTemperatureChange(temperature.value === null ? TEMP_START : null)}
          >
            <span className="settings-toggle-label">Model default</span>
            <span className={`settings-toggle ${temperature.value === null ? "on" : "off"}`}>
              {temperature.value === null ? "on" : "off"}
            </span>
          </div>
          {temperature.value !== null && (
            <div className="settings-depth-wrap">
              <div className="settings-depth-header">
                <span className="settings-depth-badge on">🌡 {tempDraft.toFixed(2)}</span>
                <span className="settings-depth-budget">this model only</span>
              </div>
              <input
                type="range"
                min={TEMP_MIN}
                max={TEMP_MAX}
                step={TEMP_STEP}
                value={tempDraft}
                className="settings-depth-slider"
                style={{ "--pct": `${((tempDraft - TEMP_MIN) * 100) / (TEMP_MAX - TEMP_MIN)}%` } as React.CSSProperties}
                onChange={(e) => setTempDraft(Math.round(Number(e.target.value) * 100) / 100)}
                onPointerUp={commitTemp}
                onKeyUp={commitTemp}
                onBlur={commitTemp}
              />
              <div className={`settings-context-hint ${tempHint.warn ? "settings-temp-warn" : ""}`}>
                {tempHint.text}
              </div>
            </div>
          )}
          <div className="settings-context-hint settings-depth-note">
            Recommended values differ per model; "Model default" uses its publisher's.
          </div>
        </>
      )}

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
