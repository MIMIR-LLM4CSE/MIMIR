// The agent's settings as the panel and the server exchange them, and the servers and skills
// switched on or off.

/** How the served model is switched between reasoning and not. Mirrors
 *  `thinking_profile()` in client/config/models.py. */
export type ThinkingMechanism = "kwarg" | "directive" | "effort";

/** What the depth control is allowed to offer for the served model.
 *  `levels` are the family's OWN effort rungs, weakest first — they are not a fixed
 *  scale (low/high/max for some families, low/medium/high for others) — and
 *  `can_disable` is false for a family that always reasons, whose "off" rung would
 *  be a control that does nothing. */
export interface ThinkingProfile {
  mechanism: ThinkingMechanism;
  levels: string[];
  can_disable: boolean;
}

/** Whether the backend honours a sampling temperature, and the one the user set for
 *  the served model. `value` null is the model's own: nothing is sent, and the server
 *  applies the model's generation_config. */
export interface TemperatureState {
  supported: boolean;
  value: number | null;
}

/** How far the user lets a sub-agent go. Mirrors config.constants.SUBAGENT_LEVELS:
 *  "explore" reads, "build" also writes — each in a copy of the repository, which
 *  is why there is no rung in between. */
export type SubAgentLevel = "explore" | "build";

export interface SubAgentLevelMessage {
  type: "subagent_level";
  level: SubAgentLevel;
}

export interface ContextModeMessage {
  type: "context_mode";
  mode: "compact" | "full";
}

export interface EnforcementModeMessage {
  type: "enforcement";
  mode: "strict" | "light" | "off";
}

/** How much of the approval flow the user still answers by hand. Mirrors
 *  ApprovalManager.APPROVAL_MODES in client/guardrails/policy/approval.py:
 *  manual (every card), auto (sensitive tools pass, leaving the workspace still
 *  asks), auto_all (nothing asks). Session state on the server — never persisted. */
export type ApprovalMode = "manual" | "auto" | "auto_all";

export interface ApprovalModeMessage {
  type: "approval_mode";
  mode: ApprovalMode;
}

/** The agent's operating mode. Mirrors VALID_MODES in client/config/models.py. */
export type AgentMode = "agent" | "plan" | "ask";

/** Server-driven mode change (e.g. plan mode switching to agent on approval). */
export interface AgentModeMessage {
  type: "mode";
  mode: AgentMode;
}

/**
 * The reasoning depth the agent actually holds, reported after any change.
 *
 * State, not narration: the depth has a control in the settings panel, so a line
 * about it in the transcript repeats the chrome — and the webview replays its
 * stored settings on every connect, which made that line greet each session.
 */
export interface ThinkingDepthMessage {
  type: "thinking_depth";
  depth: number;
  label?: string;
}

/** The temperature the agent holds, after /temperature. */
export interface TemperatureMessage extends TemperatureState {
  type: "temperature";
}

/** Whether the agent is actually streaming, reported after any change. */
export interface StreamingStateMessage {
  type: "streaming";
  enabled: boolean;
}

/** A togglable server or skill row in the toggle panel. */
export interface ToggleItem {
  name: string;
  description: string;
  enabled: boolean;
}

export interface TogglesListMessage {
  type: "toggles_list";
  servers: ToggleItem[];
  skills: ToggleItem[];
}

export interface ListTogglesMessage {
  type: "list_toggles";
}

export interface ToggleServerMessage {
  type: "toggle_server";
  name: string;
  enabled: boolean;
}

export interface ToggleSkillMessage {
  type: "toggle_skill";
  name: string;
  enabled: boolean;
}
