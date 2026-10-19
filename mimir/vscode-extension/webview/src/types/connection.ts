// Reaching the server: the endpoint, its models, the server this workspace already has, and
// leaving it running without this window.

import type { ApprovalMode, TemperatureState, ThinkingProfile } from "./settings";

export interface ReadyMessage {
  type: "ready";
  model: string;
  /** Whether the agent itself exists yet. The socket greeting is sent first and
   *  says false; the worker's own greeting, after the LLM backend answers, says
   *  true. Absent from an older server, which the client reads as ready. */
  agent_ready?: boolean;
  context_mode?: "compact" | "full";
  enforcement?: "strict" | "light" | "off";
  approval_mode?: ApprovalMode;
  thinking?: ThinkingProfile;
  temperature?: TemperatureState;
}

export interface ConfigMessage {
  type: "config";
  backend?: string;
  vllmBaseUrl?: string;
  rayBaseUrl?: string;
  ollamaBaseUrl?: string;
  anthropicModels?: string[];
  /** Endpoint the user asked the host to remember, if any. */
  remembered?: RememberedEndpoint | null;
  /** The server this workspace already has, when it has one.
   *
   *  Present means a run is going without this panel, and rejoining it is a click
   *  rather than a connection: the backend and address in this message describe how a
   *  *new* server would be started, and none of them apply to one already serving. */
  running?: RunningServer | null;
}

/** A server already serving this workspace. */
export interface RunningServer {
  url: string;
  pid: number;
  model: string;
  autonomy: string;
  /** True when it was deliberately left running without a window. */
  detached: boolean;
}

/** Address (never a key) the extension host reconnects to unattended. */
export interface RememberedEndpoint {
  backend: string;
  baseUrl: string;
  model: string;
}

/**
 * The host started connecting on its own to the remembered endpoint — the
 * webview only has to show the connecting state it didn't ask for.
 */
export interface AutoConnectMessage extends RememberedEndpoint {
  type: "auto_connect";
}

/** Models the extension host read from the endpoint the user pointed us at. */
export interface ModelsMessage {
  type: "models";
  backend: string;
  models: string[];
  error?: string;
}

/**
 * What the endpoint serves, reported by the agent server once connected.
 *
 * Distinct from `ModelsMessage`, which is the extension host's own probe of an
 * address typed into the connect form: this one comes from the process that holds
 * the endpoint and its key, so the picker has a list even when that probe never got
 * through (a corporate proxy with no route to the cluster swallows it).
 */
export interface ServedModelsMessage {
  type: "served_models";
  models: string[];
}

/** The served model changed mid-session — the status bar and derived controls follow. */
export interface ModelChangedMessage {
  type: "model_changed";
  model: string;
  /** Re-derived thinking profile for the new model, when the server reports it. */
  thinking?: ThinkingProfile;
  /** Re-derived enforcement level for the new model, when the server reports it. */
  enforcement?: "strict" | "light" | "off";
  /** The temperature stored for the new model (each model keeps its own). */
  temperature?: TemperatureState;
}

/** The server has made itself survivable — the answer to `detach`.
 *
 *  It has re-pointed its output at `log` (so the pipe this window holds can close
 *  without taking it down) and left this process group. The extension stops killing it
 *  on window close from here on: it is no longer this window's to end. `autonomy` is
 *  what it may do while unattended, and `sessions` are the conversations it applies to. */
export interface DetachedMessage {
  type: "detached";
  /** False when the decision has been taken back and this window owns the server
   *  again. Absent means detached: the message once existed only to announce one. */
  detached?: boolean;
  log: string | null;
  autonomy: "manual" | "auto" | "auto_all";
  sessions: string[];
  pid: number;
  setsid: boolean;
}

/** "Continue without me": make the server survive this window closing.
 *
 *  `autonomy` is required rather than defaulted, because the answer matters: at
 *  `manual` a detached run parks at its first sensitive tool and does almost nothing
 *  overnight, and that has to be a choice. Omitting `session_ids` means every
 *  conversation — and that is the only form that outlives a worker being rebuilt,
 *  since the server records UI settings per pool rather than per session. */
export interface DetachMessage {
  type: "detach";
  autonomy: "manual" | "auto" | "auto_all";
  session_ids?: string[];
  /** `false` takes the decision back: the window owns the server again and closing it
   *  stops the server. The process-level work is not undone — fds pointing at a log
   *  file have no pipe to return to — and does not need to be: what makes a server
   *  survive a window closing is that nobody kills it, which is a decision rather
   *  than a state of the process. */
  enabled?: boolean;
}

/**
 * Stop the server, now or once it has nothing left to do. Refused with the reasons while
 * anything works, unless `force`: the user's call about their own machine.
 */
export interface ShutdownMessage {
  type: "shutdown";
  force?: boolean;
}

export interface ConnectMessage {
  type: "connect";
  model: string;
  backend?: string;
  vllmBaseUrl?: string;
}

/** Switch the served model mid-session (no reconnect). */
export interface SetModelMessage {
  type: "set_model";
  model: string;
}

export type ConnectionState = "disconnected" | "connecting" | "connected" | "error";
