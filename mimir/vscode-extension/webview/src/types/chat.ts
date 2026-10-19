// The chat as the panel holds it: messages by kind, tool rows, diffs.

import type { ApprovalMessage } from "./prompts";
import type { ExecResult, FileTarget, MathResult } from "./tools";
import type { CommandOutputMessage } from "./turn";

export type MessageRole = "user" | "agent";

export type MessageKind =
  | "text" | "approval" | "error" | "editing" | "thinking" | "tools" | "command";

/** A single tool invocation tracked from start (tool_call) to finish (tool_result). */
export interface ToolActivity {
  id: string;
  /** The structured tool name. Nothing renders it — the row is drawn from `kind` and
   *  `doing` — but it is what a stored transcript is read back by. */
  name: string;
  label: string;
  detail: string;
  /** The family of work this call belongs to: the word shown in the row, in full
   *  colour, and what its icon is derived from at render time. The icon is deliberately
   *  NOT stored here: it is a function of this, and one of them is an element that
   *  would not survive the stored transcript's JSON. See ToolCallMessage.kind. */
  kind?: string;
  /** The model's sentence about this call, shown dim beside the family. */
  doing?: string;
  /** "background" is a running row the user detached: settled for this turn, but the
   *  job carries on and a later job_complete fills in how it really ended. */
  status: "running" | "ok" | "error" | "background";
  summary?: string;
  /** Whether the row may be detached while running — see ToolCallMessage. */
  divertible?: boolean;
  /** Full error text of a failed call, shown in the expandable panel under the row. */
  error?: string;
  /** Terminal in/out panel data, revealed when the row is expanded (exec tools only). */
  exec?: ExecResult;
  /** Typeset calculation, shown open under the row (math-shaped results only). */
  math?: MathResult;
  /** The file a successful call touched: its name in the row opens it. */
  target?: FileTarget;
  durationMs?: number;
  /** What the model said this run's output showed — set when a verdict settled it. */
  verdict?: "pass" | "fail" | "unknown";
  /** Epoch ms when the call started — drives the live elapsed timer. */
  startedAt: number;
  /** Arrival stamp, used to place the row among this step's prose and reasoning
   *  (see orderLiveStream). A sub-agent's row inherits its parent's stamp so the
   *  family stays one group. Absent on rows restored from a saved session, which
   *  are already frozen in order. */
  seq?: number;
  /** Set on rows produced by a sub-agent: the id of the call that spawned it. */
  parentId?: string;
  /** Which sub-agent this row came from, e.g. "explore #2" — several run at once. */
  origin?: string;
  /** On a delegating row: child events shed rather than shown (queue or run ceiling). */
  childrenDropped?: number;
  /** On a delegating row: the child is mid-model-turn, with nothing to show yet. */
  waiting?: string;
  /** What a blocking run is doing right now, e.g. "building solver (1/2)". Live-only,
   *  like `waiting`: it describes a moment, and a settled row has none. */
  phase?: string;
  /** How far the current phase has got, 0-100 — set only when the phase counts
   *  itself. Live-only, and cleared when the row settles. */
  percent?: number;
  /** Set once a watcher holds this call's run: the work continues after the row has
   *  settled, and this is what later job events match against. Cleared by the
   *  `job_complete` that says how it ended. */
  jobKey?: string;
}

export interface DiffEntry {
  file: string;
  patch: string;
  /** Full proposed file content, present for whole-content write previews. */
  new_content?: string;
  /** True when the file did not exist before this operation. */
  is_new?: boolean;
  /** True when this operation deletes the file (all-red preview). */
  is_delete?: boolean;
}

export interface ChatMessage {
  id: string;
  /** Arrival stamp, set on every message that can land while a step is still in
   *  flight — a steer bubble, an edit card, a session-command reply — and on
   *  everything that step freezes. Stamped messages are kept in stamp order, so a
   *  card stays under the tool row that was still running above it. Absent on
   *  messages that end or precede a step (a query, an answer, an error), which
   *  simply go last. See orderLiveStream and appendStamped. */
  seq?: number;
  role: MessageRole;
  kind: MessageKind;
  text?: string;
  diffs?: DiffEntry[];
  thinking?: string;
  /** Reasoning duration (ms) for a frozen kind="thinking" message. */
  thinkingDurationMs?: number;
  /** Reasoning size (tokens) for a frozen kind="thinking" message. */
  thinkingTokens?: number;
  approval?: ApprovalMessage;
  /** The structured answer of a session command, for kind="command". */
  command?: CommandOutputMessage;
  streaming?: boolean;
  /** True while the agent is still editing (pulsing indicator shown). */
  live?: boolean;
  /** Structured tool activity for a frozen kind="tools" message. */
  tools?: ToolActivity[];
  /** A user steer message queued mid-run, awaiting injection (shows a "queued" tag
   *  until the server confirms with steer_injected). */
  queued?: boolean;
  /** Prose committed out of `draft` before the turn that wrote it was accepted —
   *  because the turn parked on a question, or the connection dropped under it.
   *  Kept so an interruption cannot lose the only copy, and dropped again by the
   *  `answer` that supersedes it. Renders exactly like any other text bubble, and
   *  is deliberately carried into the stored transcript: a reload landing while the
   *  turn is still parked must come back able to drop it. */
  provisional?: boolean;
}
