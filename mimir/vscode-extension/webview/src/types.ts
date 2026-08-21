// ── Message types (server → client) ──────────────────────────────────────────

export interface ReadyMessage {
  type: "ready";
  model: string;
  context_mode?: "compact" | "full";
  enforcement?: "strict" | "light" | "off";
}

export interface OutputMessage {
  type: "output";
  text: string;
}

export interface StatusMessage {
  type: "status";
  text: string;
}

export interface ApprovalMessage {
  type: "approval";
  id: string;
  /** All approval IDs when multiple concurrent approvals are merged into one card. */
  ids?: string[];
  tool: string;
  server: string;
  args: Record<string, unknown>;
  risk: string;
  /** Declared undo level for this call: how much of its effect can be taken back.
   *  Drives the severity badge from a value instead of keyword-matching `risk`.
   *  Optional so a card from an older server still renders. */
  reversibility?: "reversible" | "recoverable" | "irreversible";
  scope: string;
  /** Canonical human label ("Proxy exec: run"); absent for tools with no template. */
  label?: string;
  /** Set when the call is held because it touches this path outside the workspace.
   *  The rest of the card describes the tool call itself, as usual. */
  oow_path?: string;
}

export interface AnswerMessage {
  type: "answer";
  text: string;
  cancelled?: boolean;
}

export interface TokenMessage {
  type: "token";
  text: string;
}

export interface TodoItem {
  text: string;
  done: boolean;
}

export interface TodoMessage {
  type: "todo";
  items: TodoItem[];
}

export interface ErrorMessage {
  type: "error";
  text: string;
}

export interface ConfigMessage {
  type: "config";
  models: string[];
  vllmModels?: string[];
  anthropicModels?: string[];
  modelSizes?: Record<string, number>;
  slurmEnabled: boolean;
  clusterConfig: ClusterConfig[];
  backend?: string;
  vllmBaseUrl?: string;
  vllmMode?: "launch" | "connect";
}

export interface GpuSpec {
  type: string;
  memGB: number;
  maxCount: number;
}

export interface NodeType {
  label: string;
  partition: string;
  cpusPerNode: number;
  gpu: GpuSpec | null;
  memOptionsGB: number[];
}

export interface ClusterConfig {
  name: string;
  loginNode?: string;
  account?: string;
  nodeTypes: NodeType[];
  ollamaPath?: string;
  vllmPath?: string;
}

export interface ThinkingMessage {
  type: "thinking";
  text: string;
}

export interface ToolCallMessage {
  type: "tool_call";
  /** Correlation id matching a later tool_result. */
  id: string;
  /** Raw tool name (used to pick an icon). */
  name: string;
  /** Human-readable label, e.g. "Reading file: x.py". */
  label: string;
  /** Short arg preview, e.g. a command line or search pattern (may be empty). */
  detail: string;
}

/** Terminal in/out of an exec-shaped tool result (shell / code runner / compiler).
 *  Sent by the server only for results carrying returncode + stdout/stderr. */
export interface ExecResult {
  /** Full command or code body that was run (clipped server-side). */
  command?: string;
  stdout: string;
  stderr: string;
  returncode: number;
  cwd?: string;
  /** True when a stream was clipped (server or wire budget). */
  truncated?: boolean;
}

export interface ToolResultMessage {
  type: "tool_result";
  /** Correlation id matching an earlier tool_call. */
  id: string;
  name: string;
  ok: boolean;
  /** Short outcome summary, e.g. "3 matches", "12 lines", or an error line. */
  summary: string;
  /** Full, untruncated error text — present only on failures. */
  error?: string;
  /** Terminal panel data, present only for exec-shaped results. */
  exec?: ExecResult;
  duration_ms: number;
}

/** What the model said a run's output showed. Exit 0 says a program ended, so the run
 *  row stays unjudged until this lands on it — it is a claim, not a measurement. */
export interface VerdictMessage {
  type: "verdict";
  /** Correlation id of the run being judged (an earlier tool_call). */
  id: string;
  verdict: "pass" | "fail" | "unknown";
}

export interface ThinkingStartMessage {
  type: "thinking_start";
}

export interface ThinkingEndMessage {
  type: "thinking_end";
  /** Size of the reasoning block just closed, counted server-side with the same
   *  tokenizer as the context bar. Absent on older servers. */
  tokens?: number;
}

export interface TodoPromptMessage {
  type: "todo_prompt";
  items: TodoItem[];
}

export interface DiffMessage {
  type: "diff";
  file: string;
  patch: string;
}


/** Confirms a mid-run steer message was injected into the running agent's turn. */
export interface SteerInjectedMessage {
  type: "steer_injected";
  text: string;
}

/** A workflow reminder the guardrail layer injected into the agent's user turn.
 *  Surfaced so a run that changes direction mid-way is attributable: the nudge
 *  occupies the user's turn slot but did NOT come from the user. */
export interface NudgeInjectedMessage {
  type: "nudge_injected";
  category: string;
  text: string;
}

// ── Session types ──────────────────────────────────────────────────────────────

export interface SessionMeta {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  preview?: string;
  /** One-sentence description of what was done in the session. */
  summary?: string;
  /** True once the user renamed the session by hand — the title then wins. */
  title_custom?: boolean;
}

export interface SessionsListMessage {
  type: "sessions_list";
  sessions: SessionMeta[];
}

export interface SessionLoadedMessage {
  type: "session_loaded";
  session_id: string;
  title: string;
  display_messages: ChatMessage[];
  todos: TodoItem[];
}

export interface ContextModeMessage {
  type: "context_mode";
  mode: "compact" | "full";
}

export interface EnforcementModeMessage {
  type: "enforcement";
  mode: "strict" | "light" | "off";
}

/** The agent's operating mode. Mirrors VALID_MODES in client/config/models.py. */
export type AgentMode = "agent" | "plan" | "ask";

/** Server-driven mode change (e.g. plan mode switching to agent on approval). */
export interface AgentModeMessage {
  type: "mode";
  mode: AgentMode;
}

export interface ContinuePromptMessage {
  type: "continue_prompt";
  id: string;
  summary: string;
}

export interface JobCompleteMessage {
  type: "job_complete";
  job_key: string;
  server?: string;
  kind?: string;
  state: string;
  summary?: Record<string, unknown>;
}

export interface QuestionOption {
  label: string;
  description?: string;
}

export interface QuestionSpec {
  question: string;
  header: string;
  options: QuestionOption[];
  multiSelect: boolean;
}

export interface UserQuestionMessage {
  type: "user_question";
  id: string;
  questions: QuestionSpec[];
}

export interface QuestionAnswer {
  selected: string[];
  otherText?: string;
}

export interface ContextUsageMessage {
  type: "context_usage";
  /** Estimated tokens used by the current history. */
  used_tokens: number;
  /** Total context window size for the active mode. */
  total_tokens: number;
  /** Tokens reserved for the model's answer (auto-compact fires below this headroom). */
  reserved_tokens: number;
  /** Fixed per-call overhead (system prompt + tools schema) included in used_tokens. */
  overhead_tokens?: number;
}

export interface FileProgressMessage {
  type: "file_progress";
  diffs: DiffEntry[];
}

export interface BatchStatusMessage {
  type: "batch_status";
  files: DiffEntry[];
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

/** An MCP resource that can be @-attached to a query. */
export interface ResourceItem {
  uri: string;
  name: string;
  description: string;
  mimeType?: string | null;
}

export interface ResourcesMessage {
  type: "resources";
  resources: ResourceItem[];
}

/** The file (and optional selected line range) focused in the VS Code editor. */
export interface ActiveEditorMessage {
  type: "active_editor";
  file: string;
  selection: string | null; // "10" | "10-20" | null
}

/** Ask the host to open a file in a real editor tab (e.g. a written plan .md).
 *  Emitted by the agent on a tool result carrying open_in_editor + path. */
export interface OpenEditorMessage {
  type: "open_editor";
  path: string;
}

export type ServerMessage =
  | ReadyMessage
  | OutputMessage
  | StatusMessage
  | ApprovalMessage
  | AnswerMessage
  | TokenMessage
  | TodoMessage
  | TodoPromptMessage
  | ThinkingMessage
  | ThinkingStartMessage
  | ThinkingEndMessage
  | ToolCallMessage
  | ToolResultMessage
  | VerdictMessage
  | ErrorMessage
  | ConfigMessage
  | DiffMessage
  | SteerInjectedMessage
  | NudgeInjectedMessage
  | FileProgressMessage
  | BatchStatusMessage
  | SessionsListMessage
  | SessionLoadedMessage
  | ContextModeMessage
  | EnforcementModeMessage
  | AgentModeMessage
  | ContextUsageMessage
  | ContinuePromptMessage
  | JobCompleteMessage
  | TogglesListMessage
  | ResourcesMessage
  | ActiveEditorMessage
  | OpenEditorMessage
  | UserQuestionMessage;

// ── Message types (client → server) ──────────────────────────────────────────

export interface QueryMessage {
  type: "query";
  text: string;
}

/** A message typed while the agent is busy — injected into the running turn. */
export interface SteerMessage {
  type: "steer";
  text: string;
}

export interface ApprovalResponseMessage {
  type: "approval_response";
  id: string;
  choice: "y" | "n" | "a";
  approved_files?: string[];
}

export interface ContinueResponseMessage {
  type: "continue_response";
  id: string;
  choice: "y" | "n";
}

export interface UserQuestionResponseMessage {
  type: "user_question_response";
  id: string;
  answers: QuestionAnswer[];
}

export interface CommandMessage {
  type: "command";
  text: string;
}

export interface ConnectMessage {
  type: "connect";
  model: string;
  backend?: string;
  vllmBaseUrl?: string;
}

export interface CreateSessionMessage {
  type: "create_session";
}

export interface SwitchSessionMessage {
  type: "switch_session";
  session_id: string;
}

export interface DeleteSessionMessage {
  type: "delete_session";
  session_id: string;
}

export interface RenameSessionMessage {
  type: "rename_session";
  session_id: string;
  title: string;
}

export interface BatchReviewAcceptMessage {
  type: "batch_review_accept";
}

export interface BatchReviewRevertMessage {
  type: "batch_review_revert";
}

export interface BatchReviewAcceptFileMessage {
  type: "batch_review_accept_file";
  file: string;
}

export interface BatchReviewRevertFileMessage {
  type: "batch_review_revert_file";
  file: string;
}

export interface ListTogglesMessage {
  type: "list_toggles";
}

export interface ListResourcesMessage {
  type: "list_resources";
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

export type ClientMessage =
  | QueryMessage
  | SteerMessage
  | ApprovalResponseMessage
  | ContinueResponseMessage
  | UserQuestionResponseMessage
  | CommandMessage
  | CreateSessionMessage
  | SwitchSessionMessage
  | DeleteSessionMessage
  | RenameSessionMessage
  | BatchReviewAcceptMessage
  | BatchReviewRevertMessage
  | BatchReviewAcceptFileMessage
  | BatchReviewRevertFileMessage
  | ListTogglesMessage
  | ListResourcesMessage
  | ToggleServerMessage
  | ToggleSkillMessage;

// ── UI state types ─────────────────────────────────────────────────────────────

export type MessageRole = "user" | "agent";
export type MessageKind = "text" | "approval" | "error" | "streaming" | "editing" | "thinking" | "tools";

/** A single tool invocation tracked from start (tool_call) to finish (tool_result). */
export interface ToolActivity {
  id: string;
  name: string;
  icon: string;
  label: string;
  detail: string;
  status: "running" | "ok" | "error";
  summary?: string;
  /** Full error text of a failed call, shown in the expandable panel under the row. */
  error?: string;
  /** Terminal in/out panel data, revealed when the row is expanded (exec tools only). */
  exec?: ExecResult;
  durationMs?: number;
  /** What the model said this run's output showed — set when a verdict settled it. */
  verdict?: "pass" | "fail" | "unknown";
  /** Epoch ms when the call started — drives the live elapsed timer. */
  startedAt: number;
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
  streaming?: boolean;
  /** True while the agent is still editing (pulsing indicator shown). */
  live?: boolean;
  /** Structured tool activity for a frozen kind="tools" message. */
  tools?: ToolActivity[];
  /** A user steer message queued mid-run, awaiting injection (shows a "queued" tag
   *  until the server confirms with steer_injected). */
  queued?: boolean;
  /** Set on a bubble that the guardrail layer injected, not the user. Carries the
   *  nudge category so the badge can name what fired. */
  nudge?: string;
}

export type ConnectionState = "disconnected" | "connecting" | "connected" | "error";
