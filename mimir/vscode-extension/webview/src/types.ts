// ── Message types (server → client) ──────────────────────────────────────────

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

export interface OutputMessage {
  type: "output";
  text: string;
}

export interface StatusMessage {
  type: "status";
  text: string;
}

/** One row of a command answer: a name, optionally what it is. */
export interface CommandItem {
  label: string;
  detail?: string;
}

/**
 * How a command answer should read. "warn" is for something irreversible that has
 * just happened (a wipe), "empty" for a listing with nothing in it.
 */
/**
 * How a command's answer should read.
 *
 * `quiet` is not a weaker `ok`: it renders as a rule across the thread rather than as
 * a result, for the one thing that is neither a listing nor something the user just
 * did — a note about the state they have walked back into. A warning badge on that
 * reads as a problem to deal with, and it is only a fact.
 */
export type CommandTone = "ok" | "warn" | "empty" | "quiet";

/**
 * The answer to a session command the user typed ("/memory list", "/proxy clean x").
 *
 * Separate from OutputMessage because that one is transient tool-activity text the
 * reducer drops — which silently swallowed every command answer, including the
 * confirmation of an irreversible "/memory clear".
 *
 * Structured rather than a formatted line: a setting change and a listing of twenty
 * memories want different shapes on screen, and the frontend cannot lay out what
 * arrives as an already-indented blob.
 */
export interface CommandOutputMessage {
  type: "command_output";
  /** The command that produced this, e.g. "/memory list". */
  command: string;
  /** The headline result, e.g. "3 memories" or "Cleared 4 memories". */
  title: string;
  items?: CommandItem[];
  /** A single line under the rows, e.g. "This cannot be undone." */
  note?: string;
  tone?: CommandTone;
}

/** Which conversation a card belongs to.
 *
 *  Several conversations run turns at once, and one may raise a card while the user is
 *  reading another. The id routes the answer back to the agent that asked: the server
 *  drops an answer naming no conversation rather than handing it to whoever is on screen,
 *  which would attach the user's approval to a call they never saw. Optional, so a card
 *  from a server that does not attribute them still renders — it counts as the active
 *  conversation's. */
export interface Attributed {
  session_id?: string;
  session_title?: string;
}

export interface ApprovalMessage extends Attributed {
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
  /** Set when the call is held because it touches paths outside the workspace.
   *  Every outside path the call names travels in this one card — the user judges
   *  the call, not each of its operands. The rest of the card describes the tool
   *  call itself, as usual. */
  oow_paths?: string[];
  /** First of `oow_paths`, kept for a card built by an older server. */
  oow_path?: string;
}

/** Every agent slot is taken; this conversation's turn starts when one frees.
 *
 *  Rendered rather than dropped like other transient status: a queue nobody can see
 *  reads as a hang, which is the one thing a bounded pool must never look like. */
export interface QueuedMessage extends Attributed {
  type: "queued";
  position: number;
  text: string;
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

/**
 * Whose server this window is talking to. Emitted by the extension host, not the
 * server: only the host knows whether it spawned the process it connected to.
 *
 * It decides what disconnecting means. The host kills a server it started and leaves
 * alone one it only attached to — another window of the same workspace may be reading
 * it, and a detached one was kept running on purpose — so the same click ends the runs
 * in one case and leaves them going in the other.
 */
export interface ServerOwnershipMessage {
  type: "server_ownership";
  /** True when closing this window (or disconnecting) stops the server. */
  ours: boolean;
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
  /** Whether this row may be detached to the background while it runs. Decided by
   *  the server from the tool registry — the UI never learns which tool is a shell. */
  divertible?: boolean;
  /** The IN half of the terminal panel, for a run whose output is still to come:
   *  the command, with empty streams. Present only for exec-shaped calls (the
   *  server decides from the registry). Replaced by the full panel on tool_result. */
  exec?: ExecResult;
}

/** Terminal in/out of an exec-shaped tool result (shell / code runner / compiler).
 *  Sent for results carrying stdout/stderr plus either a returncode or, for a run
 *  the user detached mid-flight, a background job handle. */
export interface ExecResult {
  /** Full command or code body that was run (clipped server-side). */
  command?: string;
  stdout: string;
  stderr: string;
  /** Absent on a detached run: it has not produced an exit status yet. */
  returncode?: number;
  cwd?: string;
  /** True when a stream was clipped (server or wire budget). */
  truncated?: boolean;
  /** The run was moved to the background and is still going; the streams above are
   *  what it had printed at that moment. */
  running?: boolean;
  /** Handle of the background job it continues as — present with `running`. */
  job_key?: string;
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
  /** The calculation typeset by the tool, present only for math-shaped results. */
  math?: MathResult;
  /** The file the call touched, present only when it succeeded on an existing file. */
  target?: FileTarget;
  duration_ms: number;
}

/** A file a tool call touched, and the lines it covered: what the row's file name
 *  opens when clicked. `name` is the text of the row that becomes the link. */
export interface FileTarget {
  path: string;
  name: string;
  line?: number;
  end_line?: number;
}

/** A calculation and its result as one display-math LaTeX body (no `$` delimiters). */
export interface MathResult {
  latex: string;
}

/** What a blocking run is doing, pushed while its tool call is still open.
 *
 *  A run that holds the turn for twenty minutes cannot report on itself: the call
 *  does not answer until it is over. The server publishes its phase on a side channel
 *  and the client relays it here, so the row shows the work instead of a mute spinner.
 *  Transient by nature — never persisted into a session's display messages. */
export interface ToolProgressMessage {
  type: "tool_progress";
  /** Correlation id matching an earlier tool_call. */
  id: string;
  /** Human phase text, authored server-side (e.g. "building solver (1/2)"). The UI
   *  renders it without interpreting it, the way it does a tool's label template. */
  phase?: string;
  /** 0-100, present only while a phase actually counts itself — a compiler's own
   *  progress. Absent means "this phase does not say", never "0". */
  percent?: number;
}

/** A tool call whose work outlived it: a watcher now holds the run.
 *
 *  Sent only once a watcher has actually taken the job, so a row never claims to be
 *  tracked by something that declined it. It is what ties a settled row to a run
 *  still going — `exec` cannot, since a result with no output (an optimization run)
 *  has no terminal panel to hang a job key on. */
export interface ToolBackgroundedMessage {
  type: "tool_backgrounded";
  /** Correlation id matching an earlier tool_call. */
  id: string;
  /** The watcher's handle on the run, matched by later job events. */
  job_key: string;
}

/** What a detached run is doing, from the watcher that polls it.
 *
 *  The counterpart of `tool_progress`: while the call blocks, the server publishes on
 *  its run channel; once detached, the watcher's own poll is the only thing still
 *  asking. Keyed by job, because by now the row that launched it has settled. */
export interface JobProgressMessage {
  type: "job_progress";
  job_key: string;
  phase?: string;
  percent?: number;
}

/** One step of a sub-agent's run, reported while its parent tool call is still open.
 *  A delegated run lasts minutes inside a single call; these are what make it visible,
 *  and they render as ordinary tool rows under the row that spawned them. */
export interface SubAgentEventMessage {
  type: "subagent_event";
  /** The tool_call id of the delegating call these belong to. */
  parent_id: string;
  kind: "tool_call" | "tool_result" | "end" | "heartbeat";
  /** Child row id, already namespaced under parent_id (siblings reuse "c1"). */
  id?: string;
  name?: string;
  label?: string;
  detail?: string;
  ok?: boolean;
  summary?: string;
  duration_ms?: number;
  /** On "tool_result": the file the step touched, as on a tool_result. */
  target?: FileTarget | null;
  /** On "end": how many of the child's events were shed rather than forwarded. */
  dropped?: number;
  /** On "heartbeat": seconds the child has been running with nothing to report. */
  waiting_secs?: number;
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


/** Confirms a mid-run steer message reached the agent: injected into the running
 *  turn, or — when `starts_turn` is set — carried by a turn the server started for
 *  it because the run it was typed into ended before reading it. */
export interface SteerInjectedMessage {
  type: "steer_injected";
  text: string;
  starts_turn?: boolean;
}

/** A workflow reminder the guardrail layer injected into the agent's user turn.
 *  Signals that the turn the model just streamed was not accepted as its answer.
 *  Not rendered — the reducer uses it to drop the superseded draft. */
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
  /** A turn of this conversation is in flight. Several run at once, so a conversation the
   *  user is not reading can be working — invisible without this. */
  running?: boolean;
  /** Its turn is parked on a card. Worse than invisible: the wait has no timeout, so
   *  the conversation stays stopped until somebody answers. */
  parked?: boolean;
  /** It asked for a turn but every agent slot is taken; it starts when one frees. */
  queued?: boolean;
  /** How many background runs it still has going.
   *
   *  Not the same question as `running`: a conversation that launched a two-hour build
   *  and answered has no turn in flight, and is exactly the one worth asking about
   *  before walking away. */
  runs?: number;
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
  /** A turn of this session is still in flight — it outlived the last connection. */
  turn_running?: boolean;
}

/** A whole streamed prose block, recorded rather than streamed.
 *
 *  Never sent live: a connected client has already had every one of its deltas as
 *  `token`, and the aggregate on top would print the paragraph twice. It exists for
 *  the replay, where the deltas are gone — the journal deliberately keeps hundreds of
 *  `token` events out of itself, and without this a turn read back after an absence
 *  would have only its tool rows and its final answer, having lost everything the
 *  agent said in between.
 *
 *  Fed to the reducer as a `token`, which is what keeps a replayed turn interleaving
 *  prose and tool cards the way the live one did: the same draft-and-boundary logic
 *  decides where the block sits relative to the calls that followed it. */
export interface AssistantTextMessage {
  type: "assistant_text";
  text: string;
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

/** What this conversation produced while nobody was attached.
 *
 *  Read back from the session's journal past the watermark the client last reported,
 *  and sent on attach in frames of at most a few hundred. Each event is the same shape
 *  a live one has, so feed them to the same reducer: that is what rebuilds the rich
 *  transcript — tool rows, diff cards, reasoning panels — for a period this window was
 *  not there for.
 *
 *  Streamed deltas are deliberately absent. `token` and `thinking` are not journaled,
 *  so a replayed turn arrives as the aggregates that closed its blocks (`thinking_end`,
 *  `answer`) rather than its keystrokes — which is also what makes replaying a
 *  twelve-hour run affordable. `context_usage`, `batch_status` and `todo` are re-derived
 *  on load for the same reason.
 *
 *  `through_seq` is the watermark to send back on the next `transcript`; `more` says
 *  another frame follows; `truncated` says older events were elided because the replay
 *  hit its cap. */
export interface ReplayMessage {
  type: "replay";
  session_id: string;
  events: ServerMessage[];
  through_seq: number;
  more: boolean;
  truncated: boolean;
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

export interface JobCompleteMessage {
  type: "job_complete";
  job_key: string;
  server?: string;
  kind?: string;
  state: string;
  summary?: Record<string, unknown>;
  /** True when this wake auto-resumes the session on screen, so a turn is starting
   *  here. False when it resumes another conversation, which must leave this chat
   *  idle. Decided by the server: the client cannot tell whose turn it is. */
  resumes_active_session?: boolean;
}

/** A periodic bulletin on runs still going, 30s / 2min / 10min after the first
 *  launch. Unlike a `job_complete` it carries no result and settles no row: it only
 *  says the runs are still there and what they were last seen doing. It may start a
 *  short turn in the conversation that launched them (never a steer into one already
 *  running — a bulletin is not worth derailing an answer the user is waiting on). */
export interface JobCheckinMessage {
  type: "job_checkin";
  jobs: Array<{
    job_key: string;
    kind?: string;
    server?: string;
    state?: string;
    phase?: string;
    percent?: number | null;
  }>;
  /** Same meaning as on `job_complete`: a turn is starting in THIS conversation, so
   *  the chat must mark itself busy for something nobody pressed send for. */
  resumes_active_session?: boolean;
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

export interface UserQuestionMessage extends Attributed {
  type: "user_question";
  id: string;
  questions: QuestionSpec[];
  /** Seconds the agent waits for an answer before closing the card and going with
   *  the first option. Absent on a card with no wall (plan approval). */
  timeout_secs?: number;
}

/** The wait behind a card ended with nothing answered: close it, here or in the
 *  foreign-prompt strip. The turn is producing again — it went on with what it
 *  recommended — so it is not an answer and nothing is sent back. */
export interface PromptExpiredMessage extends Attributed {
  type: "prompt_expired";
  id: string;
  kind: "user_question";
  timeout_secs?: number;
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
  /** True once `overhead_tokens` is the size the server reported for a real prompt,
   *  false while it is still this client's estimate of the prompt it will send. */
  overhead_measured?: boolean;
  /** True while the conversation has no agent yet: the system prompt and tools schema
   *  are not in `used_tokens` at all, so the figure is a floor and the percentage must
   *  not be read as a verdict. */
  provisional?: boolean;
  /** Messages in the window the model actually sees this turn. */
  history_messages?: number;
  /** Messages in the untrimmed record a resume would start from. Larger than
   *  `history_messages` once the budget has trimmed the window. */
  history_messages_full?: number;
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
  | CommandOutputMessage
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
  | ToolProgressMessage
  | ToolBackgroundedMessage
  | JobProgressMessage
  | VerdictMessage
  | SubAgentEventMessage
  | ErrorMessage
  | ConfigMessage
  | AutoConnectMessage
  | ModelsMessage
  | ServedModelsMessage
  | ModelChangedMessage
  | DiffMessage
  | SteerInjectedMessage
  | NudgeInjectedMessage
  | FileProgressMessage
  | BatchStatusMessage
  | SessionsListMessage
  | SessionLoadedMessage
  | ContextModeMessage
  | EnforcementModeMessage
  | ApprovalModeMessage
  | AgentModeMessage
  | ServerOwnershipMessage
  | ThinkingDepthMessage
  | TemperatureMessage
  | StreamingStateMessage
  | ContextUsageMessage
  | JobCompleteMessage
  | JobCheckinMessage
  | TogglesListMessage
  | ResourcesMessage
  | ActiveEditorMessage
  | OpenEditorMessage
  | UserQuestionMessage
  | PromptExpiredMessage
  | ReplayMessage
  | AssistantTextMessage
  | DetachedMessage
  | QueuedMessage;

// ── Message types (client → server) ──────────────────────────────────────────

export interface QueryMessage {
  type: "query";
  text: string;
}

/** The rendered chat, handed to the server so a reload can show it again.
 *
 *  Tool rows, reasoning panels and diff cards are assembled here and nowhere else,
 *  so the server cannot rebuild them — it stores this copy verbatim. `session_id`
 *  is what stops a transcript still in flight from landing on a session the user
 *  has since switched to. */
export interface TranscriptMessage {
  type: "transcript";
  session_id: string;
  messages: ChatMessage[];
  /** How far into the session's journal these messages reach — the highest `seq` the
   *  reducer has taken in, from a live event or a `replay` frame. Stored server-side
   *  and used as the point a later re-attach replays *from*, so a window that has
   *  rendered everything is sent nothing. Omitted, the watermark simply does not
   *  advance and the next attach replays more than it needed to. */
  through_seq?: number;
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
 * Stop the server, now or once it has nothing left to do.
 *
 * Sent rather than inferred from closing the socket, because a server this window did
 * not start is not killed by the host: it has no process to signal, and ending a
 * workspace's server is a decision in any case. `force` stops it whatever is in
 * flight — the user's own call about their own machine, and the only way to end a run
 * waiting on an answer they have decided not to give. Without it the request is
 * refused while anything is still working, and the reasons come back.
 */
export interface ShutdownMessage {
  type: "shutdown";
  force?: boolean;
}

/** A message typed while the agent is busy — injected into the running turn. */
export interface SteerMessage {
  type: "steer";
  text: string;
}

/** Detach the shell run currently blocking the turn, keeping what it has done. */
export interface DivertToBackgroundMessage {
  type: "divert_to_background";
  /** The running tool row the user acted on. */
  id: string;
}

export interface ApprovalResponseMessage {
  type: "approval_response";
  id: string;
  /** The conversation that asked. Required in practice: the server drops an answer
   *  carrying none rather than guessing which agent it belongs to. */
  session_id?: string;
  choice: "y" | "n" | "a";
  approved_files?: string[];
}

export interface UserQuestionResponseMessage {
  type: "user_question_response";
  id: string;
  session_id?: string;
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

/** Switch the served model mid-session (no reconnect). */
export interface SetModelMessage {
  type: "set_model";
  model: string;
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
  | TranscriptMessage
  | SteerMessage
  | DetachMessage
  | DivertToBackgroundMessage
  | ApprovalResponseMessage
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
  | ToggleSkillMessage
  | SetModelMessage
  | ShutdownMessage;

// ── UI state types ─────────────────────────────────────────────────────────────

export type MessageRole = "user" | "agent";
export type MessageKind =
  | "text" | "approval" | "error" | "editing" | "thinking" | "tools" | "command";

/** A single tool invocation tracked from start (tool_call) to finish (tool_result). */
export interface ToolActivity {
  id: string;
  name: string;
  icon: string;
  label: string;
  detail: string;
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

export type ConnectionState = "disconnected" | "connecting" | "connected" | "error";
