// Tool rows: a call, its result and progress, a sub-agent's steps, the verdict on a run,
// and the files it changed.

import type { DiffEntry } from "./chat";

export interface ToolCallMessage {
  type: "tool_call";
  /** Correlation id matching a later tool_result. */
  id: string;
  /** Raw tool name. Never shown — the row is drawn from `kind` and `doing`. */
  name: string;
  /** Human-readable label, e.g. "Reading file: x.py". The row's tooltip; approval
   *  cards and policy messages carry the same string. */
  label: string;
  /** Short arg preview, e.g. a command line or search pattern (may be empty). */
  detail: string;
  /** The tool's family of work — "edit", "shell", "proxy eval" (see TOOL_KINDS
   *  server-side). The word the row shows, and what its icon is picked from.
   *  Absent on rows from a session recorded before the row carried one. */
  kind?: string;
  /** The model's own sentence about THIS call, under ~15 words: "fixing the
   *  off-by-one bound". Empty whenever the model did not write one. */
  doing?: string;
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
  /** On "tool_call": the child step's family of work. Named `tool_kind` because this
   *  event's own `kind` already says which half of the child's activity it carries. */
  tool_kind?: string;
  /** On "tool_call": the child model's sentence about its own call. */
  doing?: string;
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

export interface DiffMessage {
  type: "diff";
  file: string;
  patch: string;
}

export interface FileProgressMessage {
  type: "file_progress";
  diffs: DiffEntry[];
}

/** Detach the shell run currently blocking the turn, keeping what it has done. */
export interface DivertToBackgroundMessage {
  type: "divert_to_background";
  /** The running tool row the user acted on. */
  id: string;
}
