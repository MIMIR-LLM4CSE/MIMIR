// Detached runs: their progress, the wake when one ends, and the bulletin while they run.

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

/** What launched a run: this conversation, or one of its sub-agents. */
export interface RunOrigin {
  kind?: "main" | "subagent" | string;
  /** The child's task, first line — empty for the conversation's own run. */
  label?: string;
  session?: string;
}

/** A detached run the watcher is holding: a build, a Slurm job, a sub-agent. */
export interface WatchedRun {
  job_key: string;
  kind?: string;
  server?: string;
  origin?: RunOrigin;
  /** What the watcher's last poll learned. Whatever the status op reported. */
  state?: string;
  phase?: string;
  percent?: number;
  scheduler?: string;
  nodes?: string[];
  pid?: number;
  job_id?: string;
  command?: string;
  elapsed_s?: number;
  /** Its server named a call that stops it, so the panel may offer to. */
  cancellable?: boolean;
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
