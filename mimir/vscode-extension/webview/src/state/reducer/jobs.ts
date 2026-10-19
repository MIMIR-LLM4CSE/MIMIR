// Detached runs: the wake when one ends, the bulletin while they run, and their progress.

import type { ToolActivity } from "../../types";
import { type ChatHandlers } from "../chatState";
import { patchToolsEverywhere } from "./toolRows";

export const JOB_HANDLERS: ChatHandlers = {
  // A finished background run auto-resumes the session that launched it. No one
  // pressed send, so `busy` — which only `submit_query` sets — would stay false
  // for the whole turn: the composer keeps offering "send" and the stop button
  // never appears, leaving a running agent that cannot be interrupted.
  //
  // Only when the wake resumes THIS conversation. A wake for a detached session
  // runs a turn elsewhere, and marking this chat busy for it would show a stop
  // button that stops nothing.
  job_complete: (state, action) => {
    // Settle the row still waiting on this run: the one the launch left
    // `background` (known by the job key on its terminal panel or on the row) and any
    // row that adopted the key. Settled rows, such as a finished status probe that
    // carries the key, are left alone.
    const isThisJob = (t: ToolActivity) =>
      (t.status === "background" || t.status === "running") &&
      (t.exec?.job_key === action.job_key || t.jobKey === action.job_key);
    const settled = patchToolsEverywhere(
      state,
      isThisJob,
      (tools) =>
        tools.map((t) =>
          isThisJob(t)
            ? {
                ...t,
                status:
                  action.state === "done"
                    ? ("ok" as const)
                    : ("error" as const),
                summary: `background run ${action.state}`,
                // The run is over: whatever it was last seen doing is no longer
                // what it is doing, and nothing is watching it any more.
                phase: undefined,
                percent: undefined,
                jobKey: undefined,
                // Only a row that had a terminal panel gets one back. A result
                // that never produced output has no pane to fill, and inventing
                // one would show an empty terminal for a run that printed
                // nothing to it.
                exec: t.exec
                  ? {
                      ...t.exec,
                      running: undefined,
                      returncode:
                        typeof action.summary?.returncode === "number"
                          ? (action.summary.returncode as number)
                          : t.exec.returncode,
                      // The job's own log, which already carries stderr under its
                      // marker — so the partial stderr is replaced, not doubled.
                      stdout:
                        typeof action.summary?.output === "string"
                          ? (action.summary.output as string)
                          : t.exec.stdout,
                      stderr:
                        typeof action.summary?.output === "string"
                          ? ""
                          : t.exec.stderr,
                    }
                  : undefined,
              }
            : t
        )
    );
    const base = settled ?? state;
    if (!action.resumes_active_session) return base;
    return {
      ...base,
      busy: true,
      liveThinkingBlocks: [],
      liveToolCalls: [],
      toolCallAfterToken: true,
    };
  },

  // A bulletin on runs that are still going. It settles nothing — the rows it
  // describes are still running, which is the whole news — so unlike a wake it
  // touches no tool call. All it can do is mark the chat busy for the short turn
  // it may have started, for the same reason as above: nobody pressed send, and
  // without this there is no stop button for a turn that is running.
  job_checkin: (state, action) => {
    if (!action.resumes_active_session) return state;
    return {
      ...state,
      busy: true,
      liveThinkingBlocks: [],
      liveToolCalls: [],
      toolCallAfterToken: true,
    };
  },

  job_progress: (state, action) => {
    // Same fact as tool_progress, from the other side of the hand-off: by now the
    // row has settled, so the job key is what finds it.
    const { job_key: jKey, phase: jPhase, percent: jPercent } = action;
    // Only a row still waiting on the run, for the reason tool_progress checks the
    // same thing: a settled row has an outcome, and a bar on it would describe work
    // that is over.
    const waitingOn = (t: ToolActivity) =>
      (t.status === "background" || t.status === "running") &&
      (t.jobKey === jKey || t.exec?.job_key === jKey);
    return (
      patchToolsEverywhere(
        state,
        waitingOn,
        (tools) =>
          tools.map((t) =>
            waitingOn(t)
              ? { ...t, phase: jPhase,
                  percent: typeof jPercent === "number" ? jPercent : undefined }
              : t
          )
      ) ?? state
    );
  },
};
