import { describe, it, expect } from "vitest";
import {
  createChatReducer,
  initialChatState,
  type ChatState,
} from "./chatReducer";

// Deterministic id source so assertions are stable.
function makeReducer() {
  let n = 0;
  const makeId = () => `id${n++}`;
  return createChatReducer(makeId);
}

function run(actions: Parameters<ReturnType<typeof makeReducer>>[1][]): ChatState {
  const reducer = makeReducer();
  return actions.reduce((s, a) => reducer(s, a), initialChatState);
}

describe("chatReducer", () => {
  it("does not answer a card twice when it is put back over its saved copy", () => {
    // Coming back to a session whose turn was set aside: the saved transcript still
    // holds the card, and the server sends it again with the same id.
    const card = { type: "approval" as const, id: "a1", tool: "bash_run", server: "bash", args: { command: "echo hi" }, risk: "", scope: "" };
    const state = run([card, card]);
    const cards = state.messages.filter((m) => m.kind === "approval");
    expect(cards).toHaveLength(1);
    expect(cards[0].approval?.ids).toEqual(["a1"]);
  });

  it("coalesces token text across an intervening status into one draft", () => {
    const state = run([
      { type: "token", text: "Hello" },
      { type: "status", text: "Running bash command" },
      { type: "token", text: " world" },
    ]);

    // Adjacent words join without an extra paragraph break, and nothing about the
    // turn has reached the transcript yet.
    expect(state.draft).toBe("Hello world");
    expect(state.messages).toHaveLength(0);
    // Status text is dropped (decorrelated); it only flags a step boundary, which
    // is consumed by the second token. No thinking/tools messages are created.
    expect(state.messages.filter((m) => m.kind === "thinking")).toHaveLength(0);
    expect(state.messages.filter((m) => m.kind === "tools")).toHaveLength(0);
    expect(state.toolCallAfterToken).toBe(false);
  });

  it("commits the draft above the cards of the step that follows it", () => {
    const state = run([
      { type: "token", text: "Let me look." },
      { type: "tool_call", id: "t1", name: "bash_run" },
      { type: "tool_result", id: "t1", ok: true, summary: "done" },
      { type: "token", text: "Found it." },
    ]);

    expect(state.messages.map((m) => m.kind)).toEqual(["text", "tools"]);
    expect(state.messages[0].text).toBe("Let me look.");
    expect(state.draft).toBe("Found it.");
  });

  it("freezes reasoning that arrived after the prose below it, not above it", () => {
    // The live view and the freeze both read the arrival stamps, so a block the
    // user watched appear under a paragraph stays under it. Freezing by kind used
    // to lift the prose above everything, which moved the card at step end.
    const state = run([
      { type: "token", text: "Here is one way." },
      { type: "thinking", text: "though maybe not" },
      { type: "error", text: "boom" },
    ]);

    expect(state.messages.map((m) => m.kind)).toEqual(["text", "thinking", "error"]);
  });

  it("freezes prose that arrived after a reasoning block below it", () => {
    // Reasoning chunks with no thinking_start do not flag a step boundary, so
    // the prose that follows them lands in the same freeze — after them.
    const state = run([
      { type: "thinking", text: "let me think" },
      { type: "token", text: "Done." },
      { type: "error", text: "boom" },
    ]);

    expect(state.messages.map((m) => m.kind)).toEqual(["thinking", "text", "error"]);
  });

  it("re-stamps the draft after a freeze, so the next step orders on its own", () => {
    const state = run([
      { type: "token", text: "First." },
      { type: "tool_call", id: "t1", name: "bash_run" },
      { type: "token", text: "Second." },
      { type: "tool_call", id: "t2", name: "bash_run" },
    ]);

    // "First." was committed above t1 when the second token opened a new step;
    // "Second." is still in flight and stamped before t2.
    expect(state.messages.map((m) => m.kind)).toEqual(["text", "tools"]);
    expect(state.draft).toBe("Second.");
    expect(state.draftSeq).not.toBeNull();
    expect(state.draftSeq!).toBeLessThan(state.liveToolCalls[0].seq!);
  });

  it("puts a steer bubble under the prose and cards it interrupted", () => {
    const state = run([
      { type: "submit_query", text: "go" },
      { type: "token", text: "Looking into it." },
      { type: "tool_call", id: "t1", name: "bash_run" },
      { type: "steer_query", text: "also check the tests" },
      { type: "error", text: "boom" },
    ]);

    // The bubble arrived after the prose and after the tool row, and stays under
    // both — including once the step is frozen, which is what used to lift the
    // prose back above it.
    expect(state.messages.map((m) => [m.role, m.kind])).toEqual([
      ["user", "text"],    // the query
      ["agent", "text"],   // "Looking into it."
      ["agent", "tools"],  // t1
      ["user", "text"],    // the steer
      ["agent", "error"],
    ]);
    expect(state.messages[3].text).toBe("also check the tests");
    expect(state.messages[3].queued).toBe(true);
  });

  it("keeps a steer bubble above the prose that started after it", () => {
    const state = run([
      { type: "submit_query", text: "go" },
      { type: "steer_query", text: "one more thing" },
      { type: "token", text: "Right, both then." },
      { type: "error", text: "boom" },
    ]);

    expect(state.messages.map((m) => m.text)).toEqual([
      "go", "one more thing", "Right, both then.", "boom",
    ]);
  });

  it("drops the draft when a guardrail nudge sends the model back to work", () => {
    // The symptom this exists for: a finished-looking answer appearing in the
    // transcript and then being taken out of it again.
    const state = run([
      { type: "submit_query", text: "fix it" },
      { type: "token", text: "All done!" },
      { type: "nudge_injected", category: "validation", text: "check it first" },
    ]);

    expect(state.draft).toBe("");
    expect(state.messages.map((m) => m.role)).toEqual(["user"]);
  });

  it("accumulates thinking chunks into one live block and pendingThinking", () => {
    const state = run([
      { type: "thinking_start" },
      { type: "thinking", text: "first" },
      { type: "thinking", text: "second" },
      { type: "thinking_end" },
    ]);

    expect(state.liveThinkingBlocks).toHaveLength(1);
    expect(state.liveThinkingBlocks[0].text).toContain("first");
    expect(state.liveThinkingBlocks[0].text).toContain("second");
    expect(state.pendingThinking).toBe("firstsecond");
    expect(state.toolCallAfterToken).toBe(true);
  });

  it("publishes the answer and clears the draft", () => {
    const state = run([
      { type: "submit_query", text: "hi" },
      { type: "token", text: "partial" },
      { type: "answer", text: "the final answer" },
    ]);

    expect(state.busy).toBe(false);
    expect(state.draft).toBe("");
    const last = state.messages[state.messages.length - 1];
    expect(last.kind).toBe("text");
    expect(last.text).toBe("the final answer");
    // The draft is superseded, not appended beside the answer.
    expect(state.messages.filter((m) => m.text === "partial")).toHaveLength(0);
    expect(state.pendingThinking).toBe("");
    expect(state.pendingDiffs).toHaveLength(0);
  });

  it("attaches accumulated thinking to the final answer", () => {
    const state = run([
      { type: "thinking_start" },
      { type: "thinking", text: "reasoning" },
      { type: "token", text: "body" },
      { type: "answer", text: "done" },
    ]);
    const answer = state.messages.find((m) => m.kind === "text" && m.text === "done");
    expect(answer?.thinking).toBe("reasoning");
  });

  it("creates a running tool activity in liveToolCalls on tool_call", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "grep", label: "Searching", detail: "def foo" },
    ]);
    // Tools live in their own stream — decorrelated from thinking.
    expect(state.liveThinkingBlocks).toHaveLength(0);
    expect(state.liveToolCalls).toHaveLength(1);
    expect(state.liveToolCalls[0]).toMatchObject({
      id: "c1", name: "grep", icon: "🔍", label: "Searching", detail: "def foo", status: "running",
    });
    expect(state.toolCallAfterToken).toBe(true);
  });

  it("marks the matching activity done with duration on tool_result", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "grep", label: "Searching", detail: "x" },
      { type: "tool_result", id: "c1", name: "grep", ok: true, summary: "3 matches", duration_ms: 42 },
    ]);
    const tool = state.liveToolCalls[0];
    expect(tool.status).toBe("ok");
    expect(tool.summary).toBe("3 matches");
    expect(tool.durationMs).toBe(42);
  });

  it("keeps a typeset calculation from tool_result on the activity", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "calc", label: "Evaluating", detail: "2 ** 10" },
      { type: "tool_result", id: "c1", name: "calc", ok: true, summary: "",
        math: { latex: "{2}^{10} = 1024" }, duration_ms: 3 },
    ]);
    expect(state.liveToolCalls[0].math).toEqual({ latex: "{2}^{10} = 1024" });
  });

  it("keeps the file target of a successful tool_result", () => {
    const target = { path: "/w/a.py", name: "a.py", line: 3, end_line: 9 };
    const state = run([
      { type: "tool_call", id: "c1", name: "read", label: "Reading file: a.py", detail: "" },
      { type: "tool_result", id: "c1", name: "read", ok: true, summary: "", target, duration_ms: 1 },
    ]);
    expect(state.liveToolCalls[0].target).toEqual(target);
  });

  it("drops the file target of a failed tool_result", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "edit", label: "Editing file: a.py", detail: "" },
      { type: "tool_result", id: "c1", name: "edit", ok: false, summary: "no match",
        target: { path: "/w/a.py", name: "a.py" }, duration_ms: 1 },
    ]);
    expect(state.liveToolCalls[0].target).toBeUndefined();
  });

  it("marks a failed tool_result as error", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "code_execute", label: "Running", detail: "make" },
      { type: "tool_result", id: "c1", name: "code_execute", ok: false, summary: "exit 1", duration_ms: 10 },
    ]);
    expect(state.liveToolCalls[0].status).toBe("error");
  });

  it("settles the activity with the summary and duration from tool_result", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "grep", label: "Searching", detail: "x" },
      { type: "tool_result", id: "c1", name: "grep", ok: true, summary: "3 matches", duration_ms: 5 },
    ]);
    expect(state.liveToolCalls[0].status).toBe("ok");
    expect(state.liveToolCalls[0].summary).toBe("3 matches");
    expect(state.liveToolCalls[0].durationMs).toBe(5);
  });

  it("opens the terminal panel on the call, before any output exists", () => {
    // Otherwise the command a run is executing is invisible for as long as it runs.
    const state = run([
      {
        type: "tool_call", id: "c1", name: "bash_run", label: "Running shell command",
        detail: "make -j8", exec: { command: "make -j8", stdout: "", stderr: "" },
      },
    ]);
    expect(state.liveToolCalls[0].status).toBe("running");
    expect(state.liveToolCalls[0].exec?.command).toBe("make -j8");
  });

  it("keeps the command on a result that carries no exec of its own", () => {
    // A failure sends no exec payload; erasing the IN pane would leave the row with
    // no trace of what was attempted.
    const state = run([
      {
        type: "tool_call", id: "c1", name: "bash_run", label: "Running shell command",
        detail: "make -j8", exec: { command: "make -j8", stdout: "", stderr: "" },
      },
      {
        type: "tool_result", id: "c1", name: "bash_run", ok: false,
        summary: "tool call failed", error: "server unreachable", duration_ms: 12,
      },
    ]);
    expect(state.liveToolCalls[0].status).toBe("error");
    expect(state.liveToolCalls[0].exec?.command).toBe("make -j8");
  });

  it("marks a run the user detached as background, not as finished", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "bash_run", label: "Running shell command", detail: "make -j8", divertible: true },
      {
        type: "tool_result", id: "c1", name: "bash_run", ok: true, summary: "exit 0",
        exec: { stdout: "configuring…\n", stderr: "", running: true, job_key: "J1" },
        duration_ms: 900,
      },
    ]);
    expect(state.liveToolCalls[0].status).toBe("background");
    expect(state.liveToolCalls[0].summary).toBe("moved to background");
  });

  it("settles the detached row when its job finally completes", () => {
    // The only message that ever says how the run actually ended: until it arrives
    // the row shows partial output and no exit code.
    const state = run([
      { type: "tool_call", id: "c1", name: "bash_run", label: "Running shell command", detail: "make -j8", divertible: true },
      {
        type: "tool_result", id: "c1", name: "bash_run", ok: true, summary: "exit 0",
        exec: { stdout: "configuring…\n", stderr: "", running: true, job_key: "J1" },
        duration_ms: 900,
      },
      {
        type: "job_complete", job_key: "J1", state: "done",
        summary: { returncode: 0, output: "configuring…\nbuilt\n" },
      },
    ]);
    const tool = state.liveToolCalls[0];
    expect(tool.status).toBe("ok");
    expect(tool.exec?.running).toBeUndefined();
    expect(tool.exec?.returncode).toBe(0);
    expect(tool.exec?.stdout).toContain("built");
  });

  it("badges the run's row when a verdict settles it", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "bash_run", label: "Running shell command", detail: "python solver.py" },
      { type: "tool_result", id: "c1", name: "bash_run", ok: true, summary: "exit 0", duration_ms: 8 },
      { type: "verdict", id: "c1", verdict: "fail" },
    ]);
    expect(state.liveToolCalls[0].verdict).toBe("fail");
  });

  it("badges a run frozen into an earlier step, since a verdict lands late by nature", () => {
    // The model runs, moves on, and only judges the output a step or two later —
    // by which time the row it belongs to is no longer live.
    const state = run([
      { type: "tool_call", id: "c1", name: "bash_run", label: "Running shell command", detail: "python solver.py" },
      { type: "tool_result", id: "c1", name: "bash_run", ok: true, summary: "exit 0", duration_ms: 8 },
      { type: "token", text: "reading the output" },
      { type: "verdict", id: "c1", verdict: "pass" },
    ]);
    const frozen = state.messages.find((m) => m.kind === "tools");
    expect(frozen?.tools![0].verdict).toBe("pass");
  });

  it("shows what a blocking run is doing while it is still running", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_progress", id: "c1", phase: "building solver (1/2)", percent: 34 },
    ]);
    expect(state.liveToolCalls[0].phase).toBe("building solver (1/2)");
    expect(state.liveToolCalls[0].percent).toBe(34);
  });

  it("drops a percentage the next phase no longer reports", () => {
    // Build ends, measurement begins. A bar left at the compiler's last number would
    // describe work that finished minutes ago.
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_progress", id: "c1", phase: "building solver", percent: 98 },
      { type: "tool_progress", id: "c1", phase: "case shock (1/3)" },
    ]);
    expect(state.liveToolCalls[0].phase).toBe("case shock (1/3)");
    expect(state.liveToolCalls[0].percent).toBeUndefined();
  });

  it("clears the phase when the row settles", () => {
    // The run's own ending is the last thing the row should say, not whatever it was
    // caught doing a moment before.
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_progress", id: "c1", phase: "building solver", percent: 60 },
      { type: "tool_result", id: "c1", name: "proxy_eval", ok: true, summary: "accepted", duration_ms: 12 },
    ]);
    expect(state.liveToolCalls[0].phase).toBeUndefined();
    expect(state.liveToolCalls[0].percent).toBeUndefined();
  });

  it("ignores progress for a row that has already settled", () => {
    // A tick can be in flight when the call answers. Reviving a finished row would
    // put a spinner's worth of state back on something with an outcome.
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_result", id: "c1", name: "proxy_eval", ok: true, summary: "accepted", duration_ms: 12 },
      { type: "tool_progress", id: "c1", phase: "building solver", percent: 60 },
    ]);
    expect(state.liveToolCalls[0].phase).toBeUndefined();
    expect(state.liveToolCalls[0].status).toBe("ok");
  });

  it("ignores progress for a row it has never heard of", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_progress", id: "nobody", phase: "building", percent: 10 },
    ]);
    expect(state.liveToolCalls[0].phase).toBeUndefined();
  });

  it("ties a settled row to the run that outlived it", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_result", id: "c1", name: "proxy_eval", ok: true, summary: "detached", duration_ms: 9 },
      { type: "tool_backgrounded", id: "c1", job_key: "J1" },
    ]);
    expect(state.liveToolCalls[0].jobKey).toBe("J1");
    expect(state.liveToolCalls[0].status).toBe("background");
  });

  it("keeps reporting a detached run through its job key", () => {
    // The row has settled by now, so the call id is no longer what finds it.
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_result", id: "c1", name: "proxy_eval", ok: true, summary: "detached", duration_ms: 9 },
      { type: "tool_backgrounded", id: "c1", job_key: "J1" },
      { type: "job_progress", job_key: "J1", phase: "case shock (2/3)", percent: 66 },
    ]);
    expect(state.liveToolCalls[0].phase).toBe("case shock (2/3)");
    expect(state.liveToolCalls[0].percent).toBe(66);
  });

  it("settles a detached row with no terminal panel without inventing one", () => {
    // An optimization run prints nothing to a terminal pane. Before the row carried
    // its own job key, its ending landed nowhere at all.
    const state = run([
      { type: "tool_call", id: "c1", name: "proxy_eval", label: "Proxy eval: run", detail: "" },
      { type: "tool_result", id: "c1", name: "proxy_eval", ok: true, summary: "detached", duration_ms: 9 },
      { type: "tool_backgrounded", id: "c1", job_key: "J1" },
      { type: "job_progress", job_key: "J1", phase: "building", percent: 20 },
      { type: "job_complete", job_key: "J1", state: "done" },
    ]);
    const row = state.liveToolCalls[0];
    expect(row.status).toBe("ok");
    expect(row.summary).toBe("background run done");
    expect(row.exec).toBeUndefined();
    // Nothing is watching it any more, so it has nothing left to report.
    expect(row.phase).toBeUndefined();
    expect(row.percent).toBeUndefined();
    expect(row.jobKey).toBeUndefined();
  });

  it("does not restore what a run was doing when the session is reloaded", () => {
    // The phase described a moment, and the watcher that could refresh it may have
    // died with the process that wrote the file.
    const state = run([
      {
        type: "session_loaded_messages",
        messages: [
          {
            id: "m1",
            role: "agent",
            kind: "tools",
            tools: [
              {
                id: "c1", name: "proxy_eval", icon: "x", label: "Proxy eval: run",
                detail: "", status: "background", startedAt: 0,
                jobKey: "J1", phase: "building", percent: 40,
              },
            ],
          },
        ],
      } as Parameters<ReturnType<typeof makeReducer>>[1],
    ]);
    const restored = state.messages[0].tools![0];
    expect(restored.phase).toBeUndefined();
    expect(restored.percent).toBeUndefined();
    // The job key survives: a watcher of this same client is still entitled to
    // report on it.
    expect(restored.jobKey).toBe("J1");
  });

  it("renders an activity for an unknown tool (icon fallback)", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "some_new_tool", label: "Performing", detail: "" },
    ]);
    expect(state.liveToolCalls[0].icon).toBe("🔧");
  });

  it("freezes tool activity into a tools message on the next LLM step", () => {
    // tool_call then a token (new LLM step) freezes the tools into messages,
    // as a dedicated kind:"tools" message — NOT inside a thinking message.
    const state = run([
      { type: "tool_call", id: "c1", name: "grep", label: "Searching", detail: "x" },
      { type: "tool_result", id: "c1", name: "grep", ok: true, summary: "1 match", duration_ms: 5 },
      { type: "token", text: "answer text" },
    ]);
    const frozen = state.messages.find((m) => m.kind === "tools");
    expect(frozen?.tools).toHaveLength(1);
    expect(frozen?.tools![0].status).toBe("ok");
    // Decorrelated: no thinking message was created, and the live stream cleared.
    expect(state.messages.some((m) => m.kind === "thinking")).toBe(false);
    expect(state.liveToolCalls).toHaveLength(0);
  });

  it("patches a tool frozen by an approval card when its result arrives after approval", () => {
    // A sensitive tool (e.g. bash_run) emits its approval card while still
    // running. The "approval" action runs freezePending, which bakes the
    // running tool into a kind:"tools" message and clears liveToolCalls. The
    // real tool_result — carrying the exec IN/OUT panel — lands only AFTER the
    // user approves, when the row is no longer live. It must still update the
    // frozen row rather than being dropped.
    const exec = { command: "echo hi", stdout: "hi\n", stderr: "", returncode: 0 };
    const state = run([
      { type: "tool_call", id: "c1", name: "bash_run", label: "Running shell command", detail: "echo hi" },
      { type: "approval", id: "a1", tool: "bash_run", server: "bash", args: { command: "echo hi" }, risk: "", scope: "" },
      { type: "tool_result", id: "c1", name: "bash_run", ok: true, summary: "exit 0", exec, duration_ms: 12 },
    ]);
    const frozen = state.messages.find((m) => m.kind === "tools");
    expect(frozen?.tools).toHaveLength(1);
    expect(frozen?.tools![0].status).toBe("ok");
    expect(frozen?.tools![0].exec).toEqual(exec);
    expect(frozen?.tools![0].durationMs).toBe(12);
    // The approval card is still present as its own message.
    expect(state.messages.some((m) => m.kind === "approval")).toBe(true);
  });

  it("freezes a thinking block into a kind:'thinking' message with a duration", () => {
    // thinking then a token (new LLM step) freezes reasoning into its own
    // kind:"thinking" message carrying a measured duration.
    const state = run([
      { type: "thinking_start" },
      { type: "thinking", text: "reasoning here" },
      { type: "token", text: "answer text" },
    ]);
    const frozen = state.messages.find((m) => m.kind === "thinking");
    expect(frozen?.text).toContain("reasoning here");
    expect(typeof frozen?.thinkingDurationMs).toBe("number");
    // Live block cleared once frozen.
    expect(state.liveThinkingBlocks).toHaveLength(0);
  });

  it("blocks a denied bash approval and clears it on accept", () => {
    const reducer = makeReducer();
    let s = reducer(initialChatState, {
      type: "approval",
      id: "a1",
      tool: "run_bash",
      server: "shell",
      args: { command: "rm -rf /" },
      risk: "high",
      scope: "once",
    });
    expect(s.messages.filter((m) => m.kind === "approval")).toHaveLength(1);

    // Deny → card becomes a blocked text marker quoting the command.
    s = reducer(s, { type: "approval_response", id: "a1", choice: "n" });
    const blocked = s.messages.find((m) => m.kind === "text");
    expect(blocked?.text).toContain("Blocked");
    expect(blocked?.text).toContain("rm -rf /");
    expect(s.busy).toBe(true);
  });

  it("coalesces repeated diffs of one file into a single entry and keeps is_new", () => {
    const s = run([
      // Creation: the emitter marks a new file with a /dev/null header.
      { type: "diff", file: "solver.py", patch: "--- /dev/null\n+++ b/solver.py\n+a" },
      // Two later edits of the same file: each patch supersedes the previous one.
      { type: "diff", file: "solver.py", patch: "--- a/solver.py\n+++ b/solver.py\n-a\n+b" },
      { type: "diff", file: "solver.py", patch: "--- a/solver.py\n+++ b/solver.py\n-b\n+c" },
    ]);
    const card = s.messages.find((m) => m.kind === "editing");
    expect(card?.diffs).toHaveLength(1);
    expect(card?.diffs?.[0].is_new).toBe(true);
    expect(card?.diffs?.[0].patch).toContain("+c");
    expect(s.pendingDiffs).toHaveLength(1);
  });

  it("keeps distinct files as distinct entries", () => {
    const s = run([
      { type: "diff", file: "solver.py", patch: "--- a/solver.py\n+++ b/solver.py\n+a" },
      { type: "diff", file: "driver.py", patch: "--- /dev/null\n+++ b/driver.py\n+b" },
    ]);
    const card = s.messages.find((m) => m.kind === "editing");
    expect(card?.diffs?.map((d) => d.file)).toEqual(["solver.py", "driver.py"]);
  });

  // ── Sub-agent activity ──────────────────────────────────────────────────────
  // A delegated run is one tool call lasting minutes. Its steps show up as ordinary
  // rows under the call that spawned them; what is tested here is that they land on
  // the right parent, in the right place, and settle when the parent does.

  it("shows a sub-agent's steps as rows under the call that spawned them", () => {
    const state = run([
      { type: "tool_call", id: "p1", name: "spawn_agent", label: "Sub-agent (explore): find X" },
      { type: "subagent_event", kind: "tool_call", parent_id: "p1", id: "p1:c1", name: "grep", label: "Searching: X" },
    ]);

    expect(state.liveToolCalls.map((t) => t.id)).toEqual(["p1", "p1:c1"]);
    const child = state.liveToolCalls[1];
    expect(child.parentId).toBe("p1");
    expect(child.status).toBe("running");
    // The badge says whose work it is; the role is read off the parent's own label.
    expect(child.origin).toBe("explore #1");
  });

  it("keeps concurrent sub-agents' rows grouped under their own parent", () => {
    const state = run([
      { type: "tool_call", id: "p1", name: "spawn_agent", label: "Sub-agent (explore): A" },
      { type: "tool_call", id: "p2", name: "spawn_agent", label: "Sub-agent (explore): B" },
      { type: "subagent_event", kind: "tool_call", parent_id: "p2", id: "p2:c1", name: "grep" },
      { type: "subagent_event", kind: "tool_call", parent_id: "p1", id: "p1:c1", name: "grep" },
      { type: "subagent_event", kind: "tool_call", parent_id: "p1", id: "p1:c2", name: "read_file" },
    ]);

    // Interleaved on the wire, grouped on screen — otherwise a fan-out is unreadable.
    expect(state.liveToolCalls.map((t) => t.id)).toEqual(["p1", "p1:c1", "p1:c2", "p2", "p2:c1"]);
    expect(state.liveToolCalls.find((t) => t.id === "p2:c1")!.origin).toBe("explore #2");
  });

  it("settles a child row on its own result", () => {
    const state = run([
      { type: "tool_call", id: "p1", name: "spawn_agent", label: "Sub-agent (explore): A" },
      { type: "subagent_event", kind: "tool_call", parent_id: "p1", id: "p1:c1", name: "grep" },
      { type: "subagent_event", kind: "tool_result", parent_id: "p1", id: "p1:c1", ok: true, summary: "3 matches", duration_ms: 41 },
    ]);

    const child = state.liveToolCalls[1];
    expect(child.status).toBe("ok");
    expect(child.summary).toBe("3 matches");
    expect(child.durationMs).toBe(41);
  });

  it("stops a child's timer when the delegating call itself finishes", () => {
    // A child whose last event was shed (or whose run was cut short by the cap)
    // would otherwise tick up forever under a call that is already done.
    const state = run([
      { type: "tool_call", id: "p1", name: "spawn_agent", label: "Sub-agent (explore): A" },
      { type: "subagent_event", kind: "tool_call", parent_id: "p1", id: "p1:c1", name: "grep" },
      { type: "tool_result", id: "p1", ok: true, summary: "answered" },
    ]);

    expect(state.liveToolCalls.map((t) => t.status)).toEqual(["ok", "ok"]);
  });

  it("patches a child row that was already frozen into the transcript", () => {
    // Showing an approval card mid-flight freezes the step; the child's result
    // arrives after that and must still find its row.
    const reducer = makeReducer();
    let state = initialChatState;
    for (const a of [
      { type: "tool_call" as const, id: "p1", name: "spawn_agent", label: "Sub-agent (explore): A" },
      { type: "subagent_event" as const, kind: "tool_call" as const, parent_id: "p1", id: "p1:c1", name: "grep" },
      { type: "token" as const, text: "next step" },   // commits the step, freezing the rows
    ]) state = reducer(state, a);

    state = reducer(state, {
      type: "subagent_event", kind: "tool_result", parent_id: "p1", id: "p1:c1",
      ok: false, summary: "no such file",
    });

    const frozen = state.messages.find((m) => m.kind === "tools")!;
    expect(frozen.tools!.find((t) => t.id === "p1:c1")!.status).toBe("error");
  });

  it("ignores activity for a parent that is nowhere on screen", () => {
    const before = run([{ type: "tool_call", id: "p1", name: "spawn_agent" }]);
    const reducer = makeReducer();
    const after = reducer(before, {
      type: "subagent_event", kind: "tool_call", parent_id: "gone", id: "gone:c1", name: "grep",
    });
    expect(after).toBe(before);
  });

  it("records how much of a sub-agent's activity was shed", () => {
    const state = run([
      { type: "tool_call", id: "p1", name: "spawn_agent", label: "Sub-agent (explore): A" },
      { type: "subagent_event", kind: "end", parent_id: "p1", dropped: 12 },
    ]);
    expect(state.liveToolCalls[0].childrenDropped).toBe(12);
  });
});

describe("session command replies", () => {
  const listing = {
    type: "command_output" as const,
    command: "/memory list",
    title: "3 memories",
    items: [
      { label: "a", detail: "first" },
      { label: "b", detail: "" },
      { label: "c", detail: "third" },
    ],
  };

  it("renders a command answer, unlike transient output", () => {
    const state = run([
      { type: "output", text: "  3 memory item(s): a, b, c\n" },
      listing,
    ]);

    // Transient tool chatter is still dropped; the command answer lands.
    expect(state.messages).toHaveLength(1);
    expect(state.messages[0].kind).toBe("command");
  });

  it("carries the answer whole so the card can lay it out", () => {
    // Flattening it to a line here is what made every answer a formatted blob the
    // frontend could only print.
    const state = run([listing]);
    const result = state.messages[0].command!;

    expect(result.command).toBe("/memory list");
    expect(result.title).toBe("3 memories");
    expect(result.items).toHaveLength(3);
    expect(result.items![0]).toEqual({ label: "a", detail: "first" });
  });

  it("keeps the confirmation of an irreversible clear, and marks it", () => {
    // The reason this channel exists: "/memory clear" wiped the store and said
    // nothing, so it read as a command that had not worked.
    const state = run([
      {
        type: "command_output",
        command: "/memory clear",
        title: "Cleared 4 memories",
        note: "This cannot be undone.",
        tone: "warn",
      },
    ]);
    const result = state.messages[0].command!;

    expect(result.title).toBe("Cleared 4 memories");
    expect(result.note).toBe("This cannot be undone.");
    expect(result.tone).toBe("warn");
  });

  it("defaults the tone and drops nameless rows", () => {
    const state = run([
      {
        type: "command_output",
        command: "/mode",
        title: "Mode",
        items: [{ label: "agent" }, { label: "  " }],
      },
    ]);
    const result = state.messages[0].command!;

    expect(result.tone).toBe("ok");
    expect(result.items).toEqual([{ label: "agent" }]);
  });

  it("commits streamed prose above the answer instead of losing it", () => {
    const state = run([
      { type: "token", text: "Working on it." },
      { type: "command_output", command: "/mode", title: "Mode", items: [{ label: "agent" }] },
    ]);

    expect(state.draft).toBe("");
    expect(state.messages.map((m) => m.kind)).toEqual(["text", "command"]);
    expect(state.messages[0].text).toBe("Working on it.");
  });

  it("does not end the turn — a command runs beside a run, not as one", () => {
    const state = run([
      { type: "submit_query", text: "hi" },
      { type: "command_output", command: "/batch", title: "Batch review", items: [{ label: "on" }] },
    ]);

    expect(state.busy).toBe(true);
  });

  it("ignores an answer with no title", () => {
    const state = run([{ type: "command_output", command: "/mode", title: "   " }]);
    expect(state.messages).toHaveLength(0);
  });
});

describe("background-job wake", () => {
  // A wake starts a turn nobody pressed send for. `busy` is what puts the composer
  // in stop mode, and only `submit_query` sets it — so before this the agent ran on
  // with the button still offering "send", and the user could not interrupt it.
  it("marks the turn busy when the wake resumes this conversation", () => {
    const state = run([
      {
        type: "job_complete",
        job_key: "j1",
        state: "done",
        resumes_active_session: true,
      },
    ]);
    expect(state.busy).toBe(true);
  });

  it("leaves this conversation idle when the wake resumes another one", () => {
    // The wake belongs to a session that is not on screen: a stop button here would
    // stop nothing, and the composer must stay usable.
    const state = run([
      {
        type: "job_complete",
        job_key: "j1",
        state: "done",
        resumes_active_session: false,
      },
    ]);
    expect(state.busy).toBe(false);
  });

  it("leaves this conversation idle when the server said nothing either way", () => {
    // An older server sends no flag. Guessing "busy" there would freeze the composer
    // of a chat that is not running, which is worse than the button it replaces.
    const state = run([{ type: "job_complete", job_key: "j1", state: "done" }]);
    expect(state.busy).toBe(false);
  });

  it("does not end a turn that is already running", () => {
    const state = run([
      { type: "submit_query", text: "go" },
      { type: "job_complete", job_key: "j1", state: "done", resumes_active_session: true },
    ]);
    expect(state.busy).toBe(true);
  });
});

describe("a question raised by something other than the turn", () => {
  it("leaves the turn producing, because it is still producing", () => {
    // A sub-agent works alongside the turn, not inside it: it can ask while the agent
    // is mid-step, and several can be working at once. Parking on its card showed the
    // agent as idle — spinner gone, transcript handed back — while it was still
    // running, and the next token then arrived into a turn the UI had closed.
    const state = run([
      { type: "submit_query", text: "do it" },
      { type: "token", text: "working" },
      {
        type: "user_question",
        id: "q2",
        questions: [],
        origin: { kind: "subagent", label: "vectorise the inner loop" },
      },
    ]);

    expect(state.busy).toBe(true);
  });
});

describe("a turn parked on a question", () => {
  const plan = { type: "user_question" as const, id: "q1", questions: [] };

  it("commits the streamed plan and stops the turn", () => {
    // The regression this exists for: the plan the user is being asked to approve
    // lived in `draft` alone — not in the transcript, not on the server, not on
    // disk — for as long as the card was up, so a reload in that window lost it.
    const state = run([
      { type: "submit_query", text: "plan it" },
      { type: "token", text: "Here is the plan." },
      plan,
    ]);

    expect(state.draft).toBe("");
    expect(state.busy).toBe(false);   // what hands the transcript back to the server
    const last = state.messages[state.messages.length - 1];
    expect(last.text).toBe("Here is the plan.");
    expect(last.provisional).toBe(true);
  });

  it("lets the answer that repeats it take its place", () => {
    // Nobody answered the card (no front-end, or it was dismissed): the loop
    // delivers the same prose as the answer, which must not appear twice.
    const state = run([
      { type: "submit_query", text: "plan it" },
      { type: "token", text: "Here is the plan." },
      plan,
      { type: "answer", text: "Here is the plan.\n\n— verification ledger —" },
    ]);

    const prose = state.messages.filter((m) => m.kind === "text" && m.role === "agent");
    expect(prose).toHaveLength(1);
    expect(prose[0].text).toContain("ledger");
  });

  it("keeps it when the answer is something else — an approved plan, or a refused one", () => {
    // Accepting runs the plan and answers with the report of doing so; rejecting
    // answers with the refusal. Neither repeats the plan, and dropping it would
    // take it out of a transcript the user watched it arrive in.
    const state = run([
      { type: "submit_query", text: "plan it" },
      { type: "token", text: "Here is the plan." },
      plan,
      { type: "answer", text: "Executed it: three files changed." },
    ]);

    expect(state.messages.map((m) => m.text)).toEqual([
      "plan it", "Here is the plan.", "Executed it: three files changed.",
    ]);
  });

  it("does not reach back into an older turn that was interrupted", () => {
    // A provisional bubble is only ever the current turn's. Left by a run that
    // never landed, it is the sole record of it — a later answer must not delete it.
    const state = run([
      { type: "submit_query", text: "plan it" },
      { type: "token", text: "Here is the plan." },
      plan,
      { type: "submit_query", text: "never mind, do this instead" },
      { type: "answer", text: "Here is the plan." },
    ]);

    expect(state.messages.filter((m) => m.text === "Here is the plan.")).toHaveLength(2);
    expect(state.messages.some((m) => m.provisional)).toBe(false);
  });

  it("goes back to work when the card is answered", () => {
    // Parking is not the end of the turn: the loop resumes on the answer, and
    // `busy` is what offers the stop button and makes that end observable — which
    // is when the finished transcript is handed back to be saved.
    const state = run([
      { type: "submit_query", text: "plan it" },
      { type: "token", text: "Here is the plan." },
      plan,
      { type: "prompt_answered", resumes: true },
    ]);

    expect(state.busy).toBe(true);
  });

  it("stays stopped when the answer ends the run", () => {
    const state = run([
      { type: "submit_query", text: "go" },
      plan,
      { type: "prompt_answered", resumes: false },
    ]);

    expect(state.busy).toBe(false);
  });

  it("adopts a turn that outlived the connection, then parks on its card", () => {
    // What a reconnect replays: the restore clears the turn state this window
    // arrives with, the server says a turn is still running, and the card it is
    // parked on lands last.
    expect(run([
      { type: "session_loaded_messages", messages: [] },
      { type: "turn_resumed" },
    ]).busy).toBe(true);

    expect(run([
      { type: "session_loaded_messages", messages: [] },
      { type: "turn_resumed" },
      plan,
    ]).busy).toBe(false);
  });

  it("parks the same way on a second card", () => {
    const state = run([
      { type: "submit_query", text: "go" },
      { type: "token", text: "Step one done." },
      plan,
    ]);

    expect(state.busy).toBe(false);
    expect(state.messages[state.messages.length - 1].provisional).toBe(true);
  });

  it("freezes the tool cards of the step it parked on", () => {
    const state = run([
      { type: "submit_query", text: "plan it" },
      { type: "tool_call", id: "t1", name: "grep" },
      { type: "tool_result", id: "t1", ok: true, summary: "4 hits" },
      plan,
    ]);

    expect(state.liveToolCalls).toHaveLength(0);
    expect(state.messages.some((m) => m.kind === "tools")).toBe(true);
  });
});

describe("a turn cut off by the connection", () => {
  it("keeps the prose that had arrived, provisionally", () => {
    const state = run([
      { type: "submit_query", text: "go" },
      { type: "token", text: "I started by reading" },
      { type: "connection_lost" },
    ]);

    const last = state.messages[state.messages.length - 1];
    expect(last.text).toBe("I started by reading");
    expect(last.provisional).toBe(true);
  });

  it("is replaced, not repeated, by the answer a reconnect brings", () => {
    // The worker keeps running through a dropped socket, so the turn can still
    // finish: the partial is a prefix of the answer, and only one of them is real.
    const state = run([
      { type: "submit_query", text: "go" },
      { type: "token", text: "I started by reading" },
      { type: "connection_lost" },
      { type: "answer", text: "I started by reading the loop, then fixed it." },
    ]);

    const prose = state.messages.filter((m) => m.role === "agent");
    expect(prose).toHaveLength(1);
    expect(prose[0].text).toBe("I started by reading the loop, then fixed it.");
  });
});
