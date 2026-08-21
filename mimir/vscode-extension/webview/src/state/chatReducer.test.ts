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
  it("coalesces token text across an intervening status into one bubble", () => {
    const state = run([
      { type: "token", text: "Hello" },
      { type: "status", text: "Running bash command" },
      { type: "token", text: " world" },
    ]);

    const streaming = state.messages.filter((m) => m.kind === "streaming");
    expect(streaming).toHaveLength(1);
    // Adjacent words join without an extra paragraph break.
    expect(streaming[0].text).toBe("Hello world");
    // Status text is dropped (decorrelated); it only flags a step boundary, which
    // is consumed by the second token. No thinking/tools messages are created.
    expect(state.messages.filter((m) => m.kind === "thinking")).toHaveLength(0);
    expect(state.messages.filter((m) => m.kind === "tools")).toHaveLength(0);
    expect(state.toolCallAfterToken).toBe(false);
  });

  it("does not change the streaming bubble id when carrying text forward", () => {
    const reducer = makeReducer();
    let s = reducer(initialChatState, { type: "token", text: "A" });
    const firstId = s.messages.find((m) => m.kind === "streaming")!.id;
    s = reducer(s, { type: "status", text: "Running bash command" });
    s = reducer(s, { type: "token", text: "B" });
    const sameId = s.messages.find((m) => m.kind === "streaming")!.id;
    expect(sameId).toBe(firstId);
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

  it("finalizes the streaming bubble on answer and clears busy", () => {
    const state = run([
      { type: "submit_query", text: "hi" },
      { type: "token", text: "partial" },
      { type: "answer", text: "the final answer" },
    ]);

    expect(state.busy).toBe(false);
    const streaming = state.messages.filter((m) => m.kind === "streaming");
    expect(streaming).toHaveLength(0);
    const last = state.messages[state.messages.length - 1];
    expect(last.kind).toBe("text");
    expect(last.text).toBe("the final answer");
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

  it("marks a failed tool_result as error", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "code_execute", label: "Running", detail: "make" },
      { type: "tool_result", id: "c1", name: "code_execute", ok: false, summary: "exit 1", duration_ms: 10 },
    ]);
    expect(state.liveToolCalls[0].status).toBe("error");
  });

  it("stores tool output on the activity from tool_result", () => {
    const state = run([
      { type: "tool_call", id: "c1", name: "grep", label: "Searching", detail: "x" },
      { type: "tool_result", id: "c1", name: "grep", ok: true, summary: "3 matches", output: "a.ts:1\nb.ts:2", duration_ms: 5 },
    ]);
    expect(state.liveToolCalls[0].output).toBe("a.ts:1\nb.ts:2");
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

  it("stacks one diff entry per event and keeps each entry's own is_new", () => {
    const s = run([
      // Creation: the emitter marks a new file with a /dev/null header.
      { type: "diff", file: "solver.py", patch: "--- /dev/null\n+++ b/solver.py\n+a" },
      // Two later edits of the same file, each a real patch.
      { type: "diff", file: "solver.py", patch: "--- a/solver.py\n+++ b/solver.py\n-a\n+b" },
      { type: "diff", file: "solver.py", patch: "--- a/solver.py\n+++ b/solver.py\n-b\n+c" },
    ]);
    const card = s.messages.find((m) => m.kind === "editing");
    expect(card?.diffs).toHaveLength(3);
    expect(card?.diffs?.map((d) => d.is_new)).toEqual([true, false, false]);
    expect(card?.diffs?.[2].patch).toContain("+c");
  });

  it("keeps distinct files as distinct entries", () => {
    const s = run([
      { type: "diff", file: "solver.py", patch: "--- a/solver.py\n+++ b/solver.py\n+a" },
      { type: "diff", file: "driver.py", patch: "--- /dev/null\n+++ b/driver.py\n+b" },
    ]);
    const card = s.messages.find((m) => m.kind === "editing");
    expect(card?.diffs?.map((d) => d.file)).toEqual(["solver.py", "driver.py"]);
  });
});
