import { describe, it, expect } from "vitest";
import { EXEC_CLIP_CHARS, pruneForStorage } from "./transcriptUtils";
import type { ChatMessage, ToolActivity } from "../types";

function tool(extra: Partial<ToolActivity> = {}): ToolActivity {
  return {
    id: "c1", name: "bash", icon: "💻", label: "Running", detail: "",
    status: "ok", startedAt: 1_000, ...extra,
  };
}

describe("pruneForStorage", () => {
  it("keeps what a reload has to show again", () => {
    const messages: ChatMessage[] = [
      { id: "m1", role: "user", kind: "text", text: "hi" },
      { id: "m2", role: "agent", kind: "thinking", thinking: "hmm", thinkingTokens: 12 },
      { id: "m3", role: "agent", kind: "tools", tools: [tool({ parentId: "p1", origin: "explore #1" })] },
      { id: "m4", role: "agent", kind: "editing", diffs: [{ file: "a.py", patch: "@@" }] },
    ];
    expect(pruneForStorage(messages)).toEqual(messages);
  });

  it("drops the flags that only describe a turn in flight", () => {
    const [out] = pruneForStorage([
      { id: "m1", role: "agent", kind: "text", text: "…", streaming: true, live: true, queued: true },
    ]);
    expect(out).toEqual({ id: "m1", role: "agent", kind: "text", text: "…" });
  });

  it("leaves out an approval card nobody can answer any more", () => {
    // The reducer rewrites an answered card into text or removes it, so one still
    // holding an `approval` is an unresolved prompt — and it renders as nothing.
    const out = pruneForStorage([
      { id: "m1", role: "user", kind: "text", text: "go" },
      {
        id: "m2", role: "agent", kind: "approval",
        approval: {
          type: "approval", id: "a1", tool: "bash", server: "code",
          args: {}, risk: "medium", scope: "workspace", ids: ["a1"],
        },
      } as ChatMessage,
    ]);
    expect(out.map((m) => m.id)).toEqual(["m1"]);
  });

  it("clips captured output and says so, without touching the rest of the row", () => {
    const [out] = pruneForStorage([
      {
        id: "m1", role: "agent", kind: "tools",
        tools: [tool({
          exec: { command: "pytest", stdout: "x".repeat(EXEC_CLIP_CHARS + 500), stderr: "", returncode: 0 },
        })],
      },
    ]);
    const exec = out.tools![0].exec!;
    expect(exec.stdout.length).toBeLessThan(EXEC_CLIP_CHARS + 100);
    expect(exec.truncated).toBe(true);
    expect(exec.returncode).toBe(0);
    expect(out.tools![0].id).toBe("c1");
  });

  it("leaves a short exec panel exactly as it was", () => {
    const exec = { command: "ls", stdout: "a.py", stderr: "", returncode: 0 };
    const [out] = pruneForStorage([
      { id: "m1", role: "agent", kind: "tools", tools: [tool({ exec })] },
    ]);
    expect(out.tools![0].exec).toEqual(exec);
  });
});
