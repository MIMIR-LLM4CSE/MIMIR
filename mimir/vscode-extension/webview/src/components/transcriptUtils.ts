/**
 * Preparing the rendered chat for the trip back to the server.
 *
 * The transcript is stored so a reload can show the conversation as it was, which is a
 * narrower job than holding it live: anything that only means something *during* a turn
 * is dropped, and the one field with no ceiling — a command's captured output — is
 * clipped, so a long session's transcript stays a message rather than a payload.
 */
import type { ChatMessage, ToolActivity } from "../types";

/** Per-stream ceiling on captured command output kept in a stored transcript. */
export const EXEC_CLIP_CHARS = 4096;

function clip(text: string | undefined): { text: string | undefined; clipped: boolean } {
  if (!text || text.length <= EXEC_CLIP_CHARS) return { text, clipped: false };
  return { text: text.slice(0, EXEC_CLIP_CHARS) + "\n… (clipped)", clipped: true };
}

/** Clip a row's terminal panel; rows without one are returned untouched. */
function pruneTool(tool: ToolActivity): ToolActivity {
  if (!tool.exec) return tool;
  const out = clip(tool.exec.stdout);
  const err = clip(tool.exec.stderr);
  const cmd = clip(tool.exec.command);
  if (!out.clipped && !err.clipped && !cmd.clipped) return tool;
  return {
    ...tool,
    exec: {
      ...tool.exec,
      command: cmd.text,
      stdout: out.text ?? "",
      stderr: err.text ?? "",
      truncated: true,
    },
  };
}

/**
 * The transcript as it should be stored: no in-flight state, no unbounded output.
 *
 * `streaming`, `live` and `queued` describe a turn that is still happening; restoring
 * them would put a spinner over work that finished long ago. A message still holding an
 * `approval` is a prompt nobody answered — the reducer rewrites answered ones into text
 * or drops them — and it renders as nothing, so it is left out entirely.
 */
export function pruneForStorage(messages: ChatMessage[]): ChatMessage[] {
  return messages
    .filter((m) => !(m.kind === "approval" && m.approval))
    .map((m) => {
      const { approval, streaming, live, queued, ...rest } = m;
      const tools = rest.tools?.map(pruneTool);
      return tools ? { ...rest, tools } : rest;
    });
}
