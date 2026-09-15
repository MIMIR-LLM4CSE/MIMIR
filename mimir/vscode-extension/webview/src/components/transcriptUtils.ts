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
 *
 * `provisional` is the exception that is deliberately kept: it marks prose whose turn
 * has not landed yet, and the turn it belongs to can outlive this transcript by a long
 * way — a plan parked on its approval card waits for a person. A reload in that window
 * must come back still knowing the bubble is provisional, or the `answer` that finally
 * arrives repeats it instead of replacing it.
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

/**
 * Plain-text bubbles in *messages* — the part of a transcript both sides share.
 *
 * Mirrors ``_Session._text_count`` on the server: the server assembles nothing but
 * text bubbles, so this is the one measure on which the two copies are comparable,
 * and the only honest way to ask which of them holds more of the conversation.
 */
export function textBubbleCount(messages: ChatMessage[]): number {
  return messages.filter((m) => (m.kind ?? "text") === "text").length;
}

/** What a `session_loaded` should leave on screen, and whether to push it back. */
export interface RestoreChoice {
  messages: ChatMessage[];
  /** True when what we kept is richer than what arrived — the server's copy is behind. */
  push: boolean;
}

/**
 * Decide between the stored transcript and the one already on screen.
 *
 * A `session_loaded` for a *different* session is a switch: whatever it carries is the
 * conversation, and what was on screen belongs to the one being left.
 *
 * For the *same* session it is a reconnect, and the two copies are rivals. The one on
 * screen is the richer by construction — the server only ever assembles text bubbles,
 * and it is handed the rendered transcript at points the client chooses — so taking the
 * stored copy unconditionally is what turned a dropped connection into a conversation
 * reduced to its questions. Kept whenever it holds at least as much, which is the same
 * test the server applies to transcripts we send it (``_handle_transcript``), read from
 * the other side. Pushed back when it holds strictly more, so the disk copy stops being
 * behind rather than waiting for the next turn to end.
 */
export function chooseRestoredMessages(
  incoming: ChatMessage[] | undefined,
  onScreen: ChatMessage[],
  sameSession: boolean,
): RestoreChoice {
  const stored = incoming ?? [];
  if (!sameSession) return { messages: stored, push: false };
  const here = textBubbleCount(onScreen);
  const there = textBubbleCount(stored);
  if (here < there) return { messages: stored, push: false };
  if (onScreen.length === 0) return { messages: stored, push: false };
  return { messages: onScreen, push: here > there };
}
