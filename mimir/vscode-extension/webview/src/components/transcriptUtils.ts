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
 * The transcript as it should be stored: no in-flight state (`streaming`, `live`,
 * `queued`), no unanswered approval card, no unbounded output.
 *
 * `provisional` is kept: a reload while its turn is still parked must know the bubble
 * is provisional, so the `answer` replaces it rather than repeating it.
 */
export function pruneForStorage(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter(keepForStorage).map(pruneMessage);
}

/** Whether *m* belongs in a stored transcript at all. */
function keepForStorage(m: ChatMessage): boolean {
  if (m.kind === "approval" && m.approval) return false;
  // Cards about the moment rather than about the conversation. Stored, they are
  // replayed on every load: a conversation reconnected to twenty times reopens on
  // twenty of them, each describing a past attachment as if it were now. The detach
  // notice is one by nature, however it reached the transcript — naming it is what
  // takes the copies already there back out.
  return !(m.kind === "command"
           && (m.command?.transient || m.command?.command === "detach"));
}

/** *m* as it should be stored: no in-flight state, no unbounded output. */
function pruneMessage(m: ChatMessage): ChatMessage {
  const { approval, streaming, live, queued, ...rest } = m;
  const tools = rest.tools?.map(pruneTool);
  return tools ? { ...rest, tools } : rest;
}

/** What this window last told the server, for sending only what changed since. */
export interface TranscriptSync {
  session: string;
  /** Names one run of full-then-deltas; the server drops a delta from another. */
  epoch: string;
  rev: number;
  sent: Map<string, ChatMessage>;
}

export type TranscriptBody =
  | { messages: ChatMessage[]; epoch: string; rev: number }
  | { delta: true; order: string[]; changed: ChatMessage[];
      epoch: string; base_rev: number; rev: number };

/**
 * The transcript to send, whole or as a delta, and the record to keep afterwards.
 *
 * A delta names every stored message in order and carries only those that are new
 * or were replaced since the last send. The reducer never mutates a message, so
 * "replaced" is an identity test. Whole when forced, on another session, the first
 * time, or when a message has no id to be addressed by.
 */
export function planTranscript(
  messages: ChatMessage[],
  session: string,
  prev: TranscriptSync | null,
  force: boolean,
  newEpoch: () => string,
): { body: TranscriptBody; next: TranscriptSync } {
  const kept = messages.filter(keepForStorage);
  const sent = new Map(kept.map((m) => [m.id, m]));
  const addressable = sent.size === kept.length && kept.every((m) => !!m.id);
  if (force || !prev || prev.session !== session || !addressable) {
    const epoch = newEpoch();
    return {
      body: { messages: kept.map(pruneMessage), epoch, rev: 0 },
      next: { session, epoch, rev: 0, sent },
    };
  }
  const changed = kept.filter((m) => prev.sent.get(m.id) !== m).map(pruneMessage);
  const rev = prev.rev + 1;
  return {
    body: {
      delta: true, order: kept.map((m) => m.id), changed,
      epoch: prev.epoch, base_rev: prev.rev, rev,
    },
    next: { session, epoch: prev.epoch, rev, sent },
  };
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
 * Decide between the stored transcript and the one on screen.
 *
 * A different session is a switch: take the stored one. The same session is a
 * reconnect: keep the screen's copy when it holds at least as much (the server's test,
 * from the other side), and push it back when it holds more.
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
