/**
 * Preparing the rendered chat for the trip back to the server.
 *
 * The transcript is stored so a reload can show the conversation as it was, which is a
 * narrower job than holding it live: anything that only means something *during* a turn
 * is dropped, and the fields with no ceiling — a command's captured output, a failed
 * call's error, a written file's patch — are clipped, so a long session's transcript
 * stays a message rather than a payload: past the server's frame ceiling, sending it
 * would close the connection.
 */
import type { ChatMessage, DiffEntry, ToolActivity } from "../types";
import { MAX_FRAME_BYTES } from "../../../src/shared/limits";

/** Per-stream ceiling on captured command output kept in a stored transcript. */
export const EXEC_CLIP_CHARS = 4096;
/** Ceiling on a failed call's error: a whole compiler log fits, a runaway dump does not. */
export const ERROR_CLIP_CHARS = 64 * 1024;
/** Ceiling on a patch kept in a stored transcript: far past what anyone reads in the
 *  chat, so only a generated or data file's patch, megabytes long, is cut. */
export const DIFF_CLIP_CHARS = 1024 * 1024;
/** Closes a clipped patch. A `\` line is meta to the diff parser, drawn as written. */
export const DIFF_CLIP_NOTE = "\\ … (clipped in the stored transcript)";

function clip(
  text: string | undefined, limit = EXEC_CLIP_CHARS,
): { text: string | undefined; clipped: boolean } {
  if (!text || text.length <= limit) return { text, clipped: false };
  return { text: text.slice(0, limit) + "\n… (clipped)", clipped: true };
}

/** Clip a row's terminal panel and error text; rows with neither too long are returned untouched. */
function pruneTool(tool: ToolActivity): ToolActivity {
  const error = clip(tool.error, ERROR_CLIP_CHARS);
  if (error.clipped) tool = { ...tool, error: error.text };
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
 * *diff* as it should be stored: the patch cut at a line within the ceiling, and no
 * proposed content (that belongs to an approval, which is not stored).
 *
 * A clipped patch is `truncated`: reversed against the file it no longer describes in
 * full, it would give a wrong "before", so opening it opens the file instead.
 */
function pruneDiff(diff: DiffEntry): DiffEntry {
  const { new_content, ...rest } = diff;
  const patch = rest.patch ?? "";
  if (patch.length <= DIFF_CLIP_CHARS) return new_content === undefined ? diff : rest;
  const cut = patch.lastIndexOf("\n", DIFF_CLIP_CHARS);
  return {
    ...rest,
    patch: patch.slice(0, cut > 0 ? cut : DIFF_CLIP_CHARS) + "\n" + DIFF_CLIP_NOTE,
    truncated: true,
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
  const diffs = rest.diffs?.map(pruneDiff);
  return { ...rest, ...(tools && { tools }), ...(diffs && { diffs }) };
}

/** Stored size at which the user is told the conversation is getting large: half the
 *  frame ceiling a whole transcript must fit in, which leaves room to move on. */
export const SESSION_SIZE_WARN_BYTES = MAX_FRAME_BYTES / 2;

// A message's stored size, kept per message: the reducer never mutates one, so a
// growing session costs only its new messages to measure.
const storedSizes = new WeakMap<ChatMessage, number>();

/**
 * Roughly how many bytes *messages* weigh as a stored transcript: its JSON length,
 * which counts characters rather than UTF-8 bytes — close enough for a warning.
 */
export function storedSize(messages: ChatMessage[]): number {
  let total = 0;
  for (const m of messages) {
    if (!keepForStorage(m)) continue;
    let size = storedSizes.get(m);
    if (size === undefined) {
      size = JSON.stringify(pruneMessage(m)).length;
      storedSizes.set(m, size);
    }
    total += size;
  }
  return total;
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
