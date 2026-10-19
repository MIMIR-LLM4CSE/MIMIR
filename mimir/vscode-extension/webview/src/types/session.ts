// Conversations: the list, one loaded with its transcript and checklist, a replay, and
// the transcript handed back to the server.

import type { ChatMessage } from "./chat";
import type { ServerMessage } from "./index";

export interface TodoItem {
  text: string;
  done: boolean;
}

export interface TodoMessage {
  type: "todo";
  items: TodoItem[];
}

export interface SessionMeta {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  preview?: string;
  /** One-sentence description of what was done in the session. */
  summary?: string;
  /** True once the user renamed the session by hand — the title then wins. */
  title_custom?: boolean;
  /** A turn of this conversation is in flight. Several run at once, so a conversation the
   *  user is not reading can be working — invisible without this. */
  running?: boolean;
  /** Its turn is parked on a card. Worse than invisible: the wait has no timeout, so
   *  the conversation stays stopped until somebody answers. */
  parked?: boolean;
  /** It asked for a turn but every agent slot is taken; it starts when one frees. */
  queued?: boolean;
  /** How many background runs it still has going.
   *
   *  Not the same question as `running`: a conversation that launched a two-hour build
   *  and answered has no turn in flight, and is exactly the one worth asking about
   *  before walking away. */
  runs?: number;
}

export interface SessionsListMessage {
  type: "sessions_list";
  sessions: SessionMeta[];
}

export interface SessionLoadedMessage {
  type: "session_loaded";
  session_id: string;
  title: string;
  display_messages: ChatMessage[];
  todos: TodoItem[];
  /** A turn of this session is still in flight — it outlived the last connection. */
  turn_running?: boolean;
  /** False only while this session's agent is being built. */
  agent_ready?: boolean;
}

/** What this conversation produced while nobody was attached, read from its journal
 *  past the client's watermark, in frames. Each event is fed to the same reducer.
 *
 *  Streamed deltas are absent: a turn arrives as its closing aggregates. `through_seq`
 *  is the next watermark, `more` says a frame follows, `truncated` that old events were
 *  cut at the cap. */
export interface ReplayMessage {
  type: "replay";
  session_id: string;
  events: ServerMessage[];
  through_seq: number;
  more: boolean;
  truncated: boolean;
}

/** The server does not hold the revision a transcript delta was built on: send it whole. */
export interface TranscriptResyncMessage {
  type: "transcript_resync";
  session_id: string;
}

/** The rendered chat, handed to the server so a reload can show it again.
 *
 *  Tool rows, reasoning panels and diff cards are assembled here and nowhere else,
 *  so the server cannot rebuild them — it stores this copy verbatim. `session_id`
 *  is what stops a transcript still in flight from landing on a session the user
 *  has since switched to. */
export interface TranscriptMessage {
  type: "transcript";
  session_id: string;
  /** Whole: every stored message. Absent on a delta. */
  messages?: ChatMessage[];
  /** A delta: `order` names every stored message by id, `changed` carries the ones
   *  new or replaced since revision `base_rev` of the same `epoch`. A server that
   *  does not hold that revision answers `transcript_resync`. */
  delta?: boolean;
  order?: string[];
  changed?: ChatMessage[];
  epoch?: string;
  base_rev?: number;
  rev?: number;
  /** How far into the session's journal these messages reach — the highest `seq` the
   *  reducer has taken in, from a live event or a `replay` frame. Stored server-side
   *  and used as the point a later re-attach replays *from*, so a window that has
   *  rendered everything is sent nothing. Omitted, the watermark simply does not
   *  advance and the next attach replays more than it needed to. */
  through_seq?: number;
}

export interface CreateSessionMessage {
  type: "create_session";
}

export interface SwitchSessionMessage {
  type: "switch_session";
  session_id: string;
}

export interface DeleteSessionMessage {
  type: "delete_session";
  session_id: string;
}

export interface RenameSessionMessage {
  type: "rename_session";
  session_id: string;
  title: string;
}

export interface ClearTodosMessage {
  type: "clear_todos";
}
