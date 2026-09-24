import type { ChatMessage, DiffEntry, ServerMessage, ToolActivity } from "../types";
import {
  insertAfterFamily, relabelOrigins, settleFamily,
} from "../components/subAgentUtils";
import { orderLiveStream } from "../components/liveStreamUtils";

// A reasoning block — reasoning TEXT only. Tool calls are tracked separately
// (see `liveToolCalls`); thinking and tools are fully decorrelated.
export type ThinkingBlock = {
  id: string;
  text: string;
  done: boolean;
  collapsed: boolean;
  /** Epoch ms when this block opened — drives the live timer and frozen duration. */
  startedAt: number;
  /** Arrival stamp, used to place the block among this step's prose and tool
   *  rows (see orderLiveStream). */
  seq: number;
  /** Reasoning tokens reported by the server as the block closed. Accumulated,
   *  since a block that reopens (reasoning resumed before the answer) is fed by
   *  more than one `thinking_end`. */
  tokens?: number;
};

export interface ChatState {
  messages: ChatMessage[];
  /** Prose of the turn in flight, held OUT of the transcript until the loop
   *  accepts it. A turn only becomes an answer once it ends with no tool call
   *  and no guardrail sends the model back to work, so anything streamed before
   *  that is a draft: putting it straight into the message list is what made a
   *  finished-looking answer appear and then vanish. */
  draft: string;
  /** Live reasoning blocks for the current turn (text only). */
  liveThinkingBlocks: ThinkingBlock[];
  /** Live structured tool invocations for the current turn — independent of
   *  thinking. Frozen into a `kind:"tools"` message at each step boundary. */
  liveToolCalls: ToolActivity[];
  busy: boolean;
  /** True when tool calls / a thinking phase occurred since the last token, so
   *  the next token starts a new paragraph in the draft. */
  toolCallAfterToken: boolean;
  /** Diffs accumulated for the current turn, attached to the final answer. */
  pendingDiffs: DiffEntry[];
  /** Thinking text accumulated for the current turn. */
  pendingThinking: string;
  /** Next arrival stamp. Handed to each live item — the draft, a reasoning
   *  block, a tool row — so the order they came in survives being collected
   *  into three separate streams. Monotonic for the life of the window. */
  nextSeq: number;
  /** Stamp of the draft's first token; null whenever the draft is empty. */
  draftSeq: number | null;
}

export const initialChatState: ChatState = {
  messages: [],
  draft: "",
  liveThinkingBlocks: [],
  liveToolCalls: [],
  busy: false,
  toolCallAfterToken: false,
  pendingDiffs: [],
  pendingThinking: "",
  nextSeq: 1,
  draftSeq: null,
};

// Server messages the reducer acts on are a subset of ServerMessage; the rest
// (ready/config/todo/sessions/context_*) are handled by plain setState in App.
export type ChatAction =
  | ServerMessage
  | { type: "submit_query"; text: string }
  | { type: "steer_query"; text: string }
  | { type: "reset" }
  | { type: "connection_lost" }
  | { type: "session_loaded_messages"; messages: ChatMessage[] }
  | { type: "turn_resumed" }
  | { type: "prompt_answered"; resumes: boolean }
  | { type: "toggle_thinking"; id: string }
  | { type: "approval_response"; id: string; choice: "y" | "n" | "a" };

// Tool name → icon. Keyed on the structured tool name (robust to wording),
// with prefix fallbacks so unknown/new tools still get a sensible glyph.
const TOOL_ICONS: Record<string, string> = {
  read_file: "📖", read_file_lines: "📖",
  grep: "🔍", search_code_patterns: "🔍", find_similar_files: "🔍", find_files: "🔍",
  list_directory: "📂", list_files: "📂", file_exists: "📂", get_file_info: "📂",
  tree_summary: "🌳",
  write_file: "✏️", append_file: "✏️", replace_all_in_file: "✏️",
  insert_text: "✏️", replace_lines: "✏️",
  delete_file: "🗑️",
  web_fetch: "🌐",
};

/** Returns an icon for a structured tool by name (exact, then prefix, then default). */
export function iconForTool(name: string): string {
  if (name in TOOL_ICONS) return TOOL_ICONS[name];
  if (name.includes("bash") || name.includes("shell") || name.includes("terminal")) return "💻";
  if (name.startsWith("code_")) return "⚡";
  if (name.startsWith("db_")) return "🗄️";
  if (name.startsWith("proxy_") || name.startsWith("platform_")) return "🖥️";
  if (name.startsWith("salloc") || name.startsWith("slurm") || name.startsWith("hpc_")) return "🏗️";
  if (name.includes("task") || name.includes("todo")) return "📋";
  if (name.includes("memory")) return "🧠";
  if (name.includes("search") || name.includes("find") || name.includes("grep")) return "🔍";
  if (name.includes("read") || name.includes("get")) return "📖";
  if (name.includes("write") || name.includes("edit") || name.includes("replace")) return "✏️";
  return "🔧";
}

/**
 * Apply *transform* to whichever list currently holds the row *id*.
 *
 * A row lives in `liveToolCalls` during its step, then gets baked into a
 * `kind:"tools"` message. Late-arriving news about it (its result, a verdict, a
 * sub-agent step) can land on either side of that boundary — showing an approval
 * card mid-flight runs flushLive on a still-running row — so every patch has to
 * look in both places. Returns null when the row is nowhere: an event for a turn the
 * user has already cleared is dropped, not applied to a stranger.
 */
function patchToolWhere(
  state: ChatState,
  match: (t: ToolActivity) => boolean,
  transform: (tools: ToolActivity[]) => ToolActivity[],
): ChatState | null {
  if (state.liveToolCalls.some(match)) {
    return { ...state, liveToolCalls: transform(state.liveToolCalls) };
  }
  const idx = state.messages.findIndex(
    (m) => m.kind === "tools" && (m.tools ?? []).some(match)
  );
  if (idx >= 0) {
    return {
      ...state,
      messages: state.messages.map((m, i) =>
        i === idx ? { ...m, tools: transform(m.tools ?? []) } : m
      ),
    };
  }
  return null;
}

/** The common case: the row is known by the call id it was opened with. */
function patchToolById(
  state: ChatState,
  id: string,
  transform: (tools: ToolActivity[]) => ToolActivity[],
): ChatState | null {
  return patchToolWhere(state, (t) => t.id === id, transform);
}

/**
 * Append *msg* to the transcript, keeping stamped messages in arrival order.
 *
 * Anything that can land while a step is still in flight is stamped — a steer
 * bubble, an edit card, a session-command reply — and so is everything that step
 * freezes. The tail of the transcript is therefore a run of stamped messages in
 * stamp order, and a new one is inserted into that run instead of being dropped at
 * the end. Which is what a steer needs in both directions: the bubble goes under
 * the cards that were already there, and the prose that was in flight when it
 * arrived is later committed back above it.
 *
 * An unstamped message (a query, an answer, an error) goes last and closes the run.
 */
function appendStamped(messages: ChatMessage[], msg: ChatMessage): ChatMessage[] {
  const seq = msg.seq;
  if (seq === undefined) return [...messages, msg];
  let at = messages.length;
  while (at > 0) {
    const prev = messages[at - 1].seq;
    if (prev === undefined || prev <= seq) break;
    at--;
  }
  return [...messages.slice(0, at), msg, ...messages.slice(at)];
}

/**
 * Bake the turn's live state into the transcript and clear it, in arrival order.
 *
 * Prose, reasoning blocks and tool rows are collected in three separate streams,
 * so the order has to be reconstructed from the stamps they were handed
 * (orderLiveStream). The live view reads the same stamps through the same
 * function: a card the user watched arrive under a paragraph stays under it once
 * the step ends, instead of jumping above it as the freeze regrouped by kind.
 *
 * Reasoning and tools stay decorrelated — each non-empty block is its own
 * `kind:"thinking"` message, and each run of consecutive tool rows one
 * `kind:"tools"` message.
 *
 * *provisional* flags the committed prose for the two cases where the turn was
 * not accepted and the draft is nonetheless the only copy of it anywhere: it
 * parked on a question, or the connection dropped under it. See commitDraft.
 */
function flushLive(
  state: ChatState, makeId: () => string, provisional = false,
): ChatState {
  const blocks = state.liveThinkingBlocks.filter((b) => b.text.trim().length > 0);
  const tools = state.liveToolCalls;
  const hasDraft = state.draft.trim().length > 0;
  if (!hasDraft && blocks.length === 0 && tools.length === 0) {
    // Nothing worth keeping — but a whitespace-only draft and empty reasoning
    // blocks still have to go, or they outlive the step that opened them.
    if (
      state.draft === "" &&
      state.liveThinkingBlocks.length === 0 &&
      state.liveToolCalls.length === 0
    ) return state;
    return { ...state, draft: "", draftSeq: null, liveThinkingBlocks: [], liveToolCalls: [] };
  }

  const frozenAt = Date.now();
  // A draft with no stamp cannot happen through `token`, but placing it first is
  // the safe reading of one: prose is never lost, only possibly placed early.
  const entries = orderLiveStream({
    thinking: blocks,
    tools,
    draftSeq: hasDraft ? state.draftSeq ?? 0 : null,
  });

  const frozen: ChatMessage[] = entries.map((e) => {
    // orderLiveStream also places messages that landed mid-step; the transcript
    // already holds those, and appendStamped below is what keeps them in order.
    if (e.kind === "message") return e.message;
    if (e.kind === "draft") {
      return {
        id: makeId(), seq: e.seq,
        role: "agent" as const, kind: "text" as const, text: state.draft,
        ...(provisional ? { provisional: true } : {}),
      };
    }
    if (e.kind === "thinking") {
      return {
        id: e.block.id,
        seq: e.seq,
        role: "agent" as const,
        kind: "thinking" as const,
        text: e.block.text,
        live: false, // start collapsed — user can click to expand
        thinkingDurationMs: Math.max(0, frozenAt - e.block.startedAt),
        thinkingTokens: e.block.tokens,
      };
    }
    // Any tool still "running" at freeze time never received a result; stamp a
    // final duration and mark it done so its live timer stops ticking instead
    // of counting up forever in the frozen card.
    return {
      id: makeId(),
      seq: e.seq,
      role: "agent" as const,
      kind: "tools" as const,
      live: false,
      tools: e.tools.map((t) =>
        t.status === "running"
          ? { ...t, status: "ok" as const, durationMs: t.durationMs ?? Math.max(0, frozenAt - t.startedAt) }
          : t
      ),
    };
  });

  return {
    ...state,
    draft: "",
    draftSeq: null,
    liveThinkingBlocks: [],
    liveToolCalls: [],
    // Inserted rather than appended: a steer bubble may already sit at the tail,
    // and the prose and cards of the step it interrupted belong above it.
    messages: frozen.reduce(appendStamped, state.messages),
  };
}

/**
 * Move the draft into the transcript, if there is anything in it.
 *
 * Called before anything else is appended below it — a card, a frozen tool list,
 * an error — so committed prose keeps its place in the order it was written.
 * NOT called on `answer` (the authoritative text supersedes the draft) or on
 * `nudge_injected` (the turn was not accepted, so there is nothing to keep).
 *
 * *provisional* is for the two places where the turn has not been accepted and yet
 * the draft must not be thrown away: it parked on a question, or the connection
 * dropped under it. Both interrupt the turn at a point where this prose is the only
 * copy of it anywhere — it is not in `messages`, not on the server, not on disk — so
 * it is committed, flagged, and dropped again by the `answer` that supersedes it.
 */
function commitDraft(
  state: ChatState, makeId: () => string, provisional = false,
): ChatState {
  if (!state.draft.trim()) {
    return state.draft ? { ...state, draft: "", draftSeq: null } : state;
  }
  return {
    ...state,
    draft: "",
    draftSeq: null,
    messages: appendStamped(state.messages, {
      id: makeId(), seq: state.draftSeq ?? undefined,
      role: "agent", kind: "text", text: state.draft,
      ...(provisional ? { provisional: true } : {}),
    }),
  };
}

/**
 * True when *answer* is the finished form of the provisional prose *held*.
 *
 * The two shapes that produce one: a turn whose prose was committed because it parked
 * on a question nobody was asked anything else about, and a turn cut off mid-sentence
 * that later finished — the held text is then a prefix of the answer. Compared on
 * collapsed whitespace, since the draft's paragraph breaks are inserted by the token
 * coalescing rather than by the model.
 */
function isRewordedBy(answer: string, held: string | undefined): boolean {
  const flat = (t: string) => t.replace(/\s+/g, " ").trim();
  const a = flat(answer ?? "");
  const h = flat(held ?? "");
  return h.length > 0 && a.startsWith(h);
}

/** Settle every provisional bubble: they are ordinary transcript from here on. */
function clearProvisional(messages: ChatMessage[]): ChatMessage[] {
  if (!messages.some((m) => m.provisional)) return messages;
  return messages.map((m) => {
    if (!m.provisional) return m;
    const { provisional, ...rest } = m;
    return rest;
  });
}

/**
 * Commit the draft and stop the turn's live state, for a turn parked on a question.
 *
 * A plan approval and a clarification batch both park the worker thread
 * indefinitely: the agent is waiting on a person, and until they answer there
 * is no `answer` event, so nothing else marks this moment. That matters beyond the
 * spinner — the transcript only travels back to the server when `busy` falls, and the
 * plan the user is being asked to approve lives, until this runs, in `draft` alone.
 */
function parkOnPrompt(state: ChatState, makeId: () => string): ChatState {
  const s = flushLive(state, makeId, true);
  return { ...s, busy: false, toolCallAfterToken: false };
}

/**
 * Pure reducer for the streaming / thinking / approval / diff state machine.
 * `makeId` is injected so the reducer stays deterministic under test.
 */
export function createChatReducer(makeId: () => string) {
  return function reducer(state: ChatState, action: ChatAction): ChatState {
    switch (action.type) {
      // ── Local actions ──────────────────────────────────────────────────────
      case "submit_query": {
        const userMsg: ChatMessage = { id: makeId(), role: "user", kind: "text", text: action.text };
        return {
          ...state,
          // Whatever a previous turn left provisional is settled by a new question
          // being asked: it is the transcript now, and the `answer` this turn ends on
          // must supersede its own prose only — never reach back and delete an older
          // turn's, which is all that would be left of a run that was interrupted.
          messages: [...clearProvisional(state.messages), userMsg],
          liveThinkingBlocks: [],
          liveToolCalls: [],
          busy: true,
          toolCallAfterToken: false,
        };
      }

      // A message typed while the agent is busy. Append it as a user bubble
      // tagged "queued" (awaiting injection) without disturbing the running
      // turn's live state — busy stays true; the run never stopped.
      case "steer_query": {
        // Stamped: it arrives in the middle of a step, and the prose and cards
        // already on screen came before it. Without a stamp the bubble was drawn
        // above the whole step in flight, which read as if the agent had answered
        // a message the user had not sent yet.
        const userMsg: ChatMessage = {
          id: makeId(), seq: state.nextSeq,
          role: "user", kind: "text", text: action.text, queued: true,
        };
        return {
          ...state,
          nextSeq: state.nextSeq + 1,
          messages: appendStamped(state.messages, userMsg),
        };
      }

      // Server confirmed a queued steer reached the running agent. Clear the
      // "queued" tag on the oldest still-queued bubble (FIFO injection order).
      case "steer_injected": {
        const idx = state.messages.findIndex((m) => m.queued);
        if (idx < 0) return state;
        return {
          ...state,
          messages: state.messages.map((m, i) =>
            i === idx ? { ...m, queued: false } : m
          ),
        };
      }

      // A guardrail nudge fired, which means the loop did not accept the turn the
      // model just wrote as its answer. Two things follow.
      //
      // The nudge is not rendered: it occupies the user's turn slot without coming
      // from the user, and as a bubble it reads as instructions nobody typed.
      //
      // The draft is dropped, because nothing has been answered yet — the model is
      // about to reconsider and either go back to work or write a different ending.
      // Nothing disappears from the transcript here: a provisional turn never
      // entered it, which is the whole point of holding it in `draft`.
      case "nudge_injected":
        return state.draft ? { ...state, draft: "", draftSeq: null } : state;

      case "reset":
        return initialChatState;

      case "connection_lost": {
        // Keep whatever prose had arrived: the turn was cut off, not superseded.
        // Provisional, because the turn on the other end is not necessarily dead —
        // the worker runs on, and a reconnect can still be handed the `answer` it
        // ends on, which must replace this partial rather than repeat under it.
        const s = commitDraft(state, makeId, true);
        return { ...s, busy: false, liveThinkingBlocks: [], liveToolCalls: [] };
      }

      // The turn parked on a question for the user: a plan awaiting approval or a
      // clarification batch. Rendering the card is App's business; what belongs
      // here is that the turn stopped producing.
      //
      // Unless the card came from somewhere else. A sub-agent runs alongside the turn,
      // not inside it, so its question says nothing about whether the turn is still
      // producing — parking on it would show the agent as idle while it works.
      case "user_question":
        return action.origin ? state : parkOnPrompt(state, makeId);

      case "session_loaded_messages":
        return {
          ...state,
          // What a run was doing is not worth restoring: it described a moment, and
          // the watcher that could have refreshed it may have died with the process
          // that wrote the file. Left in place, a row saved mid-build would come back
          // claiming to be building, and its dock card would never leave.
          messages: action.messages.map((m) =>
            m.tools?.some((t) => t.phase !== undefined || t.percent !== undefined)
              ? { ...m, tools: m.tools.map((t) => ({ ...t, phase: undefined, percent: undefined })) }
              : m
          ),
          draft: "",
          draftSeq: null,
          liveThinkingBlocks: [],
          liveToolCalls: [],
          busy: false,
          pendingDiffs: [],
          pendingThinking: "",
        };

      // A turn of this session outlived the connection and is still producing, so
      // the run is live even though this window did not start it. Dispatched after
      // session_loaded_messages, which clears the state a reload arrives with; a card
      // the turn is parked on lands after this and parks it again.
      case "turn_resumed":
        return state.busy ? state : { ...state, busy: true };

      // The card a turn parked on was answered. The turn goes back to producing —
      // the same tail as `approval_response`, and the same reason: `busy` is what
      // offers the stop button, and what makes the end of the turn observable at
      // all, which is when the finished transcript is handed back for saving.
      // `resumes` is false for an answer that ends the run instead, where claiming
      // a live turn would strand the composer.
      case "prompt_answered":
        return action.resumes ? { ...state, busy: true } : state;

      case "toggle_thinking": {
        const { id } = action;
        let liveThinkingBlocks = state.liveThinkingBlocks;
        if (liveThinkingBlocks.some((b) => b.id === id)) {
          liveThinkingBlocks = liveThinkingBlocks.map((b) =>
            b.id === id ? { ...b, collapsed: !b.collapsed } : b
          );
        }
        // 'live' doubles as the "expanded" flag for frozen thinking messages.
        const messages = state.messages.map((m) =>
          m.id === id && m.kind === "thinking" ? { ...m, live: !m.live } : m
        );
        return { ...state, liveThinkingBlocks, messages };
      }

      case "approval_response": {
        const { id, choice } = action;
        const messages = state.messages
          .map((m) => {
            if (m.kind === "approval" && m.approval?.id === id) {
              if (choice === "n") {
                const tool = m.approval?.tool ?? "";
                const isBash = /bash|shell|run/i.test(tool);
                const approvalArgs = m.approval?.args ?? {};
                const commandKey = Object.keys(approvalArgs).find((k) => /^(command|cmd|script)$/i.test(k))
                  ?? Object.keys(approvalArgs).find((k) => /^(filepath|path|file)$/i.test(k));
                const firstArgValue = commandKey !== undefined
                  ? approvalArgs[commandKey]
                  : Object.values(approvalArgs).find((v) => typeof v === "string");
                const cmdText = typeof firstArgValue === "string"
                  ? firstArgValue.split("\n")[0].trim().slice(0, 80)
                  : null;
                const label = isBash && cmdText ? `✕ Blocked: \`${cmdText}\`` : "✕ Blocked";
                return { ...m, kind: "text" as const, text: label, approval: undefined };
              }
              // y / a → remove approval card; editing card stays live until write completes.
              return null;
            }
            // On deny: remove the live editing card too (nothing was written).
            if (choice === "n" && m.kind === "editing" && m.live) return null;
            return m;
          })
          .filter(Boolean) as ChatMessage[];
        return { ...state, messages, busy: true };
      }

      // A finished background run auto-resumes the session that launched it. No one
      // pressed send, so `busy` — which only `submit_query` sets — would stay false
      // for the whole turn: the composer keeps offering "send" and the stop button
      // never appears, leaving a running agent that cannot be interrupted.
      //
      // Only when the wake resumes THIS conversation. A wake for a detached session
      // runs a turn elsewhere, and marking this chat busy for it would show a stop
      // button that stops nothing.
      case "job_complete": {
        // First, close the loop on the row the user detached, if this is that job:
        // it has been sitting on "moved to background" with only partial output, and
        // this is the only message that ever says how the run actually ended. Done
        // before the branch below clears liveToolCalls, since the row may still be
        // there.
        // Either handle finds the row: a shell run detached mid-flight is known by
        // the job key on its terminal panel, while a run with no output to preview —
        // an optimization run — carries the key on the row itself. Before the second
        // one existed, such a row settled as an ordinary success and its ending
        // landed nowhere.
        const isThisJob = (t: ToolActivity) =>
          t.exec?.job_key === action.job_key || t.jobKey === action.job_key;
        const settled = patchToolWhere(
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
      }

      // ── Server messages ────────────────────────────────────────────────────
      case "output":
      case "status": {
        const t = action.text.trim().toLowerCase();
        // Ignore most file-operation statuses — UI is driven by diff + approval.
        if (
          !t ||
          t.startsWith("✓") ||
          t.startsWith("performing tool") ||
          (t.startsWith("reading") && !/^reading (?:lines? \d|from line \d)/.test(t))
        ) {
          return state;
        }
        // Transient status/output text is dropped (decorrelated from both
        // thinking and tools); we only note that activity occurred so the next
        // token opens a fresh streaming bubble.
        return { ...state, toolCallAfterToken: true };
      }

      // The answer to a command the user typed. Unlike "output" — transient tool
      // chatter this reducer deliberately drops — it is rendered, because dropping it
      // made "/memory list" print nothing and "/memory clear" wipe the store in
      // silence. Does not touch `busy`: a session command runs beside a turn, not
      // as one.
      case "command_output": {
        const title = (action.title ?? "").trim();
        if (!title) return state;
        const s = commitDraft(state, makeId);
        return {
          ...s,
          nextSeq: s.nextSeq + 1,
          messages: appendStamped(s.messages, {
              id: makeId(),
              seq: s.nextSeq,
              role: "agent",
              kind: "command",
              // Carried whole rather than flattened to a line: the card decides how
              // a setting change and a twenty-row listing each want to look.
              command: {
                type: "command_output",
                command: action.command ?? "",
                title,
                items: (action.items ?? []).filter((i) => (i?.label ?? "").trim()),
                note: (action.note ?? "").trim(),
                tone: action.tone ?? "ok",
              },
          }),
        };
      }

      case "token": {
        const delta = action.text;
        const isNewLLMStep = state.toolCallAfterToken;
        let s: ChatState = { ...state, toolCallAfterToken: false };
        if (isNewLLMStep) {
          // Only a step that actually produced cards is a boundary worth breaking
          // the prose at: prose written before this step's tools belongs above
          // them, so it is committed first and the cards are frozen in under it.
          // A bare status flagged activity that has nothing to show, and splitting
          // one turn's prose over it would fragment it for nothing.
          const hasCards =
            s.liveToolCalls.length > 0 ||
            s.liveThinkingBlocks.some((b) => b.text.trim().length > 0);
          if (hasCards) {
            s = flushLive(s, makeId);
          } else if (s.draft && !/\s$/.test(s.draft) && !/^\s/.test(delta)) {
            s = { ...s, draft: s.draft + "\n\n" };
          }
        }
        // The stamp of the first token is what later places this prose among the
        // step's cards; it is cleared with the draft, so a flush re-stamps.
        if (s.draftSeq === null) s = { ...s, draftSeq: s.nextSeq, nextSeq: s.nextSeq + 1 };
        return { ...s, draft: s.draft + delta };
      }

      case "file_progress": {
        const progressDiffs: DiffEntry[] = action.diffs ?? [];
        if (progressDiffs.length === 0) return state;
        const s0 = commitDraft(state, makeId);
        const liveIdx = s0.messages.findIndex((m) => m.kind === "editing" && m.live);
        let messages: ChatMessage[];
        if (liveIdx >= 0) {
          messages = s0.messages.map((m, i) => {
            if (i !== liveIdx) return m;
            const existing = m.diffs ?? [];
            const merged = [...existing];
            for (const d of progressDiffs) {
              const fi = merged.findIndex((e) => e.file === d.file);
              if (fi >= 0) merged[fi] = d;
              else merged.push(d);
            }
            return { ...m, diffs: merged };
          });
        } else {
          // Stamped: the write that produced these diffs is a tool row of the step
          // in flight, and the card belongs under it rather than above the step.
          messages = appendStamped(s0.messages, {
            id: makeId(), seq: s0.nextSeq,
            role: "agent", kind: "editing", live: true, diffs: progressDiffs,
          });
          return { ...s0, nextSeq: s0.nextSeq + 1, messages };
        }
        return { ...s0, messages };
      }

      case "approval": {
        let s = flushLive({ ...state, toolCallAfterToken: false }, makeId);
        s = { ...s, busy: false, pendingDiffs: [] };
        const base = s.messages.map((m) =>
          m.kind === "editing" && m.live ? { ...m, live: false } : m
        );
        let existingIdx = -1;
        for (let i = base.length - 1; i >= 0; i--) {
          if (base[i].kind === "approval" && base[i].approval !== undefined) { existingIdx = i; break; }
        }
        let messages: ChatMessage[];
        if (existingIdx >= 0) {
          const existing = base[existingIdx];
          const existingIds = existing.approval!.ids ?? [existing.approval!.id];
          messages = base.map((m, i) =>
            i === existingIdx
              // A card put back on return from another session can meet its own
              // saved copy: the same id twice would answer it twice.
              ? { ...m, approval: { ...action, ids: existingIds.includes(action.id) ? existingIds : [...existingIds, action.id] } }
              : m
          );
        } else {
          messages = [
            ...base,
            {
              id: makeId(),
              role: "agent",
              kind: "approval",
              approval: { ...action, ids: [action.id] },
            },
          ];
        }
        return { ...s, messages };
      }

      case "thinking_start": {
        const s = { ...state, toolCallAfterToken: true };
        const last = s.liveThinkingBlocks[s.liveThinkingBlocks.length - 1];
        if (last && !last.done) return s; // reuse active block
        return {
          ...s,
          nextSeq: s.nextSeq + 1,
          liveThinkingBlocks: [
            ...s.liveThinkingBlocks,
            { id: makeId(), text: "", done: false, collapsed: false, startedAt: Date.now(), seq: s.nextSeq },
          ],
        };
      }

      case "thinking": {
        const thinkChunk = action.text;
        const s = { ...state, pendingThinking: state.pendingThinking + thinkChunk };
        if (s.liveThinkingBlocks.length === 0) {
          return {
            ...s,
            nextSeq: s.nextSeq + 1,
            liveThinkingBlocks: [{ id: makeId(), text: thinkChunk, done: false, collapsed: false, startedAt: Date.now(), seq: s.nextSeq }],
          };
        }
        const lastIdx = s.liveThinkingBlocks.length - 1;
        const last = s.liveThinkingBlocks[lastIdx];
        if (!last.done) {
          return {
            ...s,
            liveThinkingBlocks: s.liveThinkingBlocks.map((b, i) =>
              i === lastIdx
                ? { ...b, text: b.text + thinkChunk }
                : b
            ),
          };
        }
        return {
          ...s,
          nextSeq: s.nextSeq + 1,
          liveThinkingBlocks: [
            ...s.liveThinkingBlocks,
            { id: makeId(), text: thinkChunk, done: false, collapsed: false, startedAt: Date.now(), seq: s.nextSeq },
          ],
        };
      }

      case "thinking_end": {
        // Block stays live to accumulate tool calls; closed by next start/answer.
        // Its reported size is banked onto the block so freezing carries it over.
        const tokens = action.tokens;
        const lastIdx = state.liveThinkingBlocks.length - 1;
        if (!tokens || lastIdx < 0) return state;
        return {
          ...state,
          liveThinkingBlocks: state.liveThinkingBlocks.map((b, i) =>
            i === lastIdx ? { ...b, tokens: (b.tokens ?? 0) + tokens } : b
          ),
        };
      }

      case "tool_call": {
        // Tools are tracked in their own stream — independent of thinking.
        const activity: ToolActivity = {
          id: action.id,
          name: action.name,
          icon: iconForTool(action.name),
          label: action.label,
          detail: action.detail,
          status: "running",
          divertible: action.divertible,
          // Present for an exec-shaped call: the command, with no output yet. It
          // opens the terminal panel on IN alone, and tool_result completes it.
          exec: action.exec,
          startedAt: Date.now(),
          seq: state.nextSeq,
        };
        // A tool ran → the next token opens a fresh streaming bubble.
        return {
          ...state,
          nextSeq: state.nextSeq + 1,
          liveToolCalls: [...state.liveToolCalls, activity],
          toolCallAfterToken: true,
        };
      }

      case "tool_result": {
        const { id, ok, summary, error, exec, math, target, duration_ms } = action;
        const patch = (t: ToolActivity): ToolActivity =>
          t.id === id
            ? {
                ...t,
                // A detached run has not succeeded — it has left. Its own ending
                // arrives later, on job_complete.
                status: exec?.running
                  ? ("background" as const)
                  : ok
                  ? ("ok" as const)
                  : ("error" as const),
                summary: exec?.running ? "moved to background" : summary,
                // Fall back to the (clipped) summary so a failed row is always
                // expandable, even when the server sent no full error body.
                error: ok ? undefined : error || summary || "The tool call failed.",
                // A result with no exec payload (a failure, or a tool that turned out
                // not to be exec-shaped) must not erase the IN pane the call opened:
                // the command that ran is the row's only trace of what was attempted.
                exec: exec ?? t.exec,
                math,
                // Only a success names a file worth opening.
                target: ok ? target : undefined,
                // The run is over, whatever it was last seen doing. Without this a
                // detached row keeps "building…" and its bar for the rest of the
                // session, describing a moment that has passed.
                phase: undefined,
                percent: undefined,
                durationMs: duration_ms,
              }
            : t;
        const now = Date.now();
        // A delegating call's result also settles whatever its sub-agent left running.
        return (
          patchToolById(state, id, (tools) => settleFamily(tools.map(patch), id, now)) ??
          state
        );
      }

      case "verdict": {
        // Lands on the row of the run it judges, which may already be frozen into a
        // "tools" message by the time the model gets around to judging it.
        const { id: vId, verdict } = action;
        return (
          patchToolById(state, vId, (tools) =>
            tools.map((t) => (t.id === vId ? { ...t, verdict } : t))
          ) ?? state
        );
      }

      case "tool_progress": {
        // What the run is doing, while it is still doing it. Only a running row can
        // be mid-anything: a settled one has an outcome, and relabelling it
        // "building…" would describe work that is over.
        const { id: pId, phase, percent } = action;
        return (
          patchToolById(state, pId, (tools) =>
            tools.map((t) =>
              t.id === pId && t.status === "running"
                ? { ...t, phase, percent: typeof percent === "number" ? percent : undefined }
                : t
            )
          ) ?? state
        );
      }

      case "tool_backgrounded": {
        // The row is done, the work is not. Marked here rather than inferred from the
        // result, because only the client knows whether a watcher actually took the
        // job — and a row that claimed to be tracked by one that declined would wait
        // for an ending nobody was going to report.
        const { id: bId, job_key } = action;
        return (
          patchToolById(state, bId, (tools) =>
            tools.map((t) =>
              t.id === bId
                ? { ...t, jobKey: job_key, status: "background" as const,
                    summary: t.summary || "moved to background" }
                : t
            )
          ) ?? state
        );
      }

      case "job_progress": {
        // Same fact as tool_progress, from the other side of the hand-off: by now the
        // row has settled, so the job key is what finds it.
        const { job_key: jKey, phase: jPhase, percent: jPercent } = action;
        return (
          patchToolWhere(
            state,
            (t) => t.jobKey === jKey || t.exec?.job_key === jKey,
            (tools) =>
              tools.map((t) =>
                t.jobKey === jKey || t.exec?.job_key === jKey
                  ? { ...t, phase: jPhase,
                      percent: typeof jPercent === "number" ? jPercent : undefined }
                  : t
              )
          ) ?? state
        );
      }

      case "subagent_event": {
        // A sub-agent's step, shown as an ordinary tool row under the call that
        // spawned it — same spinner, same timer, same freezing. Only the indent and
        // the origin badge say it came from somewhere else.
        const parentId = action.parent_id;
        if (action.kind === "tool_call") {
          return (
            patchToolById(state, parentId, (tools) => {
              // The child inherits its parent's stamp: it is spliced in under the
              // call that spawned it, and a stamp of its own could put a reasoning
              // block that arrived meanwhile between the two, splitting the family
              // across two cards.
              const parentSeq = tools.find((t) => t.id === parentId)?.seq;
              const child: ToolActivity = {
                id: action.id ?? `${parentId}:?`,
                name: action.name ?? "",
                icon: iconForTool(action.name ?? ""),
                label: action.label ?? "",
                detail: action.detail ?? "",
                status: "running",
                startedAt: Date.now(),
                parentId,
                seq: parentSeq,
              };
              // A step of its own supersedes the heartbeat: the row now says what
              // the child is actually doing.
              const cleared = tools.map((t) =>
                t.id === parentId ? { ...t, waiting: undefined } : t
              );
              return relabelOrigins(insertAfterFamily(cleared, parentId, child));
            }) ?? state
          );
        }
        if (action.kind === "tool_result") {
          const childId = action.id ?? "";
          return (
            patchToolById(state, childId, (tools) =>
              tools.map((t) =>
                t.id === childId
                  ? {
                      ...t,
                      status: action.ok ? ("ok" as const) : ("error" as const),
                      summary: action.summary,
                      target: action.ok ? action.target ?? undefined : undefined,
                      durationMs: action.duration_ms,
                    }
                  : t
              )
            ) ?? state
          );
        }
        if (action.kind === "heartbeat") {
          // No row of its own: the child is thinking, and the only thing worth saying
          // is that its parent has not stalled. Cleared by the next child step.
          const secs = action.waiting_secs ?? 0;
          return (
            patchToolById(state, parentId, (tools) =>
              tools.map((t) =>
                t.id === parentId ? { ...t, waiting: `working — ${secs}s` } : t
              )
            ) ?? state
          );
        }
        // "end": how much of the child's activity was shed rather than shown.
        return (
          patchToolById(state, parentId, (tools) =>
            tools.map((t) =>
              t.id === parentId ? { ...t, childrenDropped: action.dropped } : t
            )
          ) ?? state
        );
      }

      case "diff": {
        const { file: dFile, patch: dPatch } = action;
        const isNewFile = !dPatch || dPatch.includes("/dev/null");
        // The emitter diffs against the turn's baseline snapshot, so each event
        // already carries the file's full change: coalesce by path instead of
        // stacking, or a file edited N times shows N near-identical cards.
        const pending = [...state.pendingDiffs];
        const pi = pending.findIndex((d) => d.file === dFile);
        if (pi >= 0) pending[pi] = { ...pending[pi], patch: dPatch };
        else pending.push({ file: dFile, patch: dPatch });
        const s = commitDraft({ ...state, pendingDiffs: pending }, makeId);
        const liveIdx = s.messages.findIndex((m) => m.kind === "editing" && m.live);
        let messages: ChatMessage[];
        if (liveIdx >= 0) {
          messages = s.messages.map((m, i) => {
            if (i !== liveIdx) return m;
            const diffs = [...(m.diffs ?? [])];
            const fi = diffs.findIndex((d) => d.file === dFile);
            if (fi >= 0) {
              // is_new is sticky: a file created then edited is still a creation.
              diffs[fi] = { ...diffs[fi], patch: dPatch, is_new: diffs[fi].is_new || isNewFile };
            } else {
              diffs.push({ file: dFile, patch: dPatch, is_new: isNewFile });
            }
            return { ...m, diffs };
          });
        } else {
          messages = appendStamped(s.messages, {
            id: makeId(),
            seq: s.nextSeq,
            role: "agent",
            kind: "editing",
            live: true,
            diffs: [{ file: dFile, patch: dPatch, is_new: isNewFile }],
          });
          return { ...s, nextSeq: s.nextSeq + 1, messages };
        }
        return { ...s, messages };
      }

      case "answer": {
        // The draft is discarded rather than committed: `action.text` is the same
        // turn, authoritative and complete (it carries the verification ledger).
        let s = flushLive({ ...state, draft: "", draftSeq: null, toolCallAfterToken: false }, makeId);
        const diffs = [...s.pendingDiffs];
        const thinkingText = s.pendingThinking;
        s = { ...s, pendingDiffs: [], pendingThinking: "", busy: false };
        // Dropped alongside the unanswered approval cards: a provisional bubble the
        // answer turns out to be a fuller copy of. Only *that* one — a plan the user
        // approved is followed by the report of executing it, and a plan they rejected
        // by the refusal, neither of which repeats the plan; dropping those would take
        // the plan out of a transcript the user watched it arrive in. What does repeat
        // is a plan nobody was asked about (delivered as the answer) and a turn cut off
        // mid-prose that later finished, whose bubble is a prefix of the answer.
        // Whitespace-insensitive, and a comparison that fails leaves both: a visible
        // duplicate is recoverable, a silently deleted answer is not.
        const noApproval = s.messages.filter(
          (m) => !(m.kind === "approval" && !m.text)
            && !(m.provisional && isRewordedBy(action.text, m.text))
        );
        const finalized = noApproval.map((m) =>
          m.kind === "editing" && m.live ? { ...m, live: false } : m
        );
        const hasEditCard = finalized.some((m) => m.kind === "editing");
        const attachDiffs = !hasEditCard && diffs.length > 0 ? diffs : undefined;
        const messages: ChatMessage[] = [
          ...finalized,
          {
            id: makeId(),
            role: "agent",
            kind: "text",
            text: action.text,
            diffs: attachDiffs,
            thinking: thinkingText || undefined,
          },
        ];
        return { ...s, messages };
      }

      case "error": {
        const s = flushLive({ ...state, toolCallAfterToken: false }, makeId);
        const messages: ChatMessage[] = [
          ...s.messages.map((m) =>
            m.kind === "editing" && m.live ? { ...m, live: false } : m
          ),
          { id: makeId(), role: "agent", kind: "error", text: action.text },
        ];
        return { ...s, busy: false, messages };
      }

      // Server messages handled outside the reducer (ready/config/todo/
      // todo_prompt/batch_status/sessions_list/session_loaded/context_*) and
      // any future additions fall through unchanged.
      default:
        return state;
    }
  };
}
