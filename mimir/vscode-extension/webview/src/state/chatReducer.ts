import type { ChatMessage, DiffEntry, ServerMessage, ToolActivity } from "../types";
import {
  insertAfterFamily, relabelOrigins, settleFamily,
} from "../components/subAgentUtils";

// A reasoning block — reasoning TEXT only. Tool calls are tracked separately
// (see `liveToolCalls`); thinking and tools are fully decorrelated.
export type ThinkingBlock = {
  id: string;
  text: string;
  done: boolean;
  collapsed: boolean;
  /** Epoch ms when this block opened — drives the live timer and frozen duration. */
  startedAt: number;
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
  if (name.startsWith("ft_")) return "🧪";
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
 * card mid-flight runs freezePending on a still-running row — so every patch has to
 * look in both places. Returns null when the row is nowhere: an event for a turn the
 * user has already cleared is dropped, not applied to a stranger.
 */
function patchToolById(
  state: ChatState,
  id: string,
  transform: (tools: ToolActivity[]) => ToolActivity[],
): ChatState | null {
  if (state.liveToolCalls.some((t) => t.id === id)) {
    return { ...state, liveToolCalls: transform(state.liveToolCalls) };
  }
  const idx = state.messages.findIndex(
    (m) => m.kind === "tools" && (m.tools ?? []).some((t) => t.id === id)
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

/**
 * Bake the current turn's live state into the message stream and clear it.
 * Thinking and tools are baked as SEPARATE, decorrelated messages:
 *   - each non-empty reasoning block → a `kind:"thinking"` message (text only);
 *   - the accumulated tool calls → one `kind:"tools"` message.
 */
function freezePending(state: ChatState, makeId: () => string): ChatState {
  const blocksToFreeze = state.liveThinkingBlocks.filter((b) => b.text.trim().length > 0);
  const tools = state.liveToolCalls;
  if (blocksToFreeze.length === 0 && tools.length === 0) {
    if (state.liveThinkingBlocks.length === 0 && state.liveToolCalls.length === 0) return state;
    return { ...state, liveThinkingBlocks: [], liveToolCalls: [] };
  }
  let messages = state.messages;
  if (blocksToFreeze.length > 0) {
    const frozenAt = Date.now();
    messages = [
      ...messages,
      ...blocksToFreeze.map((b) => ({
        id: b.id,
        role: "agent" as const,
        kind: "thinking" as const,
        text: b.text,
        live: false, // start collapsed — user can click to expand
        thinkingDurationMs: Math.max(0, frozenAt - b.startedAt),
        thinkingTokens: b.tokens,
      })),
    ];
  }
  if (tools.length > 0) {
    // Any tool still "running" at freeze time never received a result; stamp a
    // final duration and mark it done so its live timer stops ticking instead
    // of counting up forever in the frozen card.
    const frozenAt = Date.now();
    const frozenTools = tools.map((t) =>
      t.status === "running"
        ? { ...t, status: "ok" as const, durationMs: t.durationMs ?? Math.max(0, frozenAt - t.startedAt) }
        : t
    );
    messages = [
      ...messages,
      { id: makeId(), role: "agent" as const, kind: "tools" as const, tools: frozenTools, live: false },
    ];
  }
  return { ...state, messages, liveThinkingBlocks: [], liveToolCalls: [] };
}

/**
 * Move the draft into the transcript, if there is anything in it.
 *
 * Called before anything else is appended below it — a card, a frozen tool list,
 * an error — so committed prose keeps its place in the order it was written.
 * NOT called on `answer` (the authoritative text supersedes the draft) or on
 * `nudge_injected` (the turn was not accepted, so there is nothing to keep).
 */
function commitDraft(state: ChatState, makeId: () => string): ChatState {
  if (!state.draft.trim()) return state.draft ? { ...state, draft: "" } : state;
  return {
    ...state,
    draft: "",
    messages: [
      ...state.messages,
      { id: makeId(), role: "agent", kind: "text", text: state.draft },
    ],
  };
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
          messages: [...state.messages, userMsg],
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
        const userMsg: ChatMessage = {
          id: makeId(), role: "user", kind: "text", text: action.text, queued: true,
        };
        return { ...state, messages: [...state.messages, userMsg] };
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
        return state.draft ? { ...state, draft: "" } : state;

      case "reset":
        return initialChatState;

      case "connection_lost": {
        // Keep whatever prose had arrived: the turn was cut off, not superseded.
        const s = commitDraft(state, makeId);
        return { ...s, busy: false, liveThinkingBlocks: [], liveToolCalls: [] };
      }

      case "session_loaded_messages":
        return {
          ...state,
          messages: action.messages,
          draft: "",
          liveThinkingBlocks: [],
          liveToolCalls: [],
          busy: false,
          pendingDiffs: [],
          pendingThinking: "",
        };

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
            s = freezePending(commitDraft(s, makeId), makeId);
          } else if (s.draft && !/\s$/.test(s.draft) && !/^\s/.test(delta)) {
            s = { ...s, draft: s.draft + "\n\n" };
          }
        }
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
          messages = [
            ...s0.messages,
            { id: makeId(), role: "agent", kind: "editing", live: true, diffs: progressDiffs },
          ];
        }
        return { ...s0, messages };
      }

      case "approval": {
        let s = freezePending(commitDraft({ ...state, toolCallAfterToken: false }, makeId), makeId);
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
              ? { ...m, approval: { ...action, ids: [...existingIds, action.id] } }
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
          liveThinkingBlocks: [
            ...s.liveThinkingBlocks,
            { id: makeId(), text: "", done: false, collapsed: false, startedAt: Date.now() },
          ],
        };
      }

      case "thinking": {
        const thinkChunk = action.text;
        const s = { ...state, pendingThinking: state.pendingThinking + thinkChunk };
        if (s.liveThinkingBlocks.length === 0) {
          return {
            ...s,
            liveThinkingBlocks: [{ id: makeId(), text: thinkChunk, done: false, collapsed: false, startedAt: Date.now() }],
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
          liveThinkingBlocks: [
            ...s.liveThinkingBlocks,
            { id: makeId(), text: thinkChunk, done: false, collapsed: false, startedAt: Date.now() },
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
          startedAt: Date.now(),
        };
        // A tool ran → the next token opens a fresh streaming bubble.
        return {
          ...state,
          liveToolCalls: [...state.liveToolCalls, activity],
          toolCallAfterToken: true,
        };
      }

      case "tool_result": {
        const { id, ok, summary, error, exec, duration_ms } = action;
        const patch = (t: ToolActivity): ToolActivity =>
          t.id === id
            ? {
                ...t,
                status: ok ? ("ok" as const) : ("error" as const),
                summary,
                // Fall back to the (clipped) summary so a failed row is always
                // expandable, even when the server sent no full error body.
                error: ok ? undefined : error || summary || "The tool call failed.",
                exec,
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

      case "subagent_event": {
        // A sub-agent's step, shown as an ordinary tool row under the call that
        // spawned it — same spinner, same timer, same freezing. Only the indent and
        // the origin badge say it came from somewhere else.
        const parentId = action.parent_id;
        if (action.kind === "tool_call") {
          return (
            patchToolById(state, parentId, (tools) => {
              const child: ToolActivity = {
                id: action.id ?? `${parentId}:?`,
                name: action.name ?? "",
                icon: iconForTool(action.name ?? ""),
                label: action.label ?? "",
                detail: action.detail ?? "",
                status: "running",
                startedAt: Date.now(),
                parentId,
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
          messages = [
            ...s.messages,
            {
              id: makeId(),
              role: "agent",
              kind: "editing",
              live: true,
              diffs: [{ file: dFile, patch: dPatch, is_new: isNewFile }],
            },
          ];
        }
        return { ...s, messages };
      }

      case "answer": {
        // The draft is discarded rather than committed: `action.text` is the same
        // turn, authoritative and complete (it carries the verification ledger).
        let s = freezePending({ ...state, draft: "", toolCallAfterToken: false }, makeId);
        const diffs = [...s.pendingDiffs];
        const thinkingText = s.pendingThinking;
        s = { ...s, pendingDiffs: [], pendingThinking: "", busy: false };
        const noApproval = s.messages.filter((m) => !(m.kind === "approval" && !m.text));
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
        const s = freezePending(commitDraft({ ...state, toolCallAfterToken: false }, makeId), makeId);
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
