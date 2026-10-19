// The turn's life: a question asked or steered, the answer or error that ends it, and
// the moments it parks, resumes or loses its connection.

import type { ChatMessage } from "../../types";
import { initialChatState, type ChatHandlers } from "../chatState";
import { appendStamped, appendAboveUnclaimed, flushLive, commitDraft, isRewordedBy, clearProvisional, parkOnPrompt } from "./transcript";

export const TURN_HANDLERS: ChatHandlers = {
  submit_query: (state, action, makeId) => {
    // Committed, not dropped. Anything still live when a new question is asked
    // happened, and the live buffers are a staging area rather than a record: a
    // replayed turn whose cards and prose are sitting there — the ordinary state
    // after reattaching to a run that has not answered yet — would otherwise be
    // erased by the next thing typed, leaving the question where the work was.
    const flushed = flushLive(state, makeId);
    const userMsg: ChatMessage = { id: makeId(), role: "user", kind: "text", text: action.text };
    return {
      ...flushed,
      // Whatever a previous turn left provisional is settled by a new question
      // being asked: it is the transcript now, and the `answer` this turn ends on
      // must supersede its own prose only — never reach back and delete an older
      // turn's, which is all that would be left of a run that was interrupted.
      messages: [...clearProvisional(flushed.messages), userMsg],
      liveThinkingBlocks: [],
      liveToolCalls: [],
      busy: true,
      toolCallAfterToken: false,
    };
  },

  // A message typed while the agent is busy. Append it as a user bubble
  // tagged "queued" (awaiting injection) without disturbing the running
  // turn's live state — busy stays true; the run never stopped.
  steer_query: (state, action, makeId) => {
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
  },

  // A queued steer reached the agent: clear the tag on the oldest queued bubble.
  // `starts_turn`: it arrived after the run ended and is a turn of its own, so `busy`
  // is set here. While tagged, the bubble stays below whatever ends the turn.
  steer_injected: (state, action) => {
    const busy = action.starts_turn ? true : state.busy;
    const idx = state.messages.findIndex((m) => m.queued);
    if (idx < 0) return busy === state.busy ? state : { ...state, busy };
    return {
      ...state,
      busy,
      messages: state.messages.map((m, i) =>
        i === idx ? { ...m, queued: false } : m
      ),
    };
  },

  // A nudge fired: the loop did not accept the turn the model just wrote. The nudge
  // is not rendered (it is not the user's), and the draft is dropped; it never
  // entered the transcript.
  nudge_injected: (state) => {
    return state.draft ? { ...state, draft: "", draftSeq: null } : state;
  },

  // Every agent slot is taken. Rendered rather than dropped like other transient
  // status, and deliberately leaving `busy` set: the turn was accepted and will
  // run, so the chat must keep reading as working rather than invite a second send.
  queued: (state, action, makeId) => {
    const s = flushLive({ ...state, toolCallAfterToken: false }, makeId);
    return {
      ...s,
      messages: [
        ...s.messages,
        { id: makeId(), role: "agent", kind: "text", text: action.text },
      ],
    };
  },

  answer: (state, action, makeId) => {
    // The draft is discarded rather than committed: `action.text` is the same
    // turn, authoritative and complete (it carries the verification ledger).
    let s = flushLive({ ...state, draft: "", draftSeq: null, toolCallAfterToken: false }, makeId);
    const diffs = [...s.pendingDiffs];
    const thinkingText = s.pendingThinking;
    s = { ...s, pendingDiffs: [], pendingThinking: "", busy: false };
    // Dropped with the unanswered approval cards: a provisional bubble the answer is a
    // fuller copy of (a plan nobody was asked about, a turn cut off mid-prose). A
    // failed comparison keeps both: a duplicate is recoverable, a lost answer is not.
    const noApproval = s.messages.filter(
      (m) => !(m.kind === "approval" && !m.text)
        && !(m.provisional && isRewordedBy(action.text, m.text))
    );
    const finalized = noApproval.map((m) =>
      m.kind === "editing" && m.live ? { ...m, live: false } : m
    );
    const hasEditCard = finalized.some((m) => m.kind === "editing");
    const attachDiffs = !hasEditCard && diffs.length > 0 ? diffs : undefined;
    const messages = appendAboveUnclaimed(finalized, {
      id: makeId(),
      role: "agent",
      kind: "text",
      text: action.text,
      diffs: attachDiffs,
      thinking: thinkingText || undefined,
    });
    return { ...s, messages };
  },

  error: (state, action, makeId) => {
    const s = flushLive({ ...state, toolCallAfterToken: false }, makeId);
    const messages: ChatMessage[] = appendAboveUnclaimed(
      s.messages.map((m) =>
        m.kind === "editing" && m.live ? { ...m, live: false } : m
      ),
      { id: makeId(), role: "agent", kind: "error", text: action.text },
    );
    return { ...s, busy: false, messages };
  },

  reset: () => initialChatState,

  connection_lost: (state, _action, makeId) => {
    // Keep whatever prose had arrived: the turn was cut off, not superseded.
    // Provisional, because the turn on the other end is not necessarily dead —
    // the worker runs on, and a reconnect can still be handed the `answer` it
    // ends on, which must replace this partial rather than repeat under it.
    const s = commitDraft(state, makeId, true);
    return { ...s, busy: false, liveThinkingBlocks: [], liveToolCalls: [] };
  },

  // A turn of this session outlived the connection and is still producing, so
  // the run is live even though this window did not start it. Dispatched after
  // session_loaded_messages, which clears the state a reload arrives with; a card
  // the turn is parked on lands after this and parks it again.
  turn_resumed: (state) => {
    return state.busy ? state : { ...state, busy: true };
  },

  // The turn parked on a question for the user: a plan awaiting approval or a
  // clarification batch. Rendering the card is App's business; what belongs
  // here is that the turn stopped producing.
  //
  // Unless the card came from somewhere else. A sub-agent runs alongside the turn,
  // not inside it, so its question says nothing about whether the turn is still
  // producing — parking on it would show the agent as idle while it works.
  user_question: (state, action, makeId) => {
    return action.origin ? state : parkOnPrompt(state, makeId);
  },

  // The card a turn parked on was answered. The turn goes back to producing —
  // the same tail as `approval_response`, and the same reason: `busy` is what
  // offers the stop button, and what makes the end of the turn observable at
  // all, which is when the finished transcript is handed back for saving.
  // `resumes` is false for an answer that ends the run instead, where claiming
  // a live turn would strand the composer.
  prompt_answered: (state, action) => {
    return action.resumes ? { ...state, busy: true } : state;
  },
};
