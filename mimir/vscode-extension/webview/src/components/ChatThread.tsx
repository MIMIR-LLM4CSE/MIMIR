import React, { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { ChatMessage, ToolActivity } from "../types";
import type { ThinkingBlock } from "../state/chatReducer";
import { ChatMessage as ChatMessageView } from "./ChatMessage";
import { MarkdownContent } from "./MarkdownContent";
import { ToolActivityList } from "./ToolActivityList";
import { ThinkingPanel } from "./ThinkingPanel";
import { GlobalApprovalBar } from "./GlobalApprovalBar";
import { StreamingStatus } from "./StreamingStatus";
import { orderLiveStream } from "./liveStreamUtils";

/** Messages rendered at first, and added each time the reader reaches the top. */
export const THREAD_WINDOW_STEP = 120;
/** How close to the top, in px, counts as reaching it. */
const GROW_THRESHOLD_PX = 300;

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  /** The turn waits on its agent being built, not on the model. */
  starting?: boolean;
  /** Prose of the turn in flight. Rendered below the transcript, never in it:
   *  the loop may still send the model back to work, and a draft that lived in
   *  the transcript would then have to be deleted out of it. */
  draft?: string;
  /** Arrival stamp of the draft's first token — places the prose among this
   *  step's cards. Null while there is no prose. */
  draftSeq?: number | null;
  /** Tool calls for the current (in-flight) step, not yet frozen into a message. */
  liveToolCalls?: ToolActivity[];
  /** Reasoning blocks for the current (in-flight) step, not yet frozen. */
  liveThinkingBlocks?: ThinkingBlock[];
  /** Rendered in place of the message list when there are no messages. */
  emptyState?: React.ReactNode;
  /** When true, a spinner replaces the thread (e.g. while a session loads). */
  loading?: boolean;
  chatThreadRef: React.RefObject<HTMLDivElement>;
  bottomRef: React.RefObject<HTMLDivElement>;
  lastUserMsgRef: React.RefObject<HTMLDivElement>;
  onScroll: () => void;
  onApprovalResponse: (id: string, choice: "y" | "n" | "a", approvedFiles?: string[]) => void;
  /** Re-runs the last turn; surfaced as a Retry button on the latest error card. */
  onRetry?: () => void;
  /** Detach a still-running tool call. Offered on live rows only — a frozen row
   *  has already ended, and there is nothing left to move. */
  onDivert?: (id: string) => void;
}

/**
 * The scrollable conversation pane: messages by kind, frozen tool and reasoning cards,
 * and the live ones at the bottom until they freeze. Purely presentational.
 */
export const ChatThread: React.FC<Props> = ({
  messages,
  busy,
  starting = false,
  draft = "",
  draftSeq = null,
  liveToolCalls = [],
  liveThinkingBlocks = [],
  emptyState,
  loading,
  chatThreadRef,
  bottomRef,
  lastUserMsgRef,
  onScroll,
  onApprovalResponse,
  onRetry,
  onDivert,
}) => {
  // All message kinds render via renderMessage; thinking blocks (frozen) and
  // tool lists each have their own render branch in ChatMessage.
  const visible = messages;

  const lastIdx = visible.length - 1;
  let lastUserIdx = -1;
  for (let i = lastIdx; i >= 0; i--) {
    if (visible[i].role === "user") { lastUserIdx = i; break; }
  }

  const liveBlocks = liveThinkingBlocks.filter((b) => b.text.trim().length > 0);
  const liveDraftSeq = draft.trim().length > 0 ? draftSeq ?? 0 : null;

  // A message can land while the step is still in flight — a steer bubble, an edit
  // card, a session-command reply. It sits at the tail of the transcript, stamped,
  // and is drawn inside the live area at its stamp: rendering the whole transcript
  // above the live area put a steer bubble above the prose it interrupted.
  const liveFloor = Math.min(
    liveDraftSeq ?? Infinity,
    ...liveBlocks.map((b) => b.seq),
    ...liveToolCalls.map((t) => t.seq ?? Infinity),
  );
  let settledCount = visible.length;
  while (settledCount > 0) {
    const seq = visible[settledCount - 1].seq;
    if (seq === undefined || seq <= liveFloor) break;
    settledCount--;
  }
  const settled = visible.slice(0, settledCount);
  const duringStep = visible.slice(settledCount);

  // Only the tail of a long history is rendered: a session of thousands of rows cost
  // every render and every relayout. The window starts at the last STEP messages of
  // the conversation on screen, and opening another conversation starts it again.
  const firstId = visible[0]?.id;
  const [win, setWin] = useState<{ first: string | undefined; start: number }>(
    () => ({ first: firstId, start: Math.max(0, settledCount - THREAD_WINDOW_STEP) }));
  let start = win.start;
  if (win.first !== firstId) {
    start = Math.max(0, settledCount - THREAD_WINDOW_STEP);
    setWin({ first: firstId, start });
  }
  start = Math.min(start, settledCount);

  // Where the reader was when the window grew, so the rows they were looking at stay
  // put once the older ones are inserted above them.
  const anchorRef = useRef<{ height: number; top: number } | null>(null);
  const showEarlier = useCallback(() => {
    const el = chatThreadRef.current;
    if (!el || anchorRef.current) return;
    anchorRef.current = { height: el.scrollHeight, top: el.scrollTop };
    setWin((w) => ({ ...w, start: Math.max(0, w.start - THREAD_WINDOW_STEP) }));
  }, [chatThreadRef]);

  useLayoutEffect(() => {
    const el = chatThreadRef.current;
    const anchor = anchorRef.current;
    if (!el || !anchor) return;
    anchorRef.current = null;
    el.scrollTop = anchor.top + (el.scrollHeight - anchor.height);
  }, [start, chatThreadRef]);

  const handleScroll = useCallback(() => {
    onScroll();
    const el = chatThreadRef.current;
    if (el && start > 0 && el.scrollTop < GROW_THRESHOLD_PX) showEarlier();
  }, [onScroll, chatThreadRef, start, showEarlier]);

  const liveEntries = orderLiveStream({
    thinking: liveBlocks,
    tools: liveToolCalls,
    draftSeq: liveDraftSeq,
    messages: duringStep,
  });

  // One animated status line, pinned to the bottom of the thread for as long as
  // the agent is busy — whatever else is on screen. It is the only place the
  // status word shows, so the draft and the live reasoning line carry none.
  const showStatusLine = busy;

  // ── One message → one render path, dispatched by kind ──────────────────────
  const renderMessage = (msg: ChatMessage, idx: number) => {
    if (msg.kind === "approval" && msg.approval) {
      return (
        <div className="inline-approval-wrap" key={msg.id}>
          <GlobalApprovalBar
            approval={msg.approval}
            onRespond={(choice) => onApprovalResponse(msg.approval!.id, choice)}
          />
        </div>
      );
    }

    // Skip approval shells that carry no content.
    if (msg.kind === "approval") return null;

    const isLastUser = idx === lastUserIdx;
    const isLatestError = msg.kind === "error" && idx === lastIdx;

    return (
      <div key={msg.id} ref={isLastUser ? lastUserMsgRef : undefined}>
        <ChatMessageView
          message={msg}
          onApprovalResponse={onApprovalResponse}
          onRetry={isLatestError ? onRetry : undefined}
        />
      </div>
    );
  };

  return (
    <div
      className="chat-thread"
      ref={chatThreadRef}
      onScroll={handleScroll}
      role="log"
      aria-live="polite"
      aria-label="Conversation"
    >
      {visible.length === 0 ? (
        emptyState
      ) : loading ? (
        <div className="session-loading">
          <span className="inline-spinner" aria-hidden="true" />
          Loading session…
        </div>
      ) : (
        <>
          {start > 0 && (
            <button className="thread-more" onClick={showEarlier}>
              Show earlier messages ({start} hidden)
            </button>
          )}
          {settled.slice(start).map((m, i) => renderMessage(m, start + i))}

          {/* The step in flight, in the order it arrived: prose, reasoning and
              tool rows are three separate streams, and orderLiveStream reads
              their arrival stamps back into one list. The freeze goes through
              the same function, so nothing moves when the step ends. */}
          {liveEntries.map((entry) => {
            if (entry.kind === "message") {
              return renderMessage(entry.message, visible.indexOf(entry.message));
            }
            if (entry.kind === "thinking") {
              return (
                <ThinkingPanel
                  key={entry.block.id}
                  text={entry.block.text}
                  live
                  startedAt={entry.block.startedAt}
                />
              );
            }
            if (entry.kind === "tools") {
              return (
                <ToolActivityList
                  key={`tools-${entry.seq}`}
                  tools={entry.tools}
                  onDivert={onDivert}
                />
              );
            }
            // The turn in flight. It joins the transcript only once the loop
            // accepts it; until then it lives here, where clearing it reads as
            // "still working" rather than as an answer being taken away.
            return (
              <div className="chat-message agent chat-draft" key="draft">
                <div className="message-body">
                  <MarkdownContent text={draft} />
                </div>
              </div>
            );
          })}

          {/* Animated status line — pinned to the bottom while the agent is busy */}
          {showStatusLine && (
            <div className="agent-thinking">
              <StreamingStatus
                label={starting ? "Starting MIMIR — model backend and MCP servers" : undefined}
              />
            </div>
          )}
        </>
      )}
      <div ref={bottomRef} />
    </div>
  );
};
