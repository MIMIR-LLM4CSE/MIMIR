import React from "react";
import type { ChatMessage, ToolActivity } from "../types";
import type { ThinkingBlock } from "../state/chatReducer";
import { ChatMessage as ChatMessageView } from "./ChatMessage";
import { MarkdownContent } from "./MarkdownContent";
import { ToolActivityList } from "./ToolActivityList";
import { ThinkingPanel } from "./ThinkingPanel";
import { GlobalApprovalBar } from "./GlobalApprovalBar";
import { StreamingStatus } from "./StreamingStatus";
import { orderLiveStream } from "./liveStreamUtils";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
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
 * Owns the scrollable conversation pane: the message list (dispatched by kind)
 * and the typing indicator. App keeps connection, session and settings
 * orchestration; this component is purely presentational.
 *
 * Frozen `tools` and `thinking` messages are rendered as their own cards
 * (tool-activity list and collapsible reasoning panels). Live, in-flight tool
 * calls and reasoning blocks are rendered at the bottom until they freeze.
 */
export const ChatThread: React.FC<Props> = ({
  messages,
  busy,
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

    const isLastUser =
      msg.role === "user" &&
      !visible.slice(idx + 1).some((m) => m.role === "user");
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
      onScroll={onScroll}
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
          {settled.map(renderMessage)}

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
              <StreamingStatus />
            </div>
          )}
        </>
      )}
      <div ref={bottomRef} />
    </div>
  );
};
