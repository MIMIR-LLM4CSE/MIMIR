// Where the chat pane is scrolled: following the bottom as the thread grows, a new
// question anchored at the top, an approval forced into view, and a run brought back
// from the dock.

import { useCallback, useLayoutEffect, useMemo, useRef } from "react";
import type { ChatState } from "../state/chatReducer";
import { runsInFlight } from "../components/runDockUtils";
import { useOffscreenRows } from "./useOffscreenRows";
import { useStickToBottom } from "./useStickToBottom";

type Thread = Pick<ChatState, "messages" | "draft" | "liveToolCalls" | "liveThinkingBlocks">;

export function useThreadScroll({ messages, draft, liveToolCalls, liveThinkingBlocks }: Thread) {
  // The pane follows the bottom of the thread as it grows, and lets go when the
  // reader scrolls up. useStickToBottom owns that decision — see there for why it
  // cannot be taken from the scroll position alone.
  const {
    ref: chatThreadRef,
    stickRef: followBottomRef,
    scrollToBottom,
    follow: forceFollowBottom,
    release: releaseFollowBottom,
    onScroll,
  } = useStickToBottom<HTMLDivElement>();
  const bottomRef = useRef<HTMLDivElement>(null);
  // The last user message, which a new question scrolls to the top.
  const lastUserMsgRef = useRef<HTMLDivElement>(null);
  // Set on submit so the bottom-follow effect yields the next commit to the
  // top-anchoring of the new question.
  const pendingScrollToTopRef = useRef(false);

  const scrollQueryToTop = useCallback(() => {
    requestAnimationFrame(() => {
      const el = lastUserMsgRef.current;
      const container = chatThreadRef.current;
      if (!el || !container) return;
      container.scrollTop = el.offsetTop - container.offsetTop - 8;
    });
  }, [chatThreadRef]);

  /** A new question: follow again, and anchor it at the top rather than the bottom. */
  const anchorNewQuery = useCallback(() => {
    followBottomRef.current = true;
    pendingScrollToTopRef.current = true;
    scrollQueryToTop();
  }, [followBottomRef, scrollQueryToTop]);

  // Keep the view pinned to the bottom as the thread grows. After the DOM has
  // committed, unlike the synchronous scrollToBottom() calls in the message handler,
  // which read the scrollHeight from before React rendered the new content. It honours
  // the follow flag, so a reader who scrolled up is left undisturbed.
  useLayoutEffect(() => {
    // A fresh submit anchors the new question at the top, so this one commit skips
    // the bottom-follow to avoid a bottom→top jump.
    if (pendingScrollToTopRef.current) {
      pendingScrollToTopRef.current = false;
      return;
    }
    scrollToBottom();
  }, [messages, draft, liveToolCalls, liveThinkingBlocks, scrollToBottom]);

  // An approval forces the bottom into view, so the live editing card is seen even
  // when the reader had scrolled up.
  const scrollToApproval = useCallback(() => {
    setTimeout(forceFollowBottom, 80);
  }, [forceFollowBottom]);

  // Runs still going, and which of their rows have scrolled out of sight. A run can
  // outlast the part of the conversation that started it, so the dock keeps the ones
  // the reader can no longer see. `messages.length` re-runs the search when a live row
  // freezes into a message, which replaces its element.
  const inFlightRuns = useMemo(
    () => runsInFlight(messages, liveToolCalls),
    [messages, liveToolCalls],
  );
  const offscreenRuns = useOffscreenRows(
    useMemo(() => inFlightRuns.map((r) => r.id), [inFlightRuns]),
    chatThreadRef,
    messages.length,
  );
  const dockedRuns = useMemo(
    () => inFlightRuns.filter((r) => offscreenRuns.has(r.id)),
    [inFlightRuns, offscreenRuns],
  );
  const focusRun = useCallback((id: string) => {
    const el = chatThreadRef.current?.querySelector(`[data-tool-id="${CSS.escape(id)}"]`);
    if (!el) return;
    // Let go of the bottom first: the run is still reporting progress, and the next
    // update would otherwise scroll the pane back down before the reader got there.
    // A jump, not a smooth scroll: the first frames of an animation are still near the
    // bottom, and onScroll takes "near the bottom" as a reason to follow again.
    releaseFollowBottom();
    el.scrollIntoView({ behavior: "auto", block: "center" });
  }, [chatThreadRef, releaseFollowBottom]);

  return {
    chatThreadRef, bottomRef, lastUserMsgRef, onScroll,
    scrollToBottom, scrollToApproval, anchorNewQuery,
    dockedRuns, focusRun,
  };
}
