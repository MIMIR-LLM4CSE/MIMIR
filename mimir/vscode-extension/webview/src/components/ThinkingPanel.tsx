import React, { useLayoutEffect } from "react";
import { useElapsed, formatDuration } from "../hooks/useElapsed";
import { useStickToBottom } from "../hooks/useStickToBottom";
import { StreamingStatus } from "./StreamingStatus";

interface Props {
  /** Accumulated reasoning text. */
  text: string;
  /** True while the model is still emitting reasoning tokens. */
  live: boolean;
  /** Epoch ms when reasoning began (drives the live timer). */
  startedAt?: number;
  /** Final reasoning duration for a frozen panel. */
  durationMs?: number;
  /** Reasoning size in tokens, reported by the server when the block closed. */
  tokens?: number;
}

/** Compact token readout: "820 tokens", "1.2k tokens". */
function formatTokens(n: number): string {
  return n < 1000 ? `${n} tokens` : `${(n / 1000).toFixed(1)}k tokens`;
}

/**
 * Renders a model reasoning ("thinking") block as a deliberately secondary
 * element — a dim one-liner, no box, no accent color:
 *   - live  → "◌ <animated word> · 12s" with the reasoning streaming below in
 *             dim italics (reduced height);
 *   - frozen → a collapsed <details> line "✻ Thinking · 3.2s" the user can
 *              re-open. Native <details> handles toggle, so no reducer wiring.
 */
export const ThinkingPanel: React.FC<Props> = ({ text, live, startedAt, durationMs, tokens }) => {
  const elapsed = useElapsed(startedAt ?? Date.now(), live);
  // The reasoning stream follows its own bottom as tokens land, and lets go when
  // the reader scrolls up to re-read. Only a gesture stops the follow: measuring
  // the position when the scroll event arrives is what used to stop it on its own,
  // since by then the tokens that fired it have already made the pane taller.
  const { ref: contentRef, scrollToBottom, onScroll } = useStickToBottom<HTMLPreElement>(24);

  useLayoutEffect(() => {
    if (live) scrollToBottom();
  }, [text, live, scrollToBottom]);

  if (live) {
    return (
      <div className="thinking-block thinking-block--live">
        <div className="thinking-summary">
          <span className="tb-spinner" aria-hidden="true" />
          <StreamingStatus showDots={false} />
          <span className="thinking-duration">· {formatDuration(elapsed)}</span>
        </div>
        <pre className="thinking-content" ref={contentRef} onScroll={onScroll}>{text}</pre>
      </div>
    );
  }

  return (
    <details className="thinking-block">
      <summary className="thinking-summary">
        <span className="thinking-glyph" aria-hidden="true">✻</span>
        <span className="thinking-summary-label">Thinking</span>
        {durationMs !== undefined && (
          <span className="thinking-duration">· {formatDuration(durationMs)}</span>
        )}
        {tokens !== undefined && tokens > 0 && (
          <span className="thinking-duration">· {formatTokens(tokens)}</span>
        )}
      </summary>
      <pre className="thinking-content">{text}</pre>
    </details>
  );
};
