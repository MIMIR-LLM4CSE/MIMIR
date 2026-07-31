import React, { useEffect, useRef } from "react";
import { useElapsed, formatDuration } from "../hooks/useElapsed";
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
  const contentRef = useRef<HTMLPreElement>(null);
  // Whether to keep the stream pinned to the bottom. Starts true, and is only
  // updated by a *user* scroll (onScroll below) — never recomputed after new
  // text lands. The old code measured "am I at the bottom?" AFTER appending the
  // token, so any chunk taller than the 24px threshold made the panel look
  // scrolled-up and it stopped following. Capturing the intent at scroll time
  // instead keeps it glued to the bottom no matter how large the next chunk is.
  const stickRef = useRef(true);

  const handleScroll = () => {
    const el = contentRef.current;
    if (!el) return;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  // Auto-scroll the live reasoning stream to the bottom as new tokens arrive,
  // unless the user has scrolled up to read earlier reasoning (stickRef=false).
  useEffect(() => {
    if (!live) return;
    const el = contentRef.current;
    if (!el || !stickRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [text, live]);

  if (live) {
    return (
      <div className="thinking-block thinking-block--live">
        <div className="thinking-summary">
          <span className="tb-spinner" aria-hidden="true" />
          <StreamingStatus showDots={false} />
          <span className="thinking-duration">· {formatDuration(elapsed)}</span>
        </div>
        <pre className="thinking-content" ref={contentRef} onScroll={handleScroll}>{text}</pre>
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
