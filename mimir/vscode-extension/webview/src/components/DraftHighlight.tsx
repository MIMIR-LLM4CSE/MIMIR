import React, { useEffect, useMemo, useRef } from "react";
import { splitMentionSegments } from "./mentionHighlight";

interface Props {
  /** The draft as the textarea holds it. */
  text: string;
  /** The textarea this paints behind, for its box and its scroll position. */
  inputRef: React.RefObject<HTMLTextAreaElement>;
}

/**
 * The mention tints inside the chat input.
 *
 * A textarea cannot colour its text, and must stay one (the dropdowns, history and
 * steers all work on `value` + `selectionStart`), so the text is drawn again underneath
 * with the same metrics, taken from the textarea itself. Nothing renders without a mention.
 */
export function DraftHighlight({ text, inputRef }: Props) {
  const segments = useMemo(() => splitMentionSegments(text), [text]);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const hasMention = segments.some((s) => s.kind === "mention");

  // Follow the textarea's box and its scroll, so the tints stay under their words.
  useEffect(() => {
    const input = inputRef.current;
    const mirror = mirrorRef.current;
    if (!input || !mirror) return;
    const sync = () => {
      mirror.style.width = `${input.clientWidth}px`;
      mirror.style.height = `${input.clientHeight}px`;
      mirror.scrollTop = input.scrollTop;
    };
    sync();
    input.addEventListener("scroll", sync);
    const observer = new ResizeObserver(sync);
    observer.observe(input);
    return () => {
      input.removeEventListener("scroll", sync);
      observer.disconnect();
    };
  }, [inputRef, text, hasMention]);

  if (!hasMention) return null;

  return (
    <div className="chat-input-mirror" ref={mirrorRef} aria-hidden="true">
      {segments.map((seg, i) =>
        seg.kind === "text" ? (
          <React.Fragment key={i}>{seg.text}</React.Fragment>
        ) : (
          <mark
            key={i}
            className={
              "chat-input-mark" + (seg.isFile ? "" : " chat-input-mark--plain")
            }
          >
            {seg.text}
          </mark>
        )
      )}
      {/* A draft ending in a newline would otherwise be one line shorter than the
          textarea's own view of it, and scroll out of step with it. */}
      {"\n"}
    </div>
  );
}
