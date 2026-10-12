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
 * A `<textarea>` holds characters, not elements, so nothing in it can be coloured — and
 * the input has to stay a textarea: the caret arithmetic the "@" and "/" dropdowns, the
 * history browse and the steer path all do is `value` + `selectionStart`, and a
 * contenteditable has neither. So the text is drawn a second time in a div underneath,
 * with the same metrics, and only the mentions are painted; the textarea sits on top
 * with a transparent background and keeps every behaviour it had.
 *
 * Nothing is rendered when the draft carries no mention, so the ordinary case is the
 * input exactly as it was. The mirror's box is taken from the textarea itself
 * (`clientWidth`/`clientHeight`, which exclude a scrollbar when one appears), because a
 * width that disagrees by a pixel wraps a line early and slides every tint below it.
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
