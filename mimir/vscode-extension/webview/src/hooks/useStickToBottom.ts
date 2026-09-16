import { useCallback, useEffect, useRef } from "react";

/** How far from the bottom still counts as being at the bottom, in px. */
const DEFAULT_SLACK = 40;
/** How long after a gesture a scroll event is still attributed to it. */
const GESTURE_WINDOW_MS = 700;

/**
 * Keep a scrollable pane pinned to its bottom while content streams into it, and
 * stop following as soon as the reader scrolls away.
 *
 * The whole difficulty is telling those two apart, because a scroll event says
 * nothing about what caused it: our own `scrollTop` writes fire one too. Deciding
 * from the position at event time is what broke both panes that used to do it.
 * The write happens in a layout effect, the event is delivered later, and by then
 * the next chunk has landed and made the content taller — so the pane measures as
 * "scrolled up" by exactly the text that just arrived, and the follow switches
 * itself off in the middle of a stream nobody touched. Which is why it looked
 * intermittent: it takes a chunk taller than the slack landing inside that window,
 * so a slow stream follows fine and a fast one stops following.
 *
 * So: reaching the bottom resumes the follow whatever moved the pane there, and
 * leaving it pauses the follow only when a real gesture happened just before.
 */
export function useStickToBottom<T extends HTMLElement>(slack: number = DEFAULT_SLACK) {
  const ref = useRef<T>(null);
  /** True while the pane follows its bottom; false once the reader left it. */
  const stickRef = useRef(true);
  const gestureAtRef = useRef(0);

  const scrollToBottom = useCallback(() => {
    const el = ref.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, []);

  /** Resume following, whatever the reader had done. For content that must be seen. */
  const follow = useCallback(() => {
    stickRef.current = true;
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < slack;
    if (atBottom) stickRef.current = true;
    else if (Date.now() - gestureAtRef.current < GESTURE_WINDOW_MS) stickRef.current = false;
  }, [slack]);

  // Gestures are watched on the element itself rather than taken as props: the
  // list is long — wheel, touch, keys, a scrollbar drag — and none of it is the
  // business of the component doing the rendering.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const note = () => { gestureAtRef.current = Date.now(); };
    const passive = { passive: true } as const;
    el.addEventListener("wheel", note, passive);
    el.addEventListener("touchmove", note, passive);
    el.addEventListener("pointerdown", note, passive);
    el.addEventListener("keydown", note, passive);
    return () => {
      el.removeEventListener("wheel", note);
      el.removeEventListener("touchmove", note);
      el.removeEventListener("pointerdown", note);
      el.removeEventListener("keydown", note);
    };
  }, []);

  // Re-pin when the content grows after the commit that added it.
  //
  // Most late growth is avoidable and was fixed at the source, but some is not: a
  // panel that has to measure the DOM to know whether it clips its own content can
  // only add its control in a second commit, and that commit changes no state the
  // pinning effect watches. The pane was then left pinned to the height the row had
  // before it grew, with the new part below the fold.
  //
  // Children are (re)observed whenever the list changes, so the observation set
  // follows the transcript without the rendering components knowing about any of it.
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => scrollToBottom());
    const observeChildren = () => {
      ro.disconnect();
      for (const child of Array.from(el.children)) ro.observe(child);
    };
    observeChildren();
    if (typeof MutationObserver === "undefined") return () => ro.disconnect();
    const mo = new MutationObserver(observeChildren);
    mo.observe(el, { childList: true });
    return () => { mo.disconnect(); ro.disconnect(); };
  }, [scrollToBottom]);

  return { ref, stickRef, scrollToBottom, follow, onScroll };
}
