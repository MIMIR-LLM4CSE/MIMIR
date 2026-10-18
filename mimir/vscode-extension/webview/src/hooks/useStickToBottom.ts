import { useCallback, useEffect, useRef } from "react";

/** How far from the bottom still counts as being at the bottom, in px. */
const DEFAULT_SLACK = 40;
/** How long after a gesture a scroll event is still attributed to it. */
const GESTURE_WINDOW_MS = 700;

/**
 * Keep a scrollable pane pinned to its bottom while content streams into it, and stop
 * following as soon as the reader scrolls away.
 *
 * A scroll event does not say what caused it, so reaching the bottom resumes the follow
 * whatever moved the pane, and leaving it pauses only after a real gesture.
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

  /** Stop following, for a scroll the app makes on the reader's behalf. Such a scroll
   *  carries no gesture, so without this the next chunk pulls the pane straight back
   *  to the bottom — which is what made a click on a docked run land nowhere. */
  const release = useCallback(() => {
    stickRef.current = false;
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

  // Re-pin when content grows after the commit that added it (a panel that measures
  // the DOM adds its control in a second commit the pinning effect does not watch).
  // Children are re-observed as the list changes.
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

  return { ref, stickRef, scrollToBottom, follow, release, onScroll };
}
