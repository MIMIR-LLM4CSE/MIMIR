import React, { useCallback, useEffect, useRef, useState } from "react";
import { clampPanelWidth, readPanelWidth, storePanelWidth } from "./panelWidthUtils";

/**
 * A drawer's width, as the reader set it.
 *
 * Returns what to put on the panel's own style and the handle to render at its edge.
 * The width is held in state while dragging and written to storage on release only:
 * one write per drag rather than one per mouse move. During the drag the width goes
 * straight onto the element, once per frame, and React hears of it on release.
 */
export function usePanelWidth(key: string, fallback = 220) {
  const [width, setWidth] = useState(() =>
    readPanelWidth(key, fallback, window.innerWidth));
  const drag = useRef<{ x: number; width: number; main: HTMLElement | null } | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const pending = useRef<number | null>(null);
  const frame = useRef<number | null>(null);

  // Undo everything a drag set up. Safe to call twice, and from any way a drag ends.
  const endDrag = useCallback(() => {
    const from = drag.current;
    drag.current = null;
    document.body.classList.remove("is-resizing");
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    if (from?.main) {
      from.main.style.width = "";
      from.main.style.flex = "";
    }
    const final = pending.current;
    pending.current = null;
    return from ? final : null;
  }, []);

  useEffect(() => () => { endDrag(); }, [endDrag]);

  // The window can be made narrower than the drawer — by the user, or by VS Code
  // moving the view into a side bar. A drawer wider than its window would push the
  // chat off-screen, so the stored width is re-clamped rather than kept.
  useEffect(() => {
    const onResize = () => setWidth((w) => clampPanelWidth(w, window.innerWidth));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    // The chat keeps its width for the drag: re-wrapping the whole conversation on
    // every frame is what made dragging crawl. It re-lays out once, on release.
    const main = document.querySelector<HTMLElement>(".main");
    if (main) {
      main.style.width = `${main.getBoundingClientRect().width}px`;
      main.style.flex = "none";
    }
    drag.current = { x: e.clientX, width, main };
    document.body.classList.add("is-resizing");
    // Capture, so the drag survives the pointer leaving the 6px handle — at speed it
    // always does, and a drag that stops when it does is a drag that cannot be used.
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    e.preventDefault();
  }, [width]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const from = drag.current;
    if (!from) return;
    pending.current = clampPanelWidth(from.width + (e.clientX - from.x), window.innerWidth);
    if (frame.current !== null) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = null;
      if (panelRef.current && pending.current !== null) {
        panelRef.current.style.width = `${pending.current}px`;
      }
    });
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    if (!drag.current) return;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
    const final = endDrag();
    if (final === null) return;
    if (panelRef.current) panelRef.current.style.width = `${final}px`;
    setWidth(final);
    storePanelWidth(key, final);
  }, [key, endDrag]);

  // The handle with the keyboard on it: the arrow keys are what a reader who cannot
  // drag will try, and each press is a width worth keeping.
  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 48 : 16;
    const delta = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
    if (!delta) return;
    e.preventDefault();
    setWidth((w) => {
      const next = clampPanelWidth(w + delta, window.innerWidth);
      storePanelWidth(key, next);
      return next;
    });
  }, [key]);

  return {
    width,
    panelRef,
    handleProps: {
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel: onPointerUp,
      onLostPointerCapture: onPointerUp,
      onKeyDown,
    },
  };
}

interface Props {
  label: string;
  handleProps: ReturnType<typeof usePanelWidth>["handleProps"];
}

/**
 * The grab edge of a drawer.
 *
 * A separator rather than a button: it is reachable with the keyboard, where the arrow
 * keys are what a reader will try, and it says what it separates for anyone reading the
 * page aloud.
 */
export const PanelResizer: React.FC<Props> = ({ label, handleProps }) => (
  <div
    className="panel-resizer"
    role="separator"
    aria-orientation="vertical"
    aria-label={`Resize the ${label} panel`}
    title="Drag to resize"
    tabIndex={0}
    {...handleProps}
  />
);
