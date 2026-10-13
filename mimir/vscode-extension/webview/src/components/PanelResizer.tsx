import React, { useCallback, useEffect, useRef, useState } from "react";
import { clampPanelWidth, readPanelWidth, storePanelWidth } from "./panelWidthUtils";

/**
 * A drawer's width, as the reader set it.
 *
 * Returns what to put on the panel's own style and the handle to render at its edge.
 * The width is held in state while dragging and written to storage on release only:
 * one write per drag rather than one per mouse move.
 */
export function usePanelWidth(key: string, fallback = 220) {
  const [width, setWidth] = useState(() =>
    readPanelWidth(key, fallback, window.innerWidth));
  const drag = useRef<{ x: number; width: number } | null>(null);

  // The window can be made narrower than the drawer — by the user, or by VS Code
  // moving the view into a side bar. A drawer wider than its window would push the
  // chat off-screen, so the stored width is re-clamped rather than kept.
  useEffect(() => {
    const onResize = () => setWidth((w) => clampPanelWidth(w, window.innerWidth));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    drag.current = { x: e.clientX, width };
    // Capture, so the drag survives the pointer leaving the 6px handle — at speed it
    // always does, and a drag that stops when it does is a drag that cannot be used.
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    e.preventDefault();
  }, [width]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const from = drag.current;
    if (!from) return;
    setWidth(clampPanelWidth(from.width + (e.clientX - from.x), window.innerWidth));
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    if (!drag.current) return;
    drag.current = null;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
    setWidth((w) => {
      storePanelWidth(key, w);
      return w;
    });
  }, [key]);

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
    handleProps: {
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel: onPointerUp,
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
