import { useEffect, useState } from "react";

/**
 * Which of *ids* name rows that have scrolled out of the visible thread.
 *
 * A run can outlast the part of the conversation it started in: the agent keeps
 * working, the thread grows, and the row carrying the progress leaves the screen.
 * Knowing that is what lets the dock appear only when it is actually needed — a
 * card duplicating a row the reader can already see is just clutter.
 *
 * Rows are found by their `data-tool-id`, not by a ref threaded through every
 * component between here and the row: a row moves from the live list into a frozen
 * message mid-run, which replaces its element, and an attribute survives that where
 * a captured ref does not. *revision* is what re-runs the search when it happens.
 *
 * An id whose row is not in the DOM at all is deliberately NOT reported offscreen:
 * "not rendered yet" and "scrolled away" are different, and only the second one is
 * worth interrupting the reader for.
 */
export function useOffscreenRows(
  ids: string[],
  rootRef: React.RefObject<HTMLElement>,
  revision: unknown = 0,
): Set<string> {
  const [offscreen, setOffscreen] = useState<Set<string>>(new Set());
  const key = ids.join("|");

  useEffect(() => {
    const root = rootRef.current;
    if (!root || ids.length === 0) {
      setOffscreen((prev) => (prev.size === 0 ? prev : new Set()));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        setOffscreen((prev) => {
          const next = new Set(prev);
          for (const entry of entries) {
            const id = (entry.target as HTMLElement).dataset.toolId;
            if (!id) continue;
            if (entry.isIntersecting) next.delete(id);
            else next.add(id);
          }
          // Same set, same object: a new Set on every scroll tick would re-render
          // the dock continuously for no change.
          if (next.size === prev.size && [...next].every((i) => prev.has(i))) {
            return prev;
          }
          return next;
        });
      },
      { root, threshold: 0 },
    );

    const found = new Set<string>();
    for (const id of ids) {
      const el = root.querySelector(`[data-tool-id="${CSS.escape(id)}"]`);
      if (el) {
        observer.observe(el);
        found.add(id);
      }
    }
    // Forget anything we are no longer watching, so a finished run's card cannot
    // outlive the row it belonged to.
    setOffscreen((prev) => {
      const next = new Set([...prev].filter((i) => found.has(i)));
      return next.size === prev.size ? prev : next;
    });

    return () => observer.disconnect();
    // `key` stands in for `ids`, which is a fresh array on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, rootRef, revision]);

  return offscreen;
}
