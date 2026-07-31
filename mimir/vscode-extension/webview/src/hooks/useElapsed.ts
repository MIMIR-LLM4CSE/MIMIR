import { useEffect, useState } from "react";

/**
 * Live-updating elapsed time (ms) since `startedAt`.
 * While `running`, ticks every 200ms; once stopped, freezes at the last value
 * relative to `startedAt`. Shared by tool-activity and thinking panels.
 */
export function useElapsed(startedAt: number, running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [running]);
  return Math.max(0, now - startedAt);
}

/** Human-readable duration: "420ms" under a second, else "3.2s". */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
