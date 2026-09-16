import type { ChatMessage, ToolActivity } from "../types";

/**
 * Every run that is still going, newest last, across the live step and the frozen
 * transcript.
 *
 * Two shapes qualify, and they are the two halves of one run's life: a row still
 * `running`, reporting through its server's run channel while the call blocks, and a
 * row marked `background` whose watcher is still polling it.
 *
 * Both must have said what they are doing. That is the evidence, not a formality:
 * most tool calls are over in milliseconds and a spinner with nothing to say does not
 * earn a card — and a session reloaded from disk brings back rows still marked
 * `background` whose watchers died with the process that ran them. Requiring a phase
 * means a restored run shows a card only once something actually reports on it again.
 *
 * The transcript is scanned as well as the live step because a detached run outlives
 * the step that launched it — which is the whole reason the dock exists.
 */
export function runsInFlight(
  messages: ChatMessage[],
  liveToolCalls: ToolActivity[],
): ToolActivity[] {
  const out: ToolActivity[] = [];
  const seen = new Set<string>();

  const consider = (tool: ToolActivity) => {
    if (seen.has(tool.id)) return;
    const alive = tool.status === "running" || tool.status === "background";
    if (!alive || !tool.phase) return;
    seen.add(tool.id);
    out.push(tool);
  };

  for (const message of messages) {
    for (const tool of message.tools ?? []) consider(tool);
  }
  for (const tool of liveToolCalls) consider(tool);
  return out;
}
