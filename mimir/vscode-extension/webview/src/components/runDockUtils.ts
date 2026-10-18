import type { ChatMessage, ToolActivity } from "../types";

/**
 * Every run still going, newest last, across the live step and the frozen transcript:
 * a row still `running` while its call blocks, or `background` while a watcher polls.
 *
 * Only once it has said what it is doing: most calls end in milliseconds, and a
 * reloaded `background` row whose watcher died shows a card only when it reports again.
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
