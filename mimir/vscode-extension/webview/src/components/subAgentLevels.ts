import type { SubAgentLevel } from "../types";

/**
 * How far a sub-agent of this session may go.
 *
 * Its own module rather than a panel's export: the choice is an agent setting — it
 * belongs with thinking depth and enforcement, in the popover — while the list of
 * sub-agents at work is a drawer. Two different questions, read at two different
 * moments.
 *
 * Mirrors the ladder in client/config/constants.py, weakest first. The rung's value is
 * what travels; "parallel" is what it was called before, and the server still accepts
 * that from a stored preference or a typed `/subagents parallel`.
 */
export const SUBAGENT_LEVEL_OPTIONS: {
  value: SubAgentLevel;
  icon: string;
  label: string;
  desc: string;
}[] = [
  {
    value: "explore",
    icon: "🔍",
    label: "explore",
    desc: "Sub-agents read, search and navigate. They change nothing",
  },
  {
    value: "build",
    icon: "🔨",
    label: "build & run",
    desc: "They may also edit and run — each in its own copy of the repository, on its own branch",
  },
];
