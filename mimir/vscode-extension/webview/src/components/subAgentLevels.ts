import type { SubAgentLevel } from "../types";

/**
 * How far a sub-agent of this session may go: an agent setting, shown in the settings
 * popover. Mirrors the ladder in client/config/constants.py, weakest first; the server
 * also accepts the rung's former name, "parallel".
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
