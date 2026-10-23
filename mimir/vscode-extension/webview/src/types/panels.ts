// The drawers beside the chat: the sections servers report, and the sub-agents of the session.

import type { WatchedRun } from "./jobs";
import type { SubAgentLevel } from "./settings";

/** One line of a panel section, in the words of the server that filled it. */
export interface PanelLine {
  label: string;
  value: string;
  /** Optional emphasis the server asked for, e.g. "running". */
  state?: string;
}

/**
 * A request the server wrote for the panel to offer — a question to put to the model,
 * never a tool to call. The prompt comes from the end that knows the scheduler and the
 * partition, which is what keeps tool names out of this side entirely.
 */
export interface PanelGroupAction {
  kind: string;
  label: string;
  prompt: string;
}

/**
 * One node of a section's tree: a scheduler, or a kind of machine on one.
 *
 * Recursive and read without being interpreted, exactly as `lines` are — any server
 * answering with `groups` gets a tree with no further change on this side. `id` is
 * stable across refreshes because the open rows are keyed on it.
 */
export interface PanelGroup {
  id: string;
  label: string;
  /** A one-line summary shown on the row itself. */
  value?: string;
  state?: string;
  host?: string;
  local?: boolean;
  nodes?: string[];
  lines?: PanelLine[];
  detail?: string;
  groups?: PanelGroup[];
  action?: PanelGroupAction;
  /** Runs the client found to be on this machine. Attached on the way out. */
  badges?: WatchedRun[];
}

export interface PanelSection {
  section: string;
  title: string;
  lines: PanelLine[];
  /** The nested half, for a server whose answer is a tree rather than a list. */
  groups?: PanelGroup[];
  detail?: string;
  /** A view the section offers to open, by name ("code_map"). */
  open?: string;
  /** A small map of what the view shows, drawn in the section (the code's components). */
  map?: PanelMiniMap;
}

/** Components stacked by tier, highest first, and the links between them. */
export interface PanelMiniMap {
  components: { id: string; name: string; tier: number; files: number; breaks: number; cycles: number }[];
  links: { from: string; to: string; n: number; breaks: number }[];
}

export interface PanelReportMessage {
  type: "panel_report";
  /** Whether a conversation's agent was there to ask. A report from one that has none
   *  yet carries no sections — the servers that fill them are not running — which is
   *  not the same answer as "this workspace reaches no clusters". */
  agent_ready?: boolean;
  subagent_level: SubAgentLevel;
  subagents: SubAgent[];
  running?: number;
  runs: WatchedRun[];
  sections: PanelSection[];
}

/** One sub-agent of the session in view, as the spawn server's card describes it. */
export interface SubAgent {
  id: string;
  state: "running" | "finished" | "abandoned" | "failed" | "unknown";
  task?: string;
  /** The tools it was granted; empty means a read-only exploration. */
  tools?: string[];
  model?: string;
  budget_secs?: number;
  completed?: boolean;
  started_at?: string;
  ended_at?: string;
  /** Its answer or handoff — only sent when one sub-agent is opened. */
  answer?: string;
  /** Its own checklist, raw Markdown — likewise. */
  todo?: string;
  /** The tail of its activity log: one row per event, paired by `activityRows`. */
  activity?: unknown[];
  /** Set when it worked in a copy of the repository: branch, path, diffstat. */
  workspace?: {
    /** Empty when it changed nothing: that branch carried no commit and was dropped. */
    branch?: string;
    path?: string;
    /** The repository the copy was cut from. */
    repo?: string;
    files_changed?: string[];
    diffstat?: string;
    /** A copy is kept now; cards written before that say so here. */
    kept?: boolean;
    /** What the server has to say about it — how to bring the work over, mostly. */
    note?: string;
  };
}

export interface SubAgentsListMessage {
  type: "subagents_list";
  session_id: string;
  subagents: SubAgent[];
  /** How many are working right now — the badge on the button. */
  running?: number;
}

export interface SubAgentMessage {
  type: "subagent";
  subagent: SubAgent;
}

/** The drawer: the sub-agent cards alone, or every section (``build`` starts the agent
 *  whose servers fill them; ``refresh`` reads past their caches). */
export interface ListSubagentsMessage {
  type: "list_subagents";
}

export interface ListPanelMessage {
  type: "list_panel";
  build?: boolean;
  refresh?: boolean;
}

/** Ask for the workspace's code map, for the editor's map tab. */
export interface CodeMapRequestMessage {
  type: "code_map";
}

/** The code map the servers answer (``shared.code_map``'s dict); no ``map`` when none does. */
export interface CodeMapMessage {
  type: "code_map";
  map?: unknown;
  /** The optimisation session's steps, when a server answers them (``CODE_MAP_STEPS``). */
  session?: unknown;
}

/** Stop a run the watcher holds, with the cancel op its own server named. */
export interface CancelRunMessage {
  type: "cancel_run";
  job_key: string;
}

export interface ReadSubagentMessage {
  type: "read_subagent";
  id: string;
}
