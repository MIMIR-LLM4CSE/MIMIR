// The drawers beside the chat: the sub-agents of the session and the scientific-
// computing sections. Both read files other processes write, so the panel asks for
// them rather than being pushed to — on opening, at the end of a turn, and while
// anything is still working.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ClientMessage, PanelSection, SubAgent, SubAgentLevel, WatchedRun } from "../types";
import type { ServerHandlers } from "../state/serverMessages";
import { vscodePostMessage } from "./useWebSocket";

/** How often the cards are re-read while a turn runs or a sub-agent works. */
const SUBAGENT_POLL_MS = 2000;

interface Activity {
  /** A turn of the conversation on screen is running. */
  busy: boolean;
  /** Bumped on every `ready`: the agent behind the drawer may just have come up. */
  readyCount: number;
  agentReady: boolean;
}

export function usePanels(send: (msg: ClientMessage) => void,
                          { busy, readyCount, agentReady }: Activity) {
  // Work this session sent out, kept apart from the conversations to come back to.
  const [subAgents, setSubAgents] = useState<SubAgent[]>([]);
  const [openedSubAgent, setOpenedSubAgent] = useState<SubAgent | null>(null);
  const [runningSubAgents, setRunningSubAgents] = useState(0);
  const [subAgentLevel, setSubAgentLevel] = useState<SubAgentLevel>("explore");
  const [panelSections, setPanelSections] = useState<PanelSection[]>([]);
  // The runs no machine row claimed: shown on their own rather than not at all.
  const [panelRuns, setPanelRuns] = useState<WatchedRun[]>([]);
  // A ⟳ in flight, so the button can say so: a re-read of the clusters takes seconds.
  const [panelPending, setPanelPending] = useState(false);
  // Whether the last report came from a conversation that has an agent: before one is
  // built there are no servers to fill the sections, and the drawer says so.
  const [panelAgentReady, setPanelAgentReady] = useState(false);
  const [showSciencePanel, setShowSciencePanel] = useState(false);
  const [showSubAgentsPanel, setShowSubAgentsPanel] = useState(false);

  const refreshSubAgents = useCallback(() => send({ type: "list_subagents" }), [send]);
  // The server answers with a fresh report; the watcher wakes the conversation.
  const cancelRun = useCallback(
    (jobKey: string) => send({ type: "cancel_run", job_key: jobKey }),
    [send],
  );
  // One call fills the whole drawer from the servers' caches. `build`: before the
  // first query there is no agent holding the servers; a no-op once one is up.
  const refreshPanel = useCallback(() => send({ type: "list_panel", build: true }), [send]);
  // The ⟳ alone reads past every cache, at the price of an ssh to each cluster.
  const reloadPanel = useCallback(() => {
    setPanelPending(true);
    send({ type: "list_panel", build: true, refresh: true });
  }, [send]);
  // A view a section offers ("code_map"): asked of the server, drawn in an editor tab
  // by the extension once the answer comes back.
  const openView = useCallback((view: string) => {
    if (view === "code_map") send({ type: "code_map" });
  }, [send]);
  const openSubAgent = useCallback((id: string) => {
    setOpenedSubAgent(null);
    send({ type: "read_subagent", id });
  }, [send]);
  const changeSubAgentLevel = useCallback((level: SubAgentLevel) => {
    setSubAgentLevel(level);
    send({ type: "command", text: `/subagents ${level}` });
  }, [send]);

  // A turn that delegated left new cards on disk. The cards are read whether the
  // drawer is open or not (the button shows their count); the whole report only for a
  // drawer somebody is looking at.
  const wasBusyRef = useRef(false);
  useEffect(() => {
    const wasBusy = wasBusyRef.current;
    wasBusyRef.current = busy;
    if (busy || !wasBusy) return;
    refreshSubAgents();
    if (showSciencePanel) refreshPanel();
  }, [busy, refreshSubAgents, showSciencePanel, refreshPanel]);

  // The agent came up after the drawer was opened: its first ask found no servers.
  useEffect(() => {
    if (readyCount > 0 && showSciencePanel) {
      if (agentReady && !panelAgentReady) setPanelPending(true);
      refreshPanel();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readyCount, showSciencePanel, refreshPanel]);

  // While a turn runs or a sub-agent works, keep asking. The interface polls, never
  // the model: the agent is woken when a run lands.
  const pollCards = showSciencePanel ? refreshPanel : refreshSubAgents;
  useEffect(() => {
    if (!busy && runningSubAgents === 0) return;
    const timer = setInterval(pollCards, SUBAGENT_POLL_MS);
    return () => clearInterval(timer);
  }, [busy, runningSubAgents, pollCards]);

  const on: ServerHandlers = useMemo(() => ({
    subagents_list: (msg) => {
      setSubAgents(msg.subagents ?? []);
      setRunningSubAgents(msg.running ?? 0);
    },
    subagent_level: (msg) => setSubAgentLevel(msg.level),
    panel_report: (msg) => {
      setPanelPending(false);
      setPanelAgentReady(msg.agent_ready !== false);
      setSubAgentLevel(msg.subagent_level);
      setSubAgents(msg.subagents ?? []);
      setRunningSubAgents(msg.running ?? 0);
      // Before the agent is up an empty report keeps what is on screen; the drawer
      // asks again on `ready`.
      if (msg.agent_ready !== false || (msg.sections?.length ?? 0) > 0) {
        setPanelSections(msg.sections ?? []);
        setPanelRuns(msg.runs ?? []);
      }
    },
    subagent: (msg) => setOpenedSubAgent(msg.subagent ?? null),
    code_map: (msg) => {
      if (msg.map) vscodePostMessage({ type: "show_code_map", map: msg.map });
    },
  }), []);

  return {
    subAgents, openedSubAgent, runningSubAgents, subAgentLevel, panelSections, panelRuns,
    cancelRun,
    panelPending, panelAgentReady, showSciencePanel, showSubAgentsPanel,
    setShowSciencePanel, setShowSubAgentsPanel, setOpenedSubAgent,
    refreshSubAgents, refreshPanel, reloadPanel, openSubAgent, changeSubAgentLevel, openView,
    on,
  };
}
