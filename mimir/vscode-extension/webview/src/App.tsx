import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  useReducer,
  useMemo,
} from "react";
import type {
  ChatMessage,
  ClientMessage,
  ServerMessage,
  ApprovalMode,
} from "./types";
import { createChatReducer, initialChatState } from "./state/chatReducer";
import { routeServerMessage, type ServerHandlers, type ServerMessageOf } from "./state/serverMessages";
import { useAgentSettings } from "./hooks/useAgentSettings";
import { usePanels } from "./hooks/usePanels";
import { useEndpoint, type Host } from "./hooks/useEndpoint";
import { useWebSocket, vscodePostMessage } from "./hooks/useWebSocket";
import { useThreadScroll } from "./hooks/useThreadScroll";
import { useWorkspaceState } from "./hooks/useWorkspaceState";
import { useSessions } from "./hooks/useSessions";
import { useTranscriptSync } from "./hooks/useTranscriptSync";
import { useUserQuestion } from "./hooks/useUserQuestion";
import { useDisconnect } from "./hooks/useDisconnect";
import { makeId } from "./state/ids";
import { ChatThread } from "./components/ChatThread";
import { ForeignPromptStrip } from "./components/ForeignPromptStrip";
import { PlanBar } from "./components/PlanBar";
import { AgentSettings } from "./components/AgentSettings";
import { DetachButton } from "./components/DetachButton";
import { DisconnectPrompt } from "./components/DisconnectPrompt";
import { ModeSwitcher } from "./components/ModeSwitcher";
import { BlocksIcon, GearIcon } from "./components/icons";
import { TogglesPanel } from "./components/TogglesPanel";
import { ConnectForm } from "./components/ConnectForm";
import { UserQuestion } from "./components/UserQuestion";
import { SessionsPanel } from "./components/SessionsPanel";
import { SubAgentsPanel } from "./components/SubAgentsPanel";
import { SciencePanel } from "./components/SciencePanel";
import { ContextBar } from "./components/ContextBar";
import { RunProgressDock } from "./components/RunProgressDock";
import { BatchReviewBar } from "./components/BatchReviewBar";
import { MimirIntro } from "./components/MimirIntro";
import { Composer } from "./components/Composer";
import { StatusBar } from "./components/StatusBar";
import { useComposer } from "./hooks/useComposer";

// VS Code webview API (optional — only present inside a webview).

function getWsUrl(): string {
  // Ask the extension host for config; fall back to default.
  return (window as unknown as { __MIMIR_WS_URL__?: string }).__MIMIR_WS_URL__
    ?? "ws://localhost:8765";
}


export const App: React.FC = () => {
  // Streaming / thinking / approval state machine lives in a pure reducer so
  // there are no stale-closure refs and the logic is unit-testable.
  const chatReducer = useMemo(() => createChatReducer(makeId), []);
  const [chatState, dispatch] = useReducer(chatReducer, initialChatState);
  const { messages, draft, draftSeq, busy, liveToolCalls, liveThinkingBlocks } = chatState;
  // Mirror of reducer state for reads inside event callbacks (approval lookup,
  // session-load message preservation) without stale closures.
  const chatStateRef = useRef(chatState);
  chatStateRef.current = chatState;

  // The host's actions, read once useWebSocket below has created them.
  const hostRef = useRef<Host>({ connect: () => {}, fetchModels: () => {}, disconnect: () => {} });
  const endpoint = useEndpoint(useCallback(() => hostRef.current, []));
  const {
    connection, agentReady, model, anthropicModels, backend, vllmBaseUrl, rayBaseUrl,
    ollamaBaseUrl, remembered, runningServer, endpointModels, modelsError, modelsProbed,
    connectFormKey,
  } = endpoint;
  // The socket is opened below, by a hook that needs the message handler built from
  // these; parts that send go through this, which reads the socket once it exists.
  const sendRef = useRef<(msg: ClientMessage) => void>(() => {});
  const sendVia = useCallback((msg: ClientMessage) => sendRef.current(msg), []);
  const settings = useAgentSettings(sendVia);
  const {
    agentMode, thinkingLevel, thinkingProfile, streaming, contextMode, enforcement,
    temperature, approvalMode,
  } = settings;
  const workspace = useWorkspaceState(sendVia);
  const { onSessionLoaded: workspaceLoaded, reset: resetWorkspace } = workspace;
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [togglesOpen, setTogglesOpen] = useState(false);
  // Last submitted query text, so an error card can offer a one-click Retry.
  const lastQueryRef = useRef<string>("");
  const {
    chatThreadRef, bottomRef, lastUserMsgRef, onScroll: handleChatScroll,
    scrollToBottom, scrollToApproval, anchorNewQuery, dockedRuns, focusRun,
  } = useThreadScroll(chatState);

  // A `replay` frame is a bundle of ordinary events, so it goes through the panel's
  // own handler — one reducer for what happened live and while nobody was attached.
  const handlerRef = useRef<(m: ServerMessage) => void>(() => {});
  const redeliver = useCallback((m: ServerMessage) => handlerRef.current(m), []);
  // A loaded conversation's share for the parts that are not the session list.
  const pushTranscriptRef = useRef<() => void>(() => {});
  const onSessionLoaded = useCallback(
    (msg: ServerMessageOf<"session_loaded">, switched: boolean, pushTranscript: boolean) => {
      // Hand the richer copy straight back rather than waiting for the next turn to
      // end: the reconnect may be the last thing that happens in this session.
      if (pushTranscript) pushTranscriptRef.current();
      workspaceLoaded(msg.todos, switched);
    },
    [workspaceLoaded],
  );
  const conversations = useSessions({
    dispatch, chatStateRef, scrollToBottom, redeliver, onLoaded: onSessionLoaded,
  });
  const {
    sessions, activeSessionId, activeSessionIdRef, sessionLoading, setSessionLoading,
    showSessionsPanel, setShowSessionsPanel, foreignPrompts, isForeign, rememberForeign,
    forgetForeign, noteSeq,
  } = conversations;
  const transcript = useTranscriptSync({
    send: sendVia, busy, chatStateRef, activeSessionIdRef,
    renderedSeqRef: conversations.renderedSeqRef,
  });
  pushTranscriptRef.current = transcript.requestPush;
  const question = useUserQuestion({
    send: sendVia, dispatch, activeSessionIdRef, isForeign, rememberForeign, forgetForeign,
    scrollToBottom,
  });

  // ── WebSocket message handler ─────────────────────────────────────────────
  // The parts that hold their own state answer through these tables; what is left is
  // the reducer's.
  const tablesRef = useRef<ServerHandlers[]>([]);

  const handleServerMessage = useCallback((msg: ServerMessage) => {
    noteSeq(msg);

    // The parts that hold their own state answer first; a type only they know ends here.
    const routed = routeServerMessage(msg, tablesRef.current);

    switch (msg.type) {
      case "ready":
        handleReady();
        return;

      case "open_editor":
        // The agent wrote a file it wants the user to read (e.g. a plan .md).
        // Forward to the extension host. `open_preview` renders Markdown files
        // (like plan .md) in VS Code's Markdown preview; other files open as a
        // normal editor tab. Absolute paths supported.
        if (msg.path) vscodePostMessage({ type: "open_preview", file: msg.path });
        return;

      // The batch keeps the file (useWorkspaceState); the reducer draws the edit card.
      case "diff":
        dispatch(msg);
        return;
    }

    if (routed) return;
    // Remaining message-state types (output/status/token/file_progress/
    // approval/thinking*/answer/error) are handled by the reducer; dispatch and
    // run the matching scroll side-effect afterwards.
    if (msg.type === "approval" && isForeign(msg)) {
      rememberForeign(msg);
      return;
    }
    dispatch(msg);
    if (msg.type === "approval") {
      scrollToApproval();
    } else if (msg.type === "file_progress") {
      if ((msg.diffs ?? []).length > 0) scrollToApproval();
    } else if (
      msg.type === "output" || msg.type === "status" ||
      msg.type === "command_output" ||
      msg.type === "token" || msg.type === "thinking" ||
      msg.type === "answer" || msg.type === "error" ||
      msg.type === "tool_call" || msg.type === "subagent_event"
    ) {
      scrollToBottom();
    }
  }, [noteSeq, isForeign, rememberForeign, scrollToBottom, scrollToApproval]);

  // Kept in a ref so a replay can run each bundled event through this handler.
  useEffect(() => {
    handlerRef.current = handleServerMessage;
  }, [handleServerMessage]);

  const resetSession = () => {
    dispatch({ type: "reset" });
    resetWorkspace();
    question.clear();
    conversations.reset();
    endpoint.reset();
  };

  const { send, getConfig, connect, fetchModels, disconnect, createSession, switchSession, deleteSession, renameSession } = useWebSocket({
    url: getWsUrl(),
    onMessage: handleServerMessage,
    onClose: () => {
      // Preserve messages and todos — only reset connection/busy state so the
      // chat history survives a transient drop (e.g. during batch-revert I/O).
      dispatch({ type: "connection_lost" });
      endpoint.lose("disconnected");
    },
    onError: () => {
      // Surface the error state; ws_closed follows and drives reconnect UI.
      dispatch({ type: "connection_lost" });
      endpoint.lose("error");
    },
  });
  sendRef.current = send;
  hostRef.current = { connect, fetchModels, disconnect };
  const departure = useDisconnect({
    send, sessions, activeSessionId, busy, approvalMode,
    leave: () => {
      disconnect();
      resetSession();
    },
  });

  // Creating a session also opens it: the server makes the new session active
  // and answers with `session_loaded`, so we only have to show the (empty)
  // thread — panel closed, spinner up until that reply lands.
  const startNewSession = useCallback(() => {
    createSession();
    setSessionLoading(true);
    setShowSessionsPanel(false);
  }, [createSession]);

  // Request config from extension host on mount
  useEffect(() => {
    getConfig();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-sync client-side settings whenever the server signals it is ready.
  // The server resets to defaults on each restart; the webview persists state.
  // We sync on every "ready" (including the worker-ready that fires after vLLM
  // finishes loading) because the first "ready" arrives before the agent exists.
  const syncSettingsRef = useRef({ thinkingLevel, streaming, agentMode });
  syncSettingsRef.current = { thinkingLevel, streaming, agentMode };
  const [readyCount, setReadyCount] = useState(0);
  const handleReady = useCallback(() => setReadyCount((n) => n + 1), []);
  const panels = usePanels(sendVia, { busy, readyCount, agentReady });
  const {
    subAgents, openedSubAgent, runningSubAgents, subAgentLevel, panelSections,
    panelPending, panelAgentReady, showSciencePanel, showSubAgentsPanel,
    setShowSciencePanel, setShowSubAgentsPanel, setOpenedSubAgent,
    refreshSubAgents, refreshPanel, reloadPanel,
  } = panels;
  tablesRef.current = [
    endpoint.on, settings.on, panels.on, workspace.on, conversations.on, transcript.on,
    question.on, departure.on,
  ];
  useEffect(() => {
    if (readyCount === 0) return;
    const { thinkingLevel: tl, streaming: st, agentMode: am } = syncSettingsRef.current;
    if (tl > 0) send({ type: "command", text: `/thinking-depth ${tl}` });
    if (!st) send({ type: "command", text: `/streaming off` });
    if (am !== "agent") send({ type: "command", text: `/mode ${am}` });
    // Fetch attachable MCP resources for the @-mention autocomplete.
    send({ type: "list_resources" });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readyCount, send]);

  // ── User actions ──────────────────────────────────────────────────────────

  /**
   * Put one message to the model, as the user's own turn. The science panel's offered
   * requests go through here too, so whatever tool the model reaches for still asks
   * for approval.
   */
  const sendPrompt = useCallback((text: string) => {
    // Chat-while-busy: a message sent during a run is a "steer" — queued and
    // injected into the agent's current turn at its next step, not a new query.
    if (busy) {
      dispatch({ type: "steer_query", text });
      send({ type: "steer", text });
      return;
    }
    lastQueryRef.current = text;
    dispatch({ type: "submit_query", text });
    send({ type: "query", text });
    anchorNewQuery();
  }, [busy, send, anchorNewQuery]);

  const composer = useComposer({
    skillToggles: workspace.skillToggles,
    sendCommand: (text: string) => send({ type: "command", text }),
    sendPrompt,
  });
  tablesRef.current = [...tablesRef.current, composer.on];

  // Re-run the most recent query — surfaced as Retry on the latest error card.
  const retryLastQuery = useCallback(() => {
    const text = lastQueryRef.current;
    if (!text || busy) return;
    dispatch({ type: "submit_query", text });
    send({ type: "query", text });
    anchorNewQuery();
  }, [busy, send, anchorNewQuery]);

  // The click travels to the shell server through the session's state dir, not
  // through the model: the agent is parked awaiting this very tool call, so an
  // instruction it had to read would arrive too late to divert anything.
  const handleDivert = useCallback(
    (id: string) => send({ type: "divert_to_background", id }),
    [send]
  );

  const handleApprovalResponse = useCallback(
    (id: string, choice: "y" | "n" | "a", approvedFiles?: string[]) => {
      // Send approval_response for every merged ID so all blocked shims unblock.
      const card = chatStateRef.current.messages.find(
        (m) => m.kind === "approval" && m.approval?.id === id
      );
      const allIds = card?.approval?.ids ?? [id];
      // The conversation that asked, read off the card. The server drops an answer
      // carrying none rather than guessing; the session on screen is exactly the wrong
      // guess for a card raised elsewhere.
      const sessionId = card?.approval?.session_id ?? activeSessionIdRef.current ?? undefined;
      for (const aid of allIds) {
        const payload: {
          type: "approval_response"; id: string; choice: "y" | "n" | "a";
          session_id?: string; approved_files?: string[];
        } = { type: "approval_response", id: aid, choice, session_id: sessionId };
        if (approvedFiles !== undefined) payload.approved_files = approvedFiles;
        send(payload);
      }
      // Update the card(s) and set busy via the reducer.
      dispatch({ type: "approval_response", id, choice });
    },
    [send]
  );

  const handleApprovalModeChange = useCallback((val: ApprovalMode) => {
    settings.changeApprovalMode(val);
    // The switch has to work mid-run, and mid-run is exactly when a card is already
    // on screen with the agent parked on it. The new mode covers that call, so
    // answering it here is what the user just asked for — leaving it standing would
    // make "auto" mean "auto, starting after you clear this one".
    const msgs = chatStateRef.current.messages;
    let card: ChatMessage | undefined;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].kind === "approval" && msgs[i].approval !== undefined) { card = msgs[i]; break; }
    }
    if (!card?.approval) return;
    const leavesWorkspace =
      (card.approval.oow_paths?.length ?? 0) > 0 || Boolean(card.approval.oow_path);
    const covered = val === "auto_all" || (val === "auto" && !leavesWorkspace);
    if (covered) handleApprovalResponse(card.approval.id, "y");
  }, [settings.changeApprovalMode, handleApprovalResponse]);

  // data-mode drives the chat's accent colour: blue agent / red plan / green ask.
  return (
    <div className={`app ${showSessionsPanel ? "with-sessions" : ""}`} data-mode={agentMode}>
      {/* ── Sessions panel ───────────────────────────────────────────── */}
      {showSessionsPanel && connection === "connected" && (
        <SessionsPanel
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelect={(id) => {
            // Selecting a session opens it and closes the panel.
            if (id !== activeSessionId) setSessionLoading(true);
            switchSession(id);
            setShowSessionsPanel(false);
          }}
          onDelete={(id) => deleteSession(id)}
          onRename={(id, title) => renameSession(id, title)}
        />
      )}

      {/* ── Scientific-computing panel ───────────────────────────────── */}
      {showSciencePanel && connection === "connected" && (
        <SciencePanel
          sections={panelSections}
          onRefresh={refreshPanel}
          onReload={reloadPanel}
          refreshing={panelPending}
          agentReady={panelAgentReady}
          onPrompt={sendPrompt}
        />
      )}

      {/* ── Sub-agents panel ─────────────────────────────────────────── */}
      {showSubAgentsPanel && connection === "connected" && (
        <SubAgentsPanel
          subAgents={subAgents}
          opened={openedSubAgent}
          onRefresh={refreshSubAgents}
          onOpen={panels.openSubAgent}
          onClose={() => setOpenedSubAgent(null)}
        />
      )}

      {/* ── Main area ───────────────────────────────────────────────── */}
      <div className="main">
        <StatusBar
          connection={connection}
          model={model}
          servedModels={endpointModels}
          runningSubAgents={runningSubAgents}
          showSessions={showSessionsPanel}
          showSubAgents={showSubAgentsPanel}
          showScience={showSciencePanel}
          onToggleSessions={() => setShowSessionsPanel((v) => !v)}
          onNewSession={startNewSession}
          onToggleSubAgents={() => setShowSubAgentsPanel((v) => !v)}
          onToggleScience={() => setShowSciencePanel((v) => !v)}
          onSwitchModel={(name: string) => send({ type: "set_model", model: name })}
          onCancelConnect={endpoint.cancelConnect}
          onDisconnect={departure.requestDisconnect}
        />

        {/* Asked only when disconnecting would end work in flight. Held, not fired:
            dismissing it cancels the disconnect. */}
        {departure.disconnectPending && (
          <DisconnectPrompt
            running={departure.runningSessions}
            onDetach={departure.detachThenDisconnect}
            onDiscard={departure.disconnectAnyway}
            onCancel={departure.cancelDisconnect}
          />
        )}

        {/* Conversations other than this one that are waiting on the user. Above the
            thread rather than in it: the card belongs to another transcript, and each
            one is a turn blocked on a person with no timeout — so it must be visible
            without being mistaken for part of this conversation. */}
        <ForeignPromptStrip
          prompts={foreignPrompts}
          onOpen={(sessionId) => {
            forgetForeign(sessionId);
            setSessionLoading(true);
            switchSession(sessionId);
          }}
          onApprove={(prompt, choice) => {
            send({
              type: "approval_response",
              id: prompt.id,
              session_id: prompt.session_id,
              choice,
            });
            forgetForeign(prompt.session_id as string);
          }}
        />

        {/* Chat thread, with the progress dock under it. The wrapper keeps the two
            together, so the dock sits at the bottom of the reading area rather than
            of the pane — above the composer, whose height changes as the user
            types. */}
        <div className="thread-area">
        <ChatThread
          messages={messages}
          busy={busy}
          draft={draft}
          draftSeq={draftSeq}
          liveToolCalls={liveToolCalls}
          liveThinkingBlocks={liveThinkingBlocks}
          loading={sessionLoading}
          chatThreadRef={chatThreadRef}
          bottomRef={bottomRef}
          lastUserMsgRef={lastUserMsgRef}
          onScroll={handleChatScroll}
          onApprovalResponse={handleApprovalResponse}
          onRetry={retryLastQuery}
          onDivert={handleDivert}
          emptyState={
            connection === "disconnected" ? (
              <div className="empty-state">
                <MimirIntro />
                <div className="empty-text">MIMIR</div>
                <ConnectForm
                  key={connectFormKey}
                  backend={backend}
                  vllmBaseUrl={vllmBaseUrl}
                  rayBaseUrl={rayBaseUrl}
                  ollamaBaseUrl={ollamaBaseUrl}
                  anthropicModels={anthropicModels}
                  models={endpointModels}
                  modelsError={modelsError}
                  modelsProbed={modelsProbed}
                  remembered={remembered}
                  onFetchModels={endpoint.fetchModels}
                  onConnect={endpoint.connect}
                />
              </div>
            ) : (
              <div className="empty-state">
                <MimirIntro loop={connection === "connecting" || connection === "error"} />
                <div className="empty-text">MIMIR</div>
                <div className="empty-hint">
                  {/* "error" is a failed attempt the host is still retrying, not an
                      agent coming up: ws_closed follows if the retries run out. */}
                  {connection === "connecting" || connection === "error"
                    ? "Connecting to agent…"
                    : agentReady
                      ? "Type a message to start"
                      : "Starting the agent — waiting for the model backend…"}
                </div>
                {(connection === "connecting" || connection === "error" || !agentReady) && (
                  <div className="loading-track" aria-hidden="true" />
                )}
                {(connection === "connecting" || connection === "error") && (
                  <button className="connect-btn cancel-connect-btn" onClick={endpoint.cancelConnect}>
                    Cancel connection
                  </button>
                )}
              </div>
            )
          }
        />
        <RunProgressDock runs={dockedRuns} onFocus={focusRun} />
        </div>

        {/* Reconnect panel — shown when disconnected with existing history */}
        {connection === "disconnected" && messages.length > 0 && (
          <div className="reconnect-panel">
            {runningServer ? (
              <>
                {/* A run is going without this panel. Rejoining it is the only thing
                    worth offering here: the form below would collect an endpoint that
                    a server already serving this workspace does not use. */}
                <div className="reconnect-hint">
                  {runningServer.detached
                    ? "MIMIR is still working without this window"
                    : "This workspace already has a MIMIR server"}
                  {runningServer.autonomy && runningServer.autonomy !== "manual"
                    ? ` — running under “${runningServer.autonomy}”`
                    : ""}
                </div>
                <button className="reconnect-btn" onClick={endpoint.reattach}>
                  Rejoin the run
                </button>
              </>
            ) : (
              <>
                <div className="reconnect-hint">Agent disconnected — reconnect to continue</div>
                {endpoint.hasLastConnect() && (
                  <button className="reconnect-btn" onClick={endpoint.reconnect}>
                    Reconnect
                  </button>
                )}
              </>
            )}
            <ConnectForm
              key={connectFormKey}
              backend={backend}
              vllmBaseUrl={vllmBaseUrl}
              rayBaseUrl={rayBaseUrl}
              ollamaBaseUrl={ollamaBaseUrl}
              anthropicModels={anthropicModels}
              models={endpointModels}
              modelsError={modelsError}
              modelsProbed={modelsProbed}
              remembered={remembered}
              onFetchModels={endpoint.fetchModels}
              onConnect={endpoint.connect}
            />
          </div>
        )}

        {/* Connecting banner (with history) */}
        {connection === "connecting" && messages.length > 0 && (
          <div className="reconnect-panel reconnect-panel--connecting">
            <span className="inline-spinner" aria-hidden="true" /> Connecting to agent…
            <button className="reconnect-btn cancel-connect-btn" onClick={endpoint.cancelConnect}>
              Cancel
            </button>
          </div>
        )}

        {/* Input area — only when connected */}
        {connection === "connected" && (
        <>
        {/* Batch review bar — accumulates file edits across queries until user accepts/reverts */}
        {workspace.batchFiles.length > 0 && (
          <BatchReviewBar
            files={workspace.batchFiles}
            onAccept={workspace.batch.acceptAll}
            onRevert={workspace.batch.revertAll}
            onAcceptFile={workspace.batch.acceptFile}
            onRevertFile={workspace.batch.revertFile}
          />
        )}
        {/* Plan bar — shown above input box when todos exist */}
        {workspace.todos.length > 0 && (
          <PlanBar items={workspace.todos} busy={busy} onClear={workspace.clearTodos} />
        )}
        {/* Clarification / plan-approval question — anchored just above the
            input box (not floating at the top of the chat) so the decision
            point sits where the user is already acting. */}
        {question.userQuestion && (
          <UserQuestion
            questions={question.userQuestion.questions}
            origin={question.userQuestion.origin}
            expiresAt={question.userQuestion.expiresAt}
            onSubmit={question.answer}
          />
        )}
        <div className="input-area">
          {/* Bottom toolbar: settings + disconnect */}
          <div className="bottom-toolbar">
            {/* Mode — its own control, colour-coded like the rest of the chat */}
            <ModeSwitcher mode={agentMode} onModeChange={settings.changeMode} />

            <div className="settings-wrapper">

              <button
                className="settings-btn"
                title="Agent settings"
                aria-label="Agent settings"
                aria-expanded={settingsOpen}
                onClick={() => setSettingsOpen((o) => !o)}
              >
                {/* The gear alone. The two pills beside it — reasoning on, streaming
                    off — stated what the popover they open already says, and a button
                    whose glyphs come and go is a button the eye has to re-find. */}
                <span className="settings-gear"><GearIcon /></span>
              </button>
              {settingsOpen && (
                <AgentSettings
                  thinkingLevel={thinkingLevel}
                  thinkingProfile={thinkingProfile}
                  streaming={streaming}
                  contextMode={contextMode}
                  enforcement={enforcement}
                  subAgentLevel={subAgentLevel}
                  onSubAgentLevelChange={panels.changeSubAgentLevel}
                  temperature={temperature}
                  onTemperatureChange={settings.changeTemperature}
                  onThinkingLevelChange={settings.changeThinkingLevel}
                  onStreamingToggle={settings.changeStreaming}
                  onContextModeChange={settings.changeContextMode}
                  onEnforcementChange={settings.changeEnforcement}
                  onClose={() => setSettingsOpen(false)}
                />
              )}
            </div>

            {/* Server / skill toggles */}
            <div className="toggles-wrapper">
              <button
                className="settings-btn"
                title="Servers & skills"
                aria-label="Servers and skills"
                aria-expanded={togglesOpen}
                onClick={() => {
                  setTogglesOpen((o) => {
                    if (!o) send({ type: "list_toggles" });
                    return !o;
                  });
                }}
              >
                <span className="settings-gear"><BlocksIcon /></span>
              </button>
              {togglesOpen && (
                <TogglesPanel
                  servers={workspace.serverToggles}
                  skills={workspace.skillToggles}
                  onToggleServer={(name, enabled) => send({ type: "toggle_server", name, enabled })}
                  onToggleSkill={(name, enabled) => send({ type: "toggle_skill", name, enabled })}
                  onClose={() => setTogglesOpen(false)}
                />
              )}
            </div>

            {/* Hard right, away from the controls that change what the next turn does:
                this one changes who owns the process, and it is pressed once a session
                rather than adjusted. The autonomy it runs under is the approval mode
                already set above the send button — it carries no picker of its own, so
                the two cannot disagree. */}
            <div className="toolbar-end">
              <DetachButton
                mode={approvalMode}
                detached={departure.detached}
                onDetach={departure.detach}
                onReattach={departure.reattach}
              />
            </div>
          </div>
          <Composer
            composer={composer}
            busy={busy}
            enabled={connection === "connected"}
            approvalMode={approvalMode}
            onApprovalModeChange={handleApprovalModeChange}
            onStop={() => send({ type: "command", text: "/cancel" })}
          />
        </div>
        {/* Context usage bar — shown below the input box */}
        {workspace.contextUsage && (
          <ContextBar usage={workspace.contextUsage} contextMode={contextMode} />
        )}
        </>
        )}
      </div>
    </div>
  );
};
