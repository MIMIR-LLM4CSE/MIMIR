import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  useReducer,
  useMemo,
} from "react";
import type {
  ClientMessage,
  ServerMessage,
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
import { useApprovals } from "./hooks/useApprovals";
import { makeId } from "./state/ids";
import { ChatThread } from "./components/ChatThread";
import { ForeignPromptStrip } from "./components/ForeignPromptStrip";
import { PlanBar } from "./components/PlanBar";
import { DisconnectPrompt } from "./components/DisconnectPrompt";
import { UserQuestion } from "./components/UserQuestion";
import { SessionsPanel } from "./components/SessionsPanel";
import { SubAgentsPanel } from "./components/SubAgentsPanel";
import { SciencePanel } from "./components/SciencePanel";
import { ContextBar } from "./components/ContextBar";
import { RunProgressDock } from "./components/RunProgressDock";
import { BatchReviewBar } from "./components/BatchReviewBar";
import { Composer } from "./components/Composer";
import { StatusBar } from "./components/StatusBar";
import { BottomToolbar } from "./components/BottomToolbar";
import { EmptyState } from "./components/EmptyState";
import { EndpointForm } from "./components/EndpointForm";
import { ReconnectPanel } from "./components/ReconnectPanel";
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
  const { messages, draft, draftSeq, busy, starting, liveToolCalls, liveThinkingBlocks } = chatState;
  // Mirror of reducer state for reads inside event callbacks (approval lookup,
  // session-load message preservation) without stale closures.
  const chatStateRef = useRef(chatState);
  chatStateRef.current = chatState;

  // The host's actions, read once useWebSocket below has created them.
  const hostRef = useRef<Host>({ connect: () => {}, fetchModels: () => {}, disconnect: () => {} });
  const endpoint = useEndpoint(useCallback(() => hostRef.current, []));
  const { connection, agentReady, model, runningServer, endpointModels } = endpoint;
  // The socket is opened below, by a hook that needs the message handler built from
  // these; parts that send go through this, which reads the socket once it exists.
  const sendRef = useRef<(msg: ClientMessage) => void>(() => {});
  const sendVia = useCallback((msg: ClientMessage) => sendRef.current(msg), []);
  const settings = useAgentSettings(sendVia);
  const { agentMode, contextMode, approvalMode } = settings;
  const workspace = useWorkspaceState(sendVia);
  const { onSessionLoaded: workspaceLoaded, reset: resetWorkspace } = workspace;
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
        // The worker's own greeting also ends the reducer's "agent starting".
        dispatch(msg);
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
  // On every "ready", the attachable MCP resources for the @-mention autocomplete.
  useEffect(() => {
    if (readyCount > 0) send({ type: "list_resources" });
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

  const approvals = useApprovals({
    send, dispatch, chatStateRef, activeSessionIdRef,
    changeApprovalMode: settings.changeApprovalMode,
  });

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
          starting={starting}
          draft={draft}
          draftSeq={draftSeq}
          liveToolCalls={liveToolCalls}
          liveThinkingBlocks={liveThinkingBlocks}
          loading={sessionLoading}
          chatThreadRef={chatThreadRef}
          bottomRef={bottomRef}
          lastUserMsgRef={lastUserMsgRef}
          onScroll={handleChatScroll}
          onApprovalResponse={approvals.respond}
          onRetry={retryLastQuery}
          onDivert={handleDivert}
          emptyState={
            <EmptyState
              connection={connection}
              agentReady={agentReady}
              form={<EndpointForm endpoint={endpoint} />}
              onCancelConnect={endpoint.cancelConnect}
            />
          }
        />
        <RunProgressDock runs={dockedRuns} onFocus={focusRun} />
        </div>

        {/* Under a thread with history while not connected: rejoin, reconnect, or the
            banner of a connection coming up. */}
        {messages.length > 0 && (
          <ReconnectPanel
            connection={connection}
            runningServer={runningServer}
            canReconnect={endpoint.hasLastConnect()}
            onRejoin={endpoint.reattach}
            onReconnect={endpoint.reconnect}
            onCancelConnect={endpoint.cancelConnect}
            form={<EndpointForm endpoint={endpoint} />}
          />
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
        {/* Hidden while a plan or a question waits: the turn is parked on that card, and
            a query sent beside it would start a second turn. The card's "Other" field
            is the way to say something else. */}
        {!question.userQuestion && (
        <div className="input-area">
          <BottomToolbar
            settings={settings}
            subAgentLevel={subAgentLevel}
            onSubAgentLevelChange={panels.changeSubAgentLevel}
            serverToggles={workspace.serverToggles}
            skillToggles={workspace.skillToggles}
            send={send}
            detached={departure.detached}
            onDetach={departure.detach}
            onReattach={departure.reattach}
          />
          <Composer
            composer={composer}
            busy={busy}
            enabled={connection === "connected"}
            approvalMode={approvalMode}
            onApprovalModeChange={approvals.changeMode}
            onStop={() => send({ type: "command", text: "/cancel" })}
          />
        </div>
        )}
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
