import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  useLayoutEffect,
  useReducer,
  useMemo,
} from "react";
import type {
  ChatMessage,
  ConnectionState,
  ServerMessage,
  SessionMeta,
  TodoItem,
  DiffEntry,
  QuestionSpec,
  ToggleItem,
  ResourceItem,
  AgentMode,
  ThinkingProfile,
} from "./types";
import { createChatReducer, initialChatState } from "./state/chatReducer";
import { useWebSocket, vscodePostMessage } from "./hooks/useWebSocket";
import { ChatThread } from "./components/ChatThread";
import { PlanBar } from "./components/PlanBar";
import { AgentSettings } from "./components/AgentSettings";
import { ModeSwitcher } from "./components/ModeSwitcher";
import { TogglesPanel } from "./components/TogglesPanel";
import { ConnectForm } from "./components/ConnectForm";
import { ResumePlanPrompt } from "./components/ResumePlanPrompt";
import { ContinuePrompt } from "./components/ContinuePrompt";
import { UserQuestion } from "./components/UserQuestion";
import { SessionsPanel } from "./components/SessionsPanel";
import { ContextBar } from "./components/ContextBar";
import type { ContextUsage } from "./components/ContextBar";
import { BatchReviewBar } from "./components/BatchReviewBar";
import { MimirIntro } from "./components/MimirIntro";
import { pruneForStorage } from "./components/transcriptUtils";
import { MentionAutocomplete } from "./components/MentionAutocomplete";
import { SlashAutocomplete } from "./components/SlashAutocomplete";
import {
  detectMentionQuery,
  filterResources,
  applyMention,
  wrapIndex,
  type MentionQuery,
} from "./components/mentionUtils";
import {
  detectSlashQuery,
  filterSkills,
  applySlash,
  type SlashQuery,
} from "./components/slashUtils";

// VS Code webview API (optional — only present inside a webview).

function getWsUrl(): string {
  // Ask the extension host for config; fall back to default.
  return (window as unknown as { __MIMIR_WS_URL__?: string }).__MIMIR_WS_URL__
    ?? "ws://localhost:8765";
}

function makeId(): string {
  return crypto.randomUUID();
}

function modelDisplayName(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "MIMIR";
  const normalized = trimmed.replace(/\\/g, "/");
  if (!normalized.includes("/")) return trimmed;
  const parts = normalized.split("/").filter(Boolean);
  return parts[parts.length - 1] || trimmed;
}

export const App: React.FC = () => {
  // Streaming / thinking / approval state machine lives in a pure reducer so
  // there are no stale-closure refs and the logic is unit-testable.
  const chatReducer = useMemo(() => createChatReducer(makeId), []);
  const [chatState, dispatch] = useReducer(chatReducer, initialChatState);
  const { messages, draft, busy, liveToolCalls, liveThinkingBlocks } = chatState;
  // Mirror of reducer state for reads inside event callbacks (approval lookup,
  // session-load message preservation) without stale closures.
  const chatStateRef = useRef(chatState);
  chatStateRef.current = chatState;

  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("disconnected");
  const [model, setModel] = useState("");
  const [anthropicModels, setAnthropicModels] = useState<string[]>([]);
  const [backend, setBackend] = useState("vllm");
  const [vllmBaseUrl, setVllmBaseUrl] = useState("http://127.0.0.1:8000");
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState("http://127.0.0.1:11434");
  // Models the endpoint reports it serves — the connect form's dropdown.
  const [endpointModels, setEndpointModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  // ── @-mention autocomplete (attach MCP resources) ─────────────────────────
  const [resources, setResources] = useState<ResourceItem[]>([]);
  const [mention, setMention] = useState<MentionQuery | null>(null);
  const [mentionIndex, setMentionIndex] = useState(0);
  const mentionItems = useMemo(
    () => (mention ? filterResources(resources, mention.query) : []),
    [mention, resources]
  );
  const mentionOpen = mention !== null && mentionItems.length > 0;
  // ── "/"-command autocomplete (run a skill) ─────────────────────────────────
  // slashItems is derived after skillToggles is declared (see below).
  const [slash, setSlash] = useState<SlashQuery | null>(null);
  const [slashIndex, setSlashIndex] = useState(0);
  // The active VS Code editor file/selection, pushed by the extension host — shown as
  // an opt-in click-to-attach chip above the input.
  const [activeEditor, setActiveEditor] = useState<{ file: string; selection: string | null } | null>(null);
  const [agentMode, setAgentMode] = useState<AgentMode>("agent");
  const [thinkingLevel, setThinkingLevel] = useState<number>(1); // 0=off,1=auto,2=quick,3=medium,4=deep,5=max
  // Which rungs of that ladder the served model can honour — reported by the server
  // in "ready". Undefined until it arrives, which the control reads as the full ladder.
  const [thinkingProfile, setThinkingProfile] = useState<ThinkingProfile | undefined>(undefined);
  const thinking = thinkingLevel > 0;
  const [streaming, setStreaming] = useState(true);

  const [contextMode, setContextMode] = useState<"compact" | "full">("full");
  const [enforcement, setEnforcement] = useState<"strict" | "light" | "off">("strict");
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [togglesOpen, setTogglesOpen] = useState(false);
  const [serverToggles, setServerToggles] = useState<ToggleItem[]>([]);
  const [skillToggles, setSkillToggles] = useState<ToggleItem[]>([]);
  const slashItems = useMemo(
    () => (slash ? filterSkills(skillToggles, slash.query) : []),
    [slash, skillToggles]
  );
  const slashOpen = slash !== null && slashItems.length > 0;
  const [pendingResumeItems, setPendingResumeItems] = useState<TodoItem[] | null>(null);
  const [continuePrompt, setContinuePrompt] = useState<{ id: string; summary: string } | null>(null);
  const [userQuestion, setUserQuestion] = useState<{
    id: string;
    questions: QuestionSpec[];
  } | null>(null);
  // Batch review: accumulated file diffs across queries; persists until user accepts/reverts.
  const [batchFiles, setBatchFiles] = useState<DiffEntry[]>([]);
  // True between a session switch request and its session_loaded reply.
  const [sessionLoading, setSessionLoading] = useState(false);
  // Last submitted query text, so an error card can offer a one-click Retry.
  const lastQueryRef = useRef<string>("");

  // ── Session state ─────────────────────────────────────────────────────────
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  // Ref kept in sync so the WS handler always reads the current session ID
  // without stale-closure issues.
  const activeSessionIdRef = useRef<string | null>(null);
  const [showSessionsPanel, setShowSessionsPanel] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const chatThreadRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // True when the user has scrolled up — pause auto-scroll until back at bottom.
  const userScrolledUpRef = useRef(false);
  // Ref to the last user message element so we can scroll it to the top.
  const lastUserMsgRef = useRef<HTMLDivElement>(null);
  // Set on submit so the bottom-follow effect yields the next commit to
  // scrollQueryToTop (which anchors the new question at the top instead).
  const pendingScrollToTopRef = useRef(false);

  const scrollToBottom = useCallback(() => {
    if (userScrolledUpRef.current) return;
    const el = chatThreadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  const handleChatScroll = useCallback(() => {
    const el = chatThreadRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    userScrolledUpRef.current = !atBottom;
  }, []);

  // Scroll the newly submitted user message to the top of the chat pane.
  const scrollQueryToTop = useCallback(() => {
    requestAnimationFrame(() => {
      const el = lastUserMsgRef.current;
      const container = chatThreadRef.current;
      if (!el || !container) return;
      container.scrollTop = el.offsetTop - container.offsetTop - 8;
    });
  }, []);

  // Keep the view pinned to the bottom as the thread grows. This runs *after*
  // the DOM has committed (unlike the synchronous scrollToBottom() calls in the
  // message handler, which read a stale scrollHeight from before React rendered
  // the new content and therefore lag behind streaming tokens / tool rows). It
  // honours userScrolledUpRef, so a user who scrolled up is left undisturbed.
  useLayoutEffect(() => {
    // A fresh submit anchors the new question at the top (scrollQueryToTop),
    // so skip the bottom-follow for that one commit to avoid a bottom→top jump.
    if (pendingScrollToTopRef.current) {
      pendingScrollToTopRef.current = false;
      return;
    }
    scrollToBottom();
  }, [messages, draft, liveToolCalls, liveThinkingBlocks, scrollToBottom]);

  // Force-scrolls to bottom when an approval arrives so the live editing card
  // is visible regardless of whether the user had scrolled up.
  const scrollToApproval = useCallback(() => {
    setTimeout(() => {
      userScrolledUpRef.current = false;
      scrollToBottom();
    }, 80);
  }, [scrollToBottom]);

  // ── WebSocket message handler ─────────────────────────────────────────────
  // Scalar/config message types update local React state directly; the
  // streaming/thinking/approval/diff state machine is delegated to the reducer.
  const handleServerMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case "ready":
        setModel(msg.model);
        setConnection("connected");
        if (msg.context_mode) setContextMode(msg.context_mode);
        if (msg.enforcement) setEnforcement(msg.enforcement);
        if (msg.thinking) setThinkingProfile(msg.thinking);
        handleReady();
        return;

      case "open_editor":
        // The agent wrote a file it wants the user to read (e.g. a plan .md).
        // Forward to the extension host. `open_preview` renders Markdown files
        // (like plan .md) in VS Code's Markdown preview; other files open as a
        // normal editor tab. Absolute paths supported.
        if (msg.path) vscodePostMessage({ type: "open_preview", file: msg.path });
        return;

      case "config":
        setAnthropicModels(msg.anthropicModels ?? []);
        if (msg.backend) setBackend(msg.backend);
        if (msg.vllmBaseUrl) setVllmBaseUrl(msg.vllmBaseUrl);
        if (msg.ollamaBaseUrl) setOllamaBaseUrl(msg.ollamaBaseUrl);
        return;

      case "models":
        setEndpointModels(msg.models);
        setModelsError(msg.error ?? null);
        setModelsLoading(false);
        return;

      case "todo":
        setTodos(msg.items);
        return;

      case "todo_prompt":
        // Don't auto-restore — ask the user.
        setPendingResumeItems(msg.items);
        return;

      case "batch_status":
        // Persistent batch review bar — accumulated file diffs.
        setBatchFiles(msg.files ?? []);
        return;

      case "sessions_list":
        setSessions(msg.sessions);
        return;

      case "toggles_list":
        setServerToggles(msg.servers ?? []);
        setSkillToggles(msg.skills ?? []);
        return;

      case "resources":
        setResources(msg.resources ?? []);
        return;

      case "active_editor":
        setActiveEditor({ file: msg.file, selection: msg.selection });
        return;

      case "context_mode":
        setContextMode(msg.mode);
        return;

      case "enforcement":
        setEnforcement(msg.mode);
        return;

      case "mode":
        setAgentMode(msg.mode);
        return;

      case "continue_prompt":
        setContinuePrompt({ id: msg.id, summary: msg.summary });
        return;

      case "user_question":
        setUserQuestion({
          id: msg.id,
          questions: msg.questions,
        });
        return;

      case "context_usage":
        setContextUsage({
          used_tokens: msg.used_tokens,
          total_tokens: msg.total_tokens,
          reserved_tokens: msg.reserved_tokens,
          overhead_tokens: msg.overhead_tokens,
          history_messages: msg.history_messages,
          history_messages_full: msg.history_messages_full,
        });
        return;

      case "session_loaded": {
        // Capture BEFORE updating so we can compare old vs new session id.
        const prevSessionId = activeSessionIdRef.current;
        activeSessionIdRef.current = msg.session_id;
        setActiveSessionId(msg.session_id);
        const prevMessages = chatStateRef.current.messages;
        let nextMessages: ChatMessage[];
        if (msg.display_messages && msg.display_messages.length > 0)
          // Server sent saved messages → always use them.
          nextMessages = msg.display_messages;
        else if (prevSessionId === msg.session_id && prevMessages.length > 0)
          // Same session reconnecting (in-memory only) → preserve chat so a
          // transient WS drop doesn't wipe the conversation.
          nextMessages = prevMessages;
        else
          // Genuinely new / different session with no messages → clear.
          nextMessages = [];
        dispatch({ type: "session_loaded_messages", messages: nextMessages });
        setTodos(msg.todos ?? []);
        if (prevSessionId !== msg.session_id) {
          // Pending prompts belong to the turn of the session we just left — the
          // server cancels that turn on a switch, so answering them here would
          // reply to nothing (and a plan-approval card would linger forever).
          // A reconnect to the *same* session keeps its cards: the worker is
          // still parked on them and will never re-send them.
          setUserQuestion(null);
          setContinuePrompt(null);
          setPendingResumeItems(null);
          setBatchFiles([]);
        }
        setContextUsage(null);
        setSessionLoading(false);
        scrollToBottom();
        return;
      }

      case "diff": {
        // BatchReviewBar update lives outside the reducer; the editing-card
        // message mutation is delegated to the reducer.
        const { file: dFile, patch: dPatch } = msg;
        const isNewFile = !dPatch || dPatch.includes("/dev/null");
        setBatchFiles((prev) => {
          const idx = prev.findIndex((d) => d.file === dFile);
          if (idx >= 0)
            return prev.map((d, i) =>
              i === idx ? { ...d, patch: dPatch, is_new: isNewFile } : d
            );
          return [...prev, { file: dFile, patch: dPatch, is_new: isNewFile }];
        });
        dispatch(msg);
        return;
      }
    }

    // Remaining message-state types (output/status/token/file_progress/
    // approval/thinking*/answer/error) are handled by the reducer; dispatch and
    // run the matching scroll side-effect afterwards.
    dispatch(msg);
    if (msg.type === "approval") {
      scrollToApproval();
    } else if (msg.type === "file_progress") {
      if ((msg.diffs ?? []).length > 0) scrollToApproval();
    } else if (
      msg.type === "output" || msg.type === "status" ||
      msg.type === "token" || msg.type === "thinking" ||
      msg.type === "answer" || msg.type === "error" ||
      msg.type === "tool_call" || msg.type === "subagent_event"
    ) {
      scrollToBottom();
    }
  }, [scrollToBottom, scrollToApproval]);

  const resetSession = () => {
    dispatch({ type: "reset" });
    setTodos([]);
    setModel("");
    setPendingResumeItems(null);
    setContinuePrompt(null);
    setUserQuestion(null);
    setActiveSessionId(null);
    setSessions([]);
    setConnection("disconnected");
  };

  const { send, getConfig, connect, fetchModels, disconnect, createSession, switchSession, deleteSession, renameSession } = useWebSocket({
    url: getWsUrl(),
    onMessage: handleServerMessage,
    onClose: () => {
      // Preserve messages and todos — only reset connection/busy state so the
      // chat history survives a transient drop (e.g. during batch-revert I/O).
      dispatch({ type: "connection_lost" });
      setConnection("disconnected");
    },
    onError: () => {
      // Surface the error state; ws_closed follows and drives reconnect UI.
      dispatch({ type: "connection_lost" });
      setConnection("error");
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

  // Hand the rendered chat to the server so a reload can show it again.
  //
  // The server only ever recorded the text bubbles: tool rows, reasoning panels and
  // diff cards are assembled by the reducer and live nowhere else, which is why a
  // reconnect used to come back stripped to prose. Sending happens when a turn ends —
  // `busy` going false — and never mid-stream, so a turn costs one frame rather than
  // one per token. The delay lets the last few events (a trailing verdict, the batch
  // status) land in the transcript before it is sent.
  const wasBusyRef = useRef(false);
  useEffect(() => {
    const wasBusy = wasBusyRef.current;
    wasBusyRef.current = chatState.busy;
    if (chatState.busy || !wasBusy) return;
    const timer = setTimeout(() => {
      const sessionId = activeSessionIdRef.current;
      const messages = chatStateRef.current.messages;
      if (!sessionId || messages.length === 0) return;
      send({ type: "transcript", session_id: sessionId, messages: pruneForStorage(messages) });
    }, 1000);
    return () => clearTimeout(timer);
  }, [chatState.busy, send]);

  // ── User actions ──────────────────────────────────────────────────────────

  const submitQuery = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    setMention(null);
    setSlash(null);
    // Chat-while-busy: a message sent during a run is a "steer" — queued and
    // injected into the agent's current turn at its next step, not a new query.
    if (busy) {
      dispatch({ type: "steer_query", text });
      send({ type: "steer", text });
      return;
    }
    lastQueryRef.current = text;
    dispatch({ type: "submit_query", text });
    userScrolledUpRef.current = false; // reset scroll lock on new query
    pendingScrollToTopRef.current = true; // let scrollQueryToTop win this commit
    send({ type: "query", text });
    scrollQueryToTop();
  }, [input, busy, send, scrollQueryToTop]);

  // Re-run the most recent query — surfaced as Retry on the latest error card.
  const retryLastQuery = useCallback(() => {
    const text = lastQueryRef.current;
    if (!text || busy) return;
    dispatch({ type: "submit_query", text });
    userScrolledUpRef.current = false;
    pendingScrollToTopRef.current = true; // let scrollQueryToTop win this commit
    send({ type: "query", text });
    scrollQueryToTop();
  }, [busy, send, scrollQueryToTop]);

  // Remembers the exact args of the last successful connect so the disconnected
  // panel can offer a one-click Reconnect (replaying model/backend/address).
  const lastConnectArgsRef = useRef<Parameters<typeof connect> | null>(null);
  const handleConnect = useCallback(
    (mdl: string, be: string, baseUrl: string, anthropicApiKey?: string) => {
      lastConnectArgsRef.current = [mdl, be, baseUrl, anthropicApiKey];
      setConnection("connecting");
      connect(mdl, be, baseUrl, anthropicApiKey);
    },
    [connect],
  );

  // Ask the extension host what the endpoint at *baseUrl* serves. Anthropic has
  // no key-free listing, so its models stay the static list from settings.
  const handleFetchModels = useCallback(
    (be: string, baseUrl: string) => {
      if (be === "anthropic") {
        setEndpointModels([]);
        setModelsError(null);
        return;
      }
      setModelsLoading(true);
      setModelsError(null);
      fetchModels(be, baseUrl);
    },
    [fetchModels],
  );
  const handleReconnect = useCallback(() => {
    const args = lastConnectArgsRef.current;
    if (!args) return;
    setConnection("connecting");
    connect(...args);
  }, [connect]);

  // Abort an in-progress connection: tell the host to tear down the server it
  // is starting and drop straight back to idle.
  const handleCancelConnect = useCallback(() => {
    disconnect();
    setConnection("disconnected");
  }, [disconnect]);

  // Recompute the @-mention query from the textarea's current value + caret.
  const syncMention = useCallback((el: HTMLTextAreaElement | null) => {
    if (!el) return;
    const caret = el.selectionStart ?? el.value.length;
    setMention(detectMentionQuery(el.value, caret));
    setMentionIndex(0);
  }, []);

  // Recompute the "/"-command query from the textarea's current value + caret.
  const syncSlash = useCallback((el: HTMLTextAreaElement | null) => {
    if (!el) return;
    const caret = el.selectionStart ?? el.value.length;
    setSlash(detectSlashQuery(el.value, caret));
    setSlashIndex(0);
  }, []);

  // Insert the chosen resource's token at the caret and refocus the input.
  const pickMention = useCallback(
    (item: ResourceItem) => {
      const el = textareaRef.current;
      if (!el || !mention) return;
      const caret = el.selectionStart ?? el.value.length;
      const { text, caret: newCaret } = applyMention(
        el.value,
        mention.start,
        caret,
        item.name || item.uri
      );
      setInput(text);
      setMention(null);
      requestAnimationFrame(() => {
        const t = textareaRef.current;
        if (t) {
          t.focus();
          t.setSelectionRange(newCaret, newCaret);
        }
      });
    },
    [mention]
  );

  // Insert the chosen skill as a "/name " slash command and refocus the input.
  const pickSlash = useCallback(
    (item: ToggleItem) => {
      const el = textareaRef.current;
      if (!el || !slash) return;
      const caret = el.selectionStart ?? el.value.length;
      const { text, caret: newCaret } = applySlash(
        el.value,
        slash.start,
        caret,
        item.name
      );
      setInput(text);
      setSlash(null);
      requestAnimationFrame(() => {
        const t = textareaRef.current;
        if (t) {
          t.focus();
          t.setSelectionRange(newCaret, newCaret);
        }
      });
    },
    [slash]
  );

  // Click-to-attach the active editor file (+ selection) — inserts an @path[:a-b]
  // token at the caret, reusing the same mention mechanism the backend resolves.
  const attachActiveEditor = useCallback(() => {
    if (!activeEditor) return;
    const token = activeEditor.selection
      ? `${activeEditor.file}:${activeEditor.selection}`
      : activeEditor.file;
    const el = textareaRef.current;
    const caret = el?.selectionStart ?? input.length;
    const { text, caret: newCaret } = applyMention(input, caret, caret, token);
    setInput(text);
    requestAnimationFrame(() => {
      const t = textareaRef.current;
      if (t) {
        t.focus();
        t.setSelectionRange(newCaret, newCaret);
      }
    });
  }, [activeEditor, input]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // While the slash-command dropdown is open, arrows/Enter/Tab/Esc drive it.
      if (slashOpen) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          setSlashIndex((i) => wrapIndex(i + 1, slashItems.length));
          return;
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          setSlashIndex((i) => wrapIndex(i - 1, slashItems.length));
          return;
        }
        if (e.key === "Enter" || e.key === "Tab") {
          e.preventDefault();
          const item = slashItems[wrapIndex(slashIndex, slashItems.length)];
          if (item) pickSlash(item);
          return;
        }
        if (e.key === "Escape") {
          e.preventDefault();
          setSlash(null);
          return;
        }
      }
      // While the mention dropdown is open, arrows/Enter/Tab/Esc drive it.
      if (mentionOpen) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          setMentionIndex((i) => wrapIndex(i + 1, mentionItems.length));
          return;
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          setMentionIndex((i) => wrapIndex(i - 1, mentionItems.length));
          return;
        }
        if (e.key === "Enter" || e.key === "Tab") {
          e.preventDefault();
          const item = mentionItems[wrapIndex(mentionIndex, mentionItems.length)];
          if (item) pickMention(item);
          return;
        }
        if (e.key === "Escape") {
          e.preventDefault();
          setMention(null);
          return;
        }
      }
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitQuery();
      }
    },
    [
      slashOpen,
      slashItems,
      slashIndex,
      pickSlash,
      mentionOpen,
      mentionItems,
      mentionIndex,
      pickMention,
      submitQuery,
    ]
  );

  const handleApprovalResponse = useCallback(
    (id: string, choice: "y" | "n" | "a", approvedFiles?: string[]) => {
      // Send approval_response for every merged ID so all blocked shims unblock.
      const card = chatStateRef.current.messages.find(
        (m) => m.kind === "approval" && m.approval?.id === id
      );
      const allIds = card?.approval?.ids ?? [id];
      for (const aid of allIds) {
        const payload: { type: "approval_response"; id: string; choice: "y" | "n" | "a"; approved_files?: string[] } = {
          type: "approval_response", id: aid, choice,
        };
        if (approvedFiles !== undefined) payload.approved_files = approvedFiles;
        send(payload);
      }
      // Update the card(s) and set busy via the reducer.
      dispatch({ type: "approval_response", id, choice });
    },
    [send]
  );

  const handleModeChange = useCallback((mode: AgentMode) => {
    setAgentMode(mode);
    send({ type: "command", text: `/mode ${mode}` });
  }, [send]);

  const handleThinkingLevelChange = useCallback((level: number) => {
    setThinkingLevel(level);
    send({ type: "command", text: `/thinking-depth ${level}` });
  }, [send]);

  const handleStreamingToggle = useCallback((val: boolean) => {
    setStreaming(val);
    send({ type: "command", text: `/streaming ${val ? "on" : "off"}` });
  }, [send]);

  const handleContextModeChange = useCallback((val: "compact" | "full") => {
    setContextMode(val);
    send({ type: "command", text: `/context ${val}` });
  }, [send]);

  const handleEnforcementChange = useCallback((val: "strict" | "light" | "off") => {
    setEnforcement(val);
    send({ type: "command", text: `/enforcement ${val}` });
  }, [send]);

  const connectionColor =
    connection === "connected"
      ? "var(--vscode-testing-iconPassed)"
      : connection === "error" || connection === "disconnected"
      ? "var(--vscode-testing-iconFailed)"
      : "var(--vscode-descriptionForeground)";

  const hasTodos = todos.length > 0;

  // data-mode drives the chat's accent colour: blue agent / red plan / green ask.
  return (
    <div className={`app ${showSessionsPanel ? "with-sessions" : ""}`} data-mode={agentMode}>
      {pendingResumeItems && (
        <ResumePlanPrompt
          items={pendingResumeItems}
          onChoice={(resume) => {
            (send as any)({ type: "resume_plan", choice: resume ? "yes" : "no" });
            if (resume) setTodos(pendingResumeItems);
            setPendingResumeItems(null);
          }}
        />
      )}

      {continuePrompt && (
        <ContinuePrompt
          summary={continuePrompt.summary}
          onChoice={(cont) => {
            send({ type: "continue_response", id: continuePrompt.id, choice: cont ? "y" : "n" });
            setContinuePrompt(null);
          }}
        />
      )}

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

      {/* ── Main area ───────────────────────────────────────────────── */}
      <div className="main">
        {/* Status bar — connection + model name + disconnect */}
        <div className="status-bar">
          {connection === "connected" && (
            <>
              <button
                className={`sessions-toggle-btn ${showSessionsPanel ? "active" : ""}`}
                title="Chat history"
                aria-label="Toggle chat history"
                aria-pressed={showSessionsPanel}
                onClick={() => setShowSessionsPanel((v) => !v)}
              >
                ☰
              </button>
              {/* New session next to the history toggle — starting a chat
                  shouldn't require opening the panel first. */}
              <button
                className="sessions-toggle-btn sessions-new-toggle-btn"
                title="New session"
                aria-label="New session"
                onClick={() => startNewSession()}
              >
                ＋
              </button>
            </>
          )}
          <span
            className="connection-dot"
            style={{ color: connectionColor }}
            title={connection}
            role="img"
            aria-label={`Connection: ${connection}`}
          >
            ●
          </span>
          <span className="model-name">{modelDisplayName(model)}</span>
          {connection === "connected" && (
            <button
              className="disconnect-btn"
              title="Disconnect agent"
              aria-label="Disconnect agent"
              onClick={() => { disconnect(); resetSession(); }}
            >
              ⏏
            </button>
          )}
        </div>

        {/* Chat thread */}
        <ChatThread
          messages={messages}
          busy={busy}
          draft={draft}
          liveToolCalls={liveToolCalls}
          liveThinkingBlocks={liveThinkingBlocks}
          loading={sessionLoading}
          chatThreadRef={chatThreadRef}
          bottomRef={bottomRef}
          lastUserMsgRef={lastUserMsgRef}
          onScroll={handleChatScroll}
          onApprovalResponse={handleApprovalResponse}
          onRetry={retryLastQuery}
          emptyState={
            connection === "disconnected" ? (
              <div className="empty-state">
                <MimirIntro />
                <div className="empty-text">MIMIR</div>
                <ConnectForm
                  backend={backend}
                  vllmBaseUrl={vllmBaseUrl}
                  ollamaBaseUrl={ollamaBaseUrl}
                  anthropicModels={anthropicModels}
                  models={endpointModels}
                  modelsLoading={modelsLoading}
                  modelsError={modelsError}
                  onFetchModels={handleFetchModels}
                  onConnect={handleConnect}
                />
              </div>
            ) : (
              <div className="empty-state">
                <MimirIntro loop={connection === "connecting"} />
                <div className="empty-text">MIMIR</div>
                <div className="empty-hint">
                  {connection === "connecting" ? "Connecting to agent…" : "Type a message to start"}
                </div>
                {connection === "connecting" && (
                  <button className="connect-btn cancel-connect-btn" onClick={handleCancelConnect}>
                    Cancel connection
                  </button>
                )}
              </div>
            )
          }
        />

        {/* Reconnect panel — shown when disconnected with existing history */}
        {connection === "disconnected" && messages.length > 0 && (
          <div className="reconnect-panel">
            <div className="reconnect-hint">Agent disconnected — reconnect to continue</div>
            {lastConnectArgsRef.current && (
              <button className="reconnect-btn" onClick={handleReconnect}>
                Reconnect
              </button>
            )}
            <ConnectForm
              backend={backend}
              vllmBaseUrl={vllmBaseUrl}
              ollamaBaseUrl={ollamaBaseUrl}
              anthropicModels={anthropicModels}
              models={endpointModels}
              modelsLoading={modelsLoading}
              modelsError={modelsError}
              onFetchModels={handleFetchModels}
              onConnect={handleConnect}
            />
          </div>
        )}

        {/* Connecting banner (with history) */}
        {connection === "connecting" && messages.length > 0 && (
          <div className="reconnect-panel reconnect-panel--connecting">
            <span className="inline-spinner" aria-hidden="true" /> Connecting to agent…
            <button className="reconnect-btn cancel-connect-btn" onClick={handleCancelConnect}>
              Cancel
            </button>
          </div>
        )}

        {/* Input area — only when connected */}
        {connection === "connected" && (
        <>
        {/* Batch review bar — accumulates file edits across queries until user accepts/reverts */}
        {batchFiles.length > 0 && (
          <BatchReviewBar
            files={batchFiles}
            onAccept={() => {
              setBatchFiles([]);
              (send as any)({ type: "batch_review_accept" });
            }}
            onRevert={() => {
              setBatchFiles([]);
              (send as any)({ type: "batch_review_revert" });
            }}
            onAcceptFile={(file) => {
              setBatchFiles((prev) => prev.filter((f) => f.file !== file));
              (send as any)({ type: "batch_review_accept_file", file });
            }}
            onRevertFile={(file) => {
              setBatchFiles((prev) => prev.filter((f) => f.file !== file));
              (send as any)({ type: "batch_review_revert_file", file });
            }}
          />
        )}
        {/* Plan bar — shown above input box when todos exist */}
        {hasTodos && <PlanBar items={todos} busy={busy} onClear={() => {
          (send as any)({ type: "clear_todos" });
          setTodos([]);
        }} />}
        {/* Clarification / plan-approval question — anchored just above the
            input box (not floating at the top of the chat) so the decision
            point sits where the user is already acting. */}
        {userQuestion && (
          <UserQuestion
            questions={userQuestion.questions}
            onSubmit={(answers) => {
              send({
                type: "user_question_response",
                id: userQuestion.id,
                answers,
              });
              setUserQuestion(null);
            }}
          />
        )}
        <div className="input-area">
          {/* Bottom toolbar: settings + disconnect */}
          <div className="bottom-toolbar">
            {/* Mode — its own control, colour-coded like the rest of the chat */}
            <ModeSwitcher mode={agentMode} onModeChange={handleModeChange} />

            <div className="settings-wrapper">

              <button
                className="settings-btn"
                title="Agent settings"
                aria-label="Agent settings"
                aria-expanded={settingsOpen}
                onClick={() => setSettingsOpen((o) => !o)}
              >
                {thinking && <span className="settings-cap-pill">💭</span>}
                {!streaming && <span className="settings-cap-pill">⏸</span>}
                <span className="settings-gear">⚙</span>
              </button>
              {settingsOpen && (
                <AgentSettings
                  thinkingLevel={thinkingLevel}
                  thinkingProfile={thinkingProfile}
                  streaming={streaming}
                  contextMode={contextMode}
                  enforcement={enforcement}
                  onThinkingLevelChange={handleThinkingLevelChange}
                  onStreamingToggle={handleStreamingToggle}
                  onContextModeChange={handleContextModeChange}
                  onEnforcementChange={handleEnforcementChange}
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
                <span className="settings-gear">🧩</span>
              </button>
              {togglesOpen && (
                <TogglesPanel
                  servers={serverToggles}
                  skills={skillToggles}
                  onToggleServer={(name, enabled) => send({ type: "toggle_server", name, enabled })}
                  onToggleSkill={(name, enabled) => send({ type: "toggle_skill", name, enabled })}
                  onClose={() => setTogglesOpen(false)}
                />
              )}
            </div>
          </div>
          {activeEditor && connection === "connected" && (
            <div className="active-file-bar">
              <button
                className="active-file-chip"
                onClick={attachActiveEditor}
                title="Attach the active editor file (and selection) to your message"
              >
                <span className="af-icon">📎</span>
                <span className="af-name">
                  {activeEditor.file}
                  {activeEditor.selection ? `:${activeEditor.selection}` : ""}
                </span>
                <span className="af-add">＋</span>
              </button>
            </div>
          )}
          <div className="input-row">
          {mentionOpen && (
            <MentionAutocomplete
              items={mentionItems}
              activeIndex={wrapIndex(mentionIndex, mentionItems.length)}
              onPick={pickMention}
              onHover={setMentionIndex}
            />
          )}
          {slashOpen && (
            <SlashAutocomplete
              items={slashItems}
              activeIndex={wrapIndex(slashIndex, slashItems.length)}
              onPick={pickSlash}
              onHover={setSlashIndex}
            />
          )}
          <textarea
            ref={textareaRef}
            className="chat-input"
            rows={3}
            placeholder={
              busy
                ? "Steer MIMIR while it works…  (Enter to send)"
                : "Message MIMIR  (Enter to send, Shift+Enter for newline, @ to attach, / for commands)"
            }
            value={input}
            disabled={connection !== "connected"}
            onChange={(e) => {
              setInput(e.target.value);
              syncMention(e.target);
              syncSlash(e.target);
            }}
            onSelect={(e) => {
              syncMention(e.currentTarget);
              syncSlash(e.currentTarget);
            }}
            onBlur={() => {
              setMention(null);
              setSlash(null);
            }}
            onKeyDown={handleKeyDown}
          />
          {busy ? (
            <>
              {input.trim() && (
                <button
                  className="send-btn"
                  onClick={submitQuery}
                  title="Send steer (Enter) — queued for the agent's next step"
                  aria-label="Send steer message"
                >
                  ↑
                </button>
              )}
              <button
                className="stop-btn"
                onClick={() => send({ type: "command", text: "/cancel" })}
                title="Stop (cancel current query)"
                aria-label="Stop current query"
              >
                ⏹
              </button>
            </>
          ) : (
            <button
              className="send-btn"
              onClick={submitQuery}
              disabled={!input.trim() || connection !== "connected"}
              title="Send (Enter)"
              aria-label="Send message"
            >
              ↑
            </button>
          )}
          </div>
        </div>
        {/* Context usage bar — shown below the input box */}
        {contextUsage && (
          <ContextBar usage={contextUsage} contextMode={contextMode} />
        )}
        </>
        )}
      </div>
    </div>
  );
};
