// The agent's settings as the panel shows them: mode, reasoning depth, streaming,
// context, enforcement, temperature and approvals. Each change is sent as the session
// command that sets it; the server's answer is what the panel then shows.

import { useCallback, useMemo, useState } from "react";
import type {
  AgentMode,
  ApprovalMode,
  ClientMessage,
  TemperatureState,
  ThinkingProfile,
} from "../types";
import type { ServerHandlers, ServerMessageOf } from "../state/serverMessages";

export type ContextMode = "compact" | "full";
export type Enforcement = "strict" | "light" | "off";

export function useAgentSettings(send: (msg: ClientMessage) => void) {
  const [agentMode, setAgentMode] = useState<AgentMode>("agent");
  // 0=off, 1=auto, 2=quick, 3=medium, 4=deep, 5=max
  const [thinkingLevel, setThinkingLevel] = useState<number>(1);
  // The rungs the served model honours, from "ready"; undefined reads as the full ladder.
  const [thinkingProfile, setThinkingProfile] = useState<ThinkingProfile | undefined>(undefined);
  const [streaming, setStreaming] = useState(true);
  const [contextMode, setContextMode] = useState<ContextMode>("full");
  const [enforcement, setEnforcement] = useState<Enforcement>("light");
  // Held by the server, per model and on disk: never replayed from here on connect.
  const [temperature, setTemperature] = useState<TemperatureState>({ supported: false, value: null });
  const [approvalMode, setApprovalMode] = useState<ApprovalMode>("manual");

  const command = useCallback((text: string) => send({ type: "command", text }), [send]);

  const changeMode = useCallback((mode: AgentMode) => {
    setAgentMode(mode);
    command(`/mode ${mode}`);
  }, [command]);
  const changeThinkingLevel = useCallback((level: number) => {
    setThinkingLevel(level);
    command(`/thinking-depth ${level}`);
  }, [command]);
  const changeStreaming = useCallback((on: boolean) => {
    setStreaming(on);
    command(`/streaming ${on ? "on" : "off"}`);
  }, [command]);
  const changeContextMode = useCallback((mode: ContextMode) => {
    setContextMode(mode);
    command(`/context ${mode}`);
  }, [command]);
  const changeEnforcement = useCallback((level: Enforcement) => {
    setEnforcement(level);
    command(`/enforcement ${level}`);
  }, [command]);
  // null = back to the model's own. The server answers with the value it holds.
  const changeTemperature = useCallback((value: number | null) => {
    setTemperature((t) => ({ ...t, value }));
    command(`/temperature ${value === null ? "default" : value}`);
  }, [command]);
  const changeApprovalMode = useCallback((mode: ApprovalMode) => {
    setApprovalMode(mode);
    // "all" is the spoken form of auto_all, the one the server's command parses.
    command(`/approvals ${mode === "auto_all" ? "all" : mode}`);
  }, [command]);

  /** What the server reports: on connect, on a model switch, and after each change. */
  const on: ServerHandlers = useMemo(() => ({
    ready: (msg: ServerMessageOf<"ready">) => {
      if (msg.context_mode) setContextMode(msg.context_mode);
      if (msg.enforcement) setEnforcement(msg.enforcement);
      if (msg.approval_mode) setApprovalMode(msg.approval_mode);
      if (msg.thinking) setThinkingProfile(msg.thinking);
      if (msg.temperature) setTemperature(msg.temperature);
    },
    model_changed: (msg: ServerMessageOf<"model_changed">) => {
      if (msg.thinking) setThinkingProfile(msg.thinking);
      if (msg.enforcement) setEnforcement(msg.enforcement);
      if (msg.temperature) setTemperature(msg.temperature);
    },
    context_mode: (msg) => setContextMode(msg.mode),
    enforcement: (msg) => setEnforcement(msg.mode),
    approval_mode: (msg) => setApprovalMode(msg.mode),
    mode: (msg) => setAgentMode(msg.mode),
    // The depth the agent holds, reported after any change.
    thinking_depth: (msg) => setThinkingLevel(msg.depth),
    streaming: (msg) => setStreaming(msg.enabled),
    temperature: (msg) => setTemperature({ supported: msg.supported, value: msg.value }),
  }), []);

  return {
    agentMode, thinkingLevel, thinkingProfile, streaming, contextMode, enforcement,
    temperature, approvalMode,
    changeMode, changeThinkingLevel, changeStreaming, changeContextMode,
    changeEnforcement, changeTemperature, changeApprovalMode,
    on,
  };
}
