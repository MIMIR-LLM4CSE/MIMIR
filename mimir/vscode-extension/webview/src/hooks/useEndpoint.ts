// The connection to the agent: the socket's state, whether the agent behind it is up,
// the endpoint the connect form offers, and the connect, rejoin and cancel actions.

import { useCallback, useMemo, useRef, useState } from "react";
import type { ConnectionState, RememberedEndpoint, RunningServer } from "../types";
import type { ServerHandlers } from "../state/serverMessages";
import { DEFAULT_BACKEND, DEFAULT_OLLAMA_URL, DEFAULT_OPENAI_URL } from "../../../src/shared/defaults";

type ConnectArgs = [string, string, string, string | undefined, boolean | undefined];

/** What the extension host does for the panel (see useWebSocket). */
export interface Host {
  connect: (...args: ConnectArgs) => void;
  fetchModels: (backend: string, baseUrl: string) => void;
  disconnect: () => void;
}

export function useEndpoint(host: () => Host) {
  const [connection, setConnection] = useState<ConnectionState>("disconnected");
  // Whether the agent behind the socket exists yet: the socket opens first and the
  // worker then waits on the LLM backend, minutes on a cold vLLM.
  const [agentReady, setAgentReady] = useState(false);
  const [model, setModel] = useState("");
  const [anthropicModels, setAnthropicModels] = useState<string[]>([]);
  const [backend, setBackend] = useState(DEFAULT_BACKEND);
  const [vllmBaseUrl, setVllmBaseUrl] = useState(DEFAULT_OPENAI_URL);
  const [rayBaseUrl, setRayBaseUrl] = useState(DEFAULT_OPENAI_URL);
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState(DEFAULT_OLLAMA_URL);
  // The endpoint the host remembers: it seeds the form and is reconnected at startup.
  const [remembered, setRemembered] = useState<RememberedEndpoint | null>(null);
  // The server this workspace already has: it turns the connect form into a rejoin.
  const [runningServer, setRunningServer] = useState<RunningServer | null>(null);
  // Models the endpoint reports it serves, and whether it answered at all.
  const [endpointModels, setEndpointModels] = useState<string[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelsProbed, setModelsProbed] = useState(false);
  // The last connect's arguments (ours or the host's auto-connect), for Reconnect.
  const lastConnectArgsRef = useRef<ConnectArgs | null>(null);

  // The form reads its fields once, at mount: it is remounted when config arrives.
  const connectFormKey =
    `${backend}|${vllmBaseUrl}|${rayBaseUrl}|${ollamaBaseUrl}|${remembered?.baseUrl ?? ""}`;
  // Point the form at *baseUrl* for *be*, so after a disconnect it shows the address
  // last used, not the settings default.
  const seedAddress = useCallback((be: string, baseUrl: string) => {
    setBackend(be);
    if (be === "ollama") setOllamaBaseUrl(baseUrl);
    else if (be === "ray") setRayBaseUrl(baseUrl);
    else if (be === "vllm") setVllmBaseUrl(baseUrl);
  }, []);

  /** The socket went away (*state*: disconnected, or error while the host retries). */
  const lose = useCallback((state: ConnectionState) => {
    setConnection(state);
    setAgentReady(false);
  }, []);
  const connecting = useCallback(() => lose("connecting"), [lose]);

  const connect = useCallback((...args: ConnectArgs) => {
    const [mdl, be, baseUrl, , remember] = args;
    lastConnectArgsRef.current = args;
    // Named while we wait; `ready` confirms the name the server serves.
    setModel(mdl);
    setRemembered(remember ? { backend: be, baseUrl, model: mdl } : null);
    seedAddress(be, baseUrl);
    connecting();
    host().connect(...args);
  }, [host, seedAddress, connecting]);

  // Anthropic has no key-free listing: its models stay the static list from settings.
  const fetchModels = useCallback((be: string, baseUrl: string) => {
    setModelsError(null);
    setModelsProbed(false);
    if (be === "anthropic") {
      setEndpointModels([]);
      return;
    }
    host().fetchModels(be, baseUrl);
  }, [host]);

  /** Rejoin the server this workspace has: an ordinary connect, which the host turns
   *  into an attach. The arguments are placeholders a running server does not read. */
  const reattach = useCallback(() => {
    connecting();
    const args = lastConnectArgsRef.current;
    if (args) {
      host().connect(...args);
      return;
    }
    const baseUrl =
      backend === "ray" ? rayBaseUrl : backend === "ollama" ? ollamaBaseUrl : vllmBaseUrl;
    host().connect(runningServer?.model ?? "", backend, baseUrl, "", false);
  }, [host, connecting, runningServer, backend, vllmBaseUrl, rayBaseUrl, ollamaBaseUrl]);

  const reconnect = useCallback(() => {
    const args = lastConnectArgsRef.current;
    if (!args) return;
    connecting();
    host().connect(...args);
  }, [host, connecting]);

  /** Abort a connection in progress: the host tears down the server it is starting. */
  const cancelConnect = useCallback(() => {
    host().disconnect();
    lose("disconnected");
  }, [host, lose]);

  const reset = useCallback(() => {
    setModel("");
    lose("disconnected");
  }, [lose]);

  const on: ServerHandlers = useMemo(() => ({
    ready: (msg) => {
      setModel(msg.model);
      setConnection("connected");
      // Absent reads as ready: an older server sends no flag.
      setAgentReady(msg.agent_ready !== false);
    },
    config: (msg) => {
      setAnthropicModels(msg.anthropicModels ?? []);
      if (msg.backend) setBackend(msg.backend);
      if (msg.vllmBaseUrl) setVllmBaseUrl(msg.vllmBaseUrl);
      if (msg.rayBaseUrl) setRayBaseUrl(msg.rayBaseUrl);
      if (msg.ollamaBaseUrl) setOllamaBaseUrl(msg.ollamaBaseUrl);
      // A remembered endpoint outranks the settings default.
      const saved = msg.remembered ?? null;
      setRemembered(saved);
      setRunningServer(msg.running ?? null);
      if (saved) seedAddress(saved.backend, saved.baseUrl);
    },
    // The host reconnecting to the remembered endpoint by itself.
    auto_connect: (msg) => {
      lastConnectArgsRef.current = [msg.model, msg.backend, msg.baseUrl, undefined, true];
      setModel(msg.model);
      lose("connecting");
    },
    // An answer, whatever it says: an empty list is not a list never asked for.
    models: (msg) => {
      setEndpointModels(msg.models);
      setModelsError(msg.error ?? null);
      setModelsProbed(true);
    },
    // The connected server's own answer about its endpoint, which wins over the probe.
    served_models: (msg) => {
      setEndpointModels(msg.models);
      setModelsError(null);
      setModelsProbed(true);
    },
    model_changed: (msg) => setModel(msg.model),
    session_loaded: (msg) => {
      if (msg.agent_ready !== undefined) setAgentReady(msg.agent_ready);
    },
  }), [seedAddress, lose]);

  return {
    connection, agentReady, model, anthropicModels, backend, vllmBaseUrl, rayBaseUrl,
    ollamaBaseUrl, remembered, runningServer, endpointModels, modelsError, modelsProbed,
    connectFormKey, hasLastConnect: () => lastConnectArgsRef.current !== null,
    connect, fetchModels, reattach, reconnect, cancelConnect, lose, reset,
    on,
  };
}
