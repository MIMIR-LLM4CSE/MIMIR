import { useEffect, useRef, useCallback } from "react";
import type { ServerMessage, ClientMessage } from "../types";

// Acquire the VS Code API once at module level — calling it twice throws.
declare function acquireVsCodeApi(): { postMessage(msg: unknown): void };
const _vscode = (() => {
  try { return acquireVsCodeApi(); } catch { return null; }
})();

/** Send any message to the extension host from anywhere in the webview. */
export function vscodePostMessage(msg: unknown): void {
  _vscode?.postMessage(msg);
}

export interface UseWebSocketOptions {
  url: string;                       // kept for API compat (unused)
  onMessage: (msg: ServerMessage) => void;
  onClose?: () => void;
  onError?: (e: Event) => void;
  reconnectDelayMs?: number;         // unused (reconnect handled by extension host)
}

export interface UseWebSocketResult {
  send: (msg: ClientMessage) => void;
  close: () => void;
  connect: (model: string, backend: string, baseUrl: string, anthropicApiKey?: string) => void;
  fetchModels: (backend: string, baseUrl: string) => void;
  disconnect: () => void;
  getConfig: () => void;
  createSession: () => void;
  switchSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  renameSession: (sessionId: string, title: string) => void;
}

/**
 * In VS Code Remote SSH the webview cannot open WebSocket connections directly.
 * We route through the extension host via postMessage:
 *
 *   webview → postMessage({type:"ws_send", payload:"..."}) → ext host → Python
 *   webview ← postMessage({type:"ws",      payload:"..."}) ← ext host ← Python
 */
export function useWebSocket({
  onMessage,
  onClose,
  onError,
}: UseWebSocketOptions): UseWebSocketResult {
  const onMessageRef = useRef(onMessage);
  const onCloseRef   = useRef(onClose);
  const onErrorRef   = useRef(onError);
  onMessageRef.current = onMessage;
  onCloseRef.current   = onClose;
  onErrorRef.current   = onError;

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      const data = event.data as { type?: string; payload?: string };
      if (data.type === "ws" && data.payload) {
        try {
          const msg: ServerMessage = JSON.parse(data.payload);
          onMessageRef.current(msg);
        } catch (err) {
          console.warn("[mimir] dropped malformed WS frame", err, data.payload);
        }
      } else if (data.type === "ws_closed") {
        onCloseRef.current?.();
      } else if (data.type === "ws_error") {
        onErrorRef.current?.(event as unknown as Event);
      } else if (data.type === "config" || data.type === "active_editor") {
        // Messages emitted directly by the extension host (not the Python server) —
        // forward as ServerMessages for the App reducer/handler.
        onMessageRef.current(data as unknown as ServerMessage);
      }
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, []); // run once

  const send = useCallback((msg: ClientMessage) => {
    _vscode?.postMessage({ type: "ws_send", payload: JSON.stringify(msg) });
  }, []);

  const close = useCallback(() => { /* managed by extension host */ }, []);

  const disconnect = useCallback(() => {
    _vscode?.postMessage({ type: "disconnect" });
  }, []);

  const connect = useCallback((model: string, backend: string, baseUrl: string, anthropicApiKey?: string) => {
    _vscode?.postMessage({ type: "connect", model, backend, baseUrl, anthropicApiKey });
  }, []);

  const fetchModels = useCallback((backend: string, baseUrl: string) => {
    _vscode?.postMessage({ type: "fetch_models", backend, baseUrl });
  }, []);

  const getConfig = useCallback(() => {
    _vscode?.postMessage({ type: "get_config" });
  }, []);

  const createSession = useCallback(() => {
    _vscode?.postMessage({ type: "ws_send", payload: JSON.stringify({ type: "create_session" }) });
  }, []);

  const switchSession = useCallback((sessionId: string) => {
    _vscode?.postMessage({ type: "ws_send", payload: JSON.stringify({ type: "switch_session", session_id: sessionId }) });
  }, []);

  const deleteSession = useCallback((sessionId: string) => {
    _vscode?.postMessage({ type: "ws_send", payload: JSON.stringify({ type: "delete_session", session_id: sessionId }) });
  }, []);

  const renameSession = useCallback((sessionId: string, title: string) => {
    _vscode?.postMessage({ type: "ws_send", payload: JSON.stringify({ type: "rename_session", session_id: sessionId, title }) });
  }, []);

  return { send, close, connect, fetchModels, disconnect, getConfig, createSession, switchSession, deleteSession, renameSession };
}

