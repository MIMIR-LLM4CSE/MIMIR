import React, { useState, useEffect, useRef } from "react";

interface Props {
  /** Backend selected in settings — the value the form starts on. */
  backend?: string;
  /** Default addresses from settings; the user can edit them here. */
  vllmBaseUrl?: string;
  ollamaBaseUrl?: string;
  /** Claude model ids (static — the hosted API is not queried without a key). */
  anthropicModels?: string[];
  /** Models the endpoint reports it serves, fetched by the extension host. */
  models?: string[];
  modelsLoading?: boolean;
  modelsError?: string | null;
  onFetchModels: (backend: string, baseUrl: string) => void;
  onConnect: (model: string, backend: string, baseUrl: string, anthropicApiKey?: string) => void;
}

/** Address the form starts on for a given backend. */
function defaultUrl(backend: string, vllm: string, ollama: string): string {
  return backend === "ollama" ? ollama : vllm;
}

export const ConnectForm: React.FC<Props> = ({
  backend: initBackend = "vllm",
  vllmBaseUrl = "http://127.0.0.1:8000",
  ollamaBaseUrl = "http://127.0.0.1:11434",
  anthropicModels = [],
  models = [],
  modelsLoading = false,
  modelsError = null,
  onFetchModels,
  onConnect,
}) => {
  const [backend, setBackend] = useState(initBackend);
  const [url, setUrl] = useState(defaultUrl(initBackend, vllmBaseUrl, ollamaBaseUrl));
  // Claude API key — kept in webview state only; forwarded on connect and never
  // persisted. Left blank means "use whatever ANTHROPIC_API_KEY the host exports".
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");

  const anthropic = backend === "anthropic";
  const options = anthropic ? anthropicModels : models;

  const handleBackendChange = (next: string) => {
    setBackend(next);
    setModel("");
    if (next !== "anthropic") setUrl(defaultUrl(next, vllmBaseUrl, ollamaBaseUrl));
  };

  // Ask the endpoint what it serves, debounced so typing an address doesn't fire
  // a request per keystroke. The ⟳ button below bypasses the wait.
  const urlRef = useRef(url);
  urlRef.current = url;
  useEffect(() => {
    if (anthropic) return;
    const t = setTimeout(() => onFetchModels(backend, urlRef.current), 600);
    return () => clearTimeout(t);
  }, [backend, url, anthropic, onFetchModels]);

  // Keep the selection valid as the list changes (backend switch, refresh).
  useEffect(() => {
    if (options.length > 0 && !options.includes(model)) setModel(options[0]);
    if (options.length === 0 && model) setModel("");
  }, [options]);

  return (
    <div className="connect-form">
      <div className="connect-field">
        <label className="connect-label">Backend</label>
        <select
          className="connect-select"
          value={backend}
          onChange={e => handleBackendChange(e.target.value)}
        >
          <option value="vllm">vLLM</option>
          <option value="ollama">Ollama</option>
          <option value="anthropic">Anthropic (Claude)</option>
        </select>
      </div>

      {anthropic ? (
        <div className="connect-field">
          <label className="connect-label">API key</label>
          <input
            className="connect-input"
            type="password"
            autoComplete="off"
            placeholder="sk-ant-…"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
          />
          <div className="connect-field-hint">Leave blank to use <code>$ANTHROPIC_API_KEY</code></div>
        </div>
      ) : (
        <div className="connect-field">
          <label className="connect-label">Address</label>
          <div className="connect-row">
            <input
              className="connect-input"
              type="text"
              spellCheck={false}
              placeholder={defaultUrl(backend, vllmBaseUrl, ollamaBaseUrl)}
              value={url}
              onChange={e => setUrl(e.target.value)}
            />
            <button
              className="connect-refresh"
              title="Reload the model list from this address"
              onClick={() => onFetchModels(backend, url)}
            >
              ⟳
            </button>
          </div>
          {modelsError && (
            <div className="connect-field-hint connect-field-error">
              No model list from this address: {modelsError}
            </div>
          )}
        </div>
      )}

      <div className="connect-field">
        <label className="connect-label">Model</label>
        <select
          className="connect-select"
          value={model}
          disabled={options.length === 0}
          onChange={e => setModel(e.target.value)}
        >
          {options.length === 0 ? (
            <option value="">{modelsLoading ? "Loading…" : "—"}</option>
          ) : (
            options.map(m => <option key={m} value={m}>{m}</option>)
          )}
        </select>
      </div>

      <button
        className="connect-btn connect-btn-primary"
        onClick={() => onConnect(model, backend, url, anthropic ? apiKey : undefined)}
      >
        Connect
      </button>
    </div>
  );
};
