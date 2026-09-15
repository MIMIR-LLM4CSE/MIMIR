import React, { useState, useEffect, useRef } from "react";
import type { RememberedEndpoint } from "../types";

interface Props {
  /** Backend selected in settings — the value the form starts on. */
  backend?: string;
  /** Default addresses from settings; the user can edit them here. */
  vllmBaseUrl?: string;
  rayBaseUrl?: string;
  ollamaBaseUrl?: string;
  /** Claude model ids (static — the hosted API is not queried without a key). */
  anthropicModels?: string[];
  /** Models the endpoint reports it serves, fetched by the extension host. */
  models?: string[];
  modelsError?: string | null;
  /** Whether the endpoint has answered the model-list question yet. */
  modelsProbed?: boolean;
  /** Endpoint the host remembers — seeds the address, model and the checkbox. */
  remembered?: RememberedEndpoint | null;
  onFetchModels: (backend: string, baseUrl: string) => void;
  onConnect: (
    model: string,
    backend: string,
    baseUrl: string,
    anthropicApiKey?: string,
    remember?: boolean,
  ) => void;
}

/** Address the form starts on for a given backend. */
function defaultUrl(backend: string, vllm: string, ray: string, ollama: string): string {
  if (backend === "ollama") return ollama;
  if (backend === "ray") return ray;
  return vllm;
}

export const ConnectForm: React.FC<Props> = ({
  backend: initBackend = "vllm",
  vllmBaseUrl = "http://127.0.0.1:8000",
  rayBaseUrl = "http://127.0.0.1:8000",
  ollamaBaseUrl = "http://127.0.0.1:11434",
  anthropicModels = [],
  models = [],
  modelsError = null,
  modelsProbed = false,
  remembered = null,
  onFetchModels,
  onConnect,
}) => {
  const [backend, setBackend] = useState(initBackend);
  const [url, setUrl] = useState(defaultUrl(initBackend, vllmBaseUrl, rayBaseUrl, ollamaBaseUrl));
  // Claude API key — kept in webview state only; forwarded on connect and never
  // persisted. Left blank means "use whatever ANTHROPIC_API_KEY the host exports".
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  // Ticked means: store this address so the next VS Code window reconnects to it
  // by itself. Starts ticked when we are already showing a remembered endpoint,
  // so unticking and connecting is how the user forgets it.
  const [remember, setRemember] = useState(remembered !== null);

  const anthropic = backend === "anthropic";
  const options = anthropic ? anthropicModels : models;

  const handleBackendChange = (next: string) => {
    setBackend(next);
    setModel("");
    if (next !== "anthropic") setUrl(defaultUrl(next, vllmBaseUrl, rayBaseUrl, ollamaBaseUrl));
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

  // The model is picked from what the endpoint serves. The remembered model wins
  // over the first entry, so a reconnect lands on the same model as last time —
  // and the control below shows that choice rather than making it silently.
  useEffect(() => {
    if (options.length > 0 && !options.includes(model)) {
      const preferred =
        remembered && remembered.backend === backend && options.includes(remembered.model)
          ? remembered.model
          : options[0];
      setModel(preferred);
    }
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
          <option value="ray">Ray Serve</option>
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
              placeholder={defaultUrl(backend, vllmBaseUrl, rayBaseUrl, ollamaBaseUrl)}
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
              {/* A refused certificate is the one failure with a one-click remedy,
                  and the one the server does not share: it verifies nothing by
                  default, so the list can work there and fail here. */}
              {/certificate|self.signed|SSL|TLS/i.test(modelsError) && (
                <> — untick <code>mimir.vllmVerifySsl</code> if this is an internal
                route behind a private CA.</>
              )}
            </div>
          )}
        </div>
      )}

      {options.length > 1 ? (
        <div className="connect-field">
          <label className="connect-label">Model</label>
          <select
            className="connect-select"
            value={model}
            onChange={e => setModel(e.target.value)}
          >
            {options.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>
      ) : options.length === 1 ? (
        // One served model is not a choice; naming it still tells the user what
        // they are about to connect to.
        <div className="connect-field">
          <label className="connect-label">Model</label>
          <div className="connect-field-hint">{options[0]}</div>
        </div>
      ) : !anthropic && !modelsError ? (
        // Nothing to pick from. "Asking", "answered with nothing" and "answered
        // with a list" must not all look like an empty space: the user is left
        // reading silence and concluding the feature is missing. An endpoint that
        // answers but names no model is the server's own distinction, kept here.
        <div className="connect-field">
          <label className="connect-label">Model</label>
          <div className="connect-field-hint">
            {modelsProbed
              ? "This address answered but named no model — connecting will let the server pick."
              : "Asking this address what it serves…"}
          </div>
        </div>
      ) : null}

      {!anthropic && (
        <label className="connect-remember">
          <input
            type="checkbox"
            checked={remember}
            onChange={e => setRemember(e.target.checked)}
          />
          <span>Remember this address and reconnect on startup</span>
        </label>
      )}

      <button
        className="connect-btn connect-btn-primary"
        onClick={() =>
          onConnect(model, backend, url, anthropic ? apiKey : undefined, !anthropic && remember)
        }
      >
        Connect
      </button>
    </div>
  );
};
