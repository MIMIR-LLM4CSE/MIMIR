// What the extension host and the webview both start from when nothing is configured.
// The same values as the `mimir.*` settings declared in package.json.

export const DEFAULT_BACKEND = "vllm";
/** vLLM and Ray Serve: both an OpenAI-compatible endpoint. */
export const DEFAULT_OPENAI_URL = "http://127.0.0.1:8000";
export const DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434";
