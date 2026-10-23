/** The interpreter the MIMIR server runs on, and the environment it is spawned with. */
import * as vscode from "vscode";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { sharedReleasePython } from "./selfUpdate";

/**
 * Interpreter that runs the WS server: `bash -c` sources no profile, so `python3` from
 * PATH is rarely the right one. In order: `mimir.pythonPath`, `MIMIR_PYTHON`,
 * `<releaseHome>/current/PYTHON_COMMAND` (a shared site release), `<state home>/python`
 * (written by install.sh), then `python3` from PATH.
 */
export function resolvePython(extensionRoot: string): string {
  const configured = vscode.workspace.getConfiguration("mimir").get<string>("pythonPath");
  if (configured) {
    return configured;
  }
  if (process.env.MIMIR_PYTHON) {
    return process.env.MIMIR_PYTHON;
  }
  const shared = sharedReleasePython(extensionRoot);
  if (shared) {
    return shared;
  }
  const stateHome = process.env.MIMIR_STATE_HOME || path.join(os.homedir(), ".mimir");
  try {
    const interpreter = fs.readFileSync(path.join(stateHome, "python"), "utf8").trim();
    if (interpreter && fs.existsSync(interpreter)) {
      return interpreter;
    }
  } catch {
    // Not installed via install.sh, or the file was removed — fall through.
  }
  return "python3";
}

/**
 * Environment the MIMIR server process is spawned with: the extension host's, without
 * `PYTHONHOME` and `PYTHONPATH`, which would run the venv's python on the user's
 * packages. Only for the processes MIMIR starts; the user's shell is untouched.
 */
export function mimirServerEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.PYTHONHOME;
  delete env.PYTHONPATH;
  return env;
}

type Env = Record<string, string>;

/** A setting the server reads as environment: its key under `mimir.`, and what it sets. */
export interface EnvSetting {
  key: string;
  env: (value: unknown) => Env;
}

/** A count or duration: set when positive, 0 keeps the server's default. */
const positive = (...names: string[]) => (value: unknown): Env =>
  typeof value === "number" && value > 0
    ? Object.fromEntries(names.map((n) => [n, String(value)]))
    : {};

/** Free text or a path: set when not blank. */
const text = (name: string) => (value: unknown): Env =>
  typeof value === "string" && value.trim() ? { [name]: value.trim() } : {};

/** A switch on by default: set to *offValue* only when turned off. */
const offAs = (name: string, offValue: string) => (value: unknown): Env =>
  value === false ? { [name]: offValue } : {};

/** A list of paths, joined the way the server splits it. */
const pathList = (name: string) => (value: unknown): Env => {
  const paths = Array.isArray(value) ? value.filter((p) => typeof p === "string" && p.trim()) : [];
  return paths.length ? { [name]: paths.join(path.delimiter) } : {};
};

/**
 * Every setting the server takes through its environment. The context-window override
 * sets both the vLLM and the Ray Serve name: only the active backend reads its own, and a
 * window the endpoint publishes wins over both.
 */
export const SERVER_ENV_SETTINGS: readonly EnvSetting[] = [
  { key: "vllmVerifySsl", env: offAs("VLLM_VERIFY_SSL", "0") },
  { key: "maxModelLen", env: positive("MIMIR_VLLM_MAX_MODEL_LEN", "MIMIR_RAY_MAX_MODEL_LEN") },
  { key: "vllmTokenize", env: offAs("MIMIR_VLLM_TOKENIZE", "0") },
  { key: "ollama.numCtx", env: positive("MIMIR_OLLAMA_NUM_CTX") },
  { key: "anthropic.effort", env: text("MIMIR_ANTHROPIC_EFFORT") },
  { key: "anthropic.thinking", env: offAs("MIMIR_ANTHROPIC_THINKING", "off") },
  { key: "anthropic.maxTokens", env: positive("MIMIR_ANTHROPIC_MAX_TOKENS") },
  { key: "anthropic.maxInputTokens", env: positive("MIMIR_ANTHROPIC_MAX_INPUT_TOKENS") },
  { key: "embeddings.model", env: text("MIMIR_EMBED_MODEL") },
  { key: "embeddings.baseUrl", env: text("MIMIR_EMBED_BASE_URL") },
  { key: "embeddings.timeout", env: positive("MIMIR_EMBED_TIMEOUT") },
  { key: "sessions.maxLive", env: positive("MIMIR_MAX_LIVE_SESSIONS") },
  { key: "sessions.idleTimeout", env: positive("MIMIR_SESSION_IDLE_TTL") },
  { key: "sessions.initTimeout", env: positive("MIMIR_INIT_TIMEOUT") },
  { key: "sessions.backendWaitTimeout", env: positive("MIMIR_BACKEND_TIMEOUT") },
  { key: "sessions.backendPollInterval", env: positive("MIMIR_BACKEND_POLL_INTERVAL") },
  { key: "subagents.maxConcurrent", env: positive("MIMIR_SUBAGENT_MAX_CONCURRENT") },
  { key: "subagents.copiesDir", env: text("MIMIR_SUBAGENT_COPIES_DIR") },
  { key: "server.idleTimeout", env: positive("MIMIR_SERVER_IDLE_TTL") },
  { key: "server.detachGrace", env: positive("MIMIR_DETACH_GRACE") },
  { key: "customization.systemPromptFile", env: text("MIMIR_SYSTEM_PROMPT_FILE") },
  { key: "customization.skillsDir", env: text("MIMIR_SKILLS_DIR") },
  { key: "customization.pluginsDir", env: text("MIMIR_PLUGINS_DIR") },
  { key: "customization.serversDir", env: text("MIMIR_SERVERS_DIR") },
  { key: "workspace.extraReadRoots", env: pathList("MIMIR_EXTRA_FILE_ROOTS") },
  { key: "workspace.scratchDir", env: text("MIMIR_SCRATCH_DIR") },
  { key: "hpc.moduleIndexBudget", env: positive("MIMIR_MODULE_INDEX_BUDGET") },
  { key: "hpc.slurmStaleAfter", env: positive("MIMIR_SLURM_STALE_AFTER") },
];

/** Free-form variables for anything without a setting of its own. */
export const EXTRA_ENV_SETTING = "server.env";

/**
 * The environment the settings give the server. Only a value the user set counts, so a
 * variable exported in the shell still reaches the server unless a setting overrides it;
 * `mimir.server.env` goes last and wins.
 */
export function settingsEnv(): Env {
  const cfg = vscode.workspace.getConfiguration("mimir");
  const env: Env = {};
  for (const setting of SERVER_ENV_SETTINGS) {
    const seen = cfg.inspect(setting.key);
    const value = seen?.workspaceFolderValue ?? seen?.workspaceValue ?? seen?.globalValue;
    if (value !== undefined) Object.assign(env, setting.env(value));
  }
  const extra = cfg.get<Record<string, unknown>>(EXTRA_ENV_SETTING) ?? {};
  for (const [name, value] of Object.entries(extra)) {
    if (name && value !== null && value !== undefined) env[name] = String(value);
  }
  return env;
}

/** Whether *event* changed a setting the server only reads when it starts. */
export function affectsServerEnv(event: vscode.ConfigurationChangeEvent): boolean {
  return [...SERVER_ENV_SETTINGS.map((s) => s.key), EXTRA_ENV_SETTING]
    .some((key) => event.affectsConfiguration(`mimir.${key}`));
}

/**
 * Append the host of *baseUrl* to the inherited no_proxy/NO_PROXY lists.
 *
 * On-prem vLLM hosts are reachable directly but an HTTP proxy will
 * black-hole them, which turns model resolution into an indefinite hang.
 */
export function noProxyFor(baseUrl: string): Record<string, string> {
  let host: string;
  try {
    host = new URL(baseUrl).hostname;
  } catch {
    return {};
  }
  if (!host) {
    return {};
  }
  const merge = (v: string | undefined) => (v ? `${v},${host}` : host);
  return { no_proxy: merge(process.env.no_proxy), NO_PROXY: merge(process.env.NO_PROXY) };
}
