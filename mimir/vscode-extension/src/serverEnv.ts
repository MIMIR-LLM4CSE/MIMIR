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

/**
 * The context-window override, for an endpoint that does not report max_model_len in
 * /v1/models (the client otherwise stays on its static 200K default). The two names are
 * the vLLM and Ray Serve escape hatches: only the active backend reads its own, and a
 * window the endpoint does publish wins over both.
 */
export function maxModelLenEnv(): Record<string, string> {
  const maxLen = vscode.workspace.getConfiguration("mimir").get<number>("maxModelLen", 0) ?? 0;
  return maxLen > 0
    ? { MIMIR_VLLM_MAX_MODEL_LEN: String(maxLen), MIMIR_RAY_MAX_MODEL_LEN: String(maxLen) }
    : {};
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
