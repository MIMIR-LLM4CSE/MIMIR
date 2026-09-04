import * as vscode from "vscode";
import * as cp from "child_process";
import * as fs from "fs";
import * as path from "path";
import WebSocket = require("ws");

let serverProcess: cp.ChildProcess | undefined;
// Tracks how the current serverProcess was started, so a connect request can
// reuse an already-running server only when it matches the requested mode and
// otherwise tear down a stale one (e.g. switching SLURM → vLLM-connect).

// ── vLLM tool-call parser auto-detection ──────────────────────────────────────
// Fallback map used if shared JSON profile loading fails. Keep these aligned
// with mimir/client/config/vllm_model_profiles.json — they are only a safety net
// for when that file can't be read, never the source of truth.
const _VLLM_TOOL_CALL_PARSERS_FALLBACK: Record<string, string> = {
  "mistral":      "mistral",
  "devstral":     "mistral",
  "mixtral":      "mistral",
  "nemotron":     "qwen3_coder",
  "qwen3.8":      "qwen3_xml",
  "qwen3-coder":  "qwen3_coder",
  "qwen3":        "hermes",
  "qwen2.5-coder": "hermes",
  "qwen2.5":      "hermes",
  "gemma":        "pythonic",
  "llama":        "llama3_json",
  "gpt":          "openai",
};

/**
 * Resolve the path to the shared vLLM model-profile JSON.
 *
 * The package lives at `<mimirPath>/mimir/...` (mimirPath is the import root put
 * on PYTHONPATH), so the profile is `<mimirPath>/mimir/client/config/...`. The
 * `mimir.mimirPath` setting is the source of truth for where the package is —
 * the open workspace folder is often a *different* project (e.g. a benchmark
 * repo) that doesn't contain the package, so deriving the path from it silently
 * misses the file and falls back to stale defaults. Prefer mimirPath, then try
 * the workspace folder (both flat and nested layouts) as a last resort.
 */
function _sharedProfilePath(): string | undefined {
  const rel = path.join("mimir", "client", "config", "vllm_model_profiles.json");
  const candidates: string[] = [];
  const mimirPath = vscode.workspace.getConfiguration("mimir").get<string>("mimirPath");
  if (mimirPath) {
    candidates.push(path.join(mimirPath, rel));
  }
  const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (root) {
    candidates.push(path.join(root, rel));
    // Workspace opened directly on the package repo (flat layout, no nested mimir/).
    candidates.push(path.join(root, "client", "config", "vllm_model_profiles.json"));
  }
  return candidates.find((p) => fs.existsSync(p));
}

/**
 * Append the host of *baseUrl* to the inherited no_proxy/NO_PROXY lists.
 *
 * On-prem vLLM hosts are reachable directly but an HTTP proxy will
 * black-hole them, which turns model resolution into an indefinite hang.
 */
function noProxyFor(baseUrl: string): Record<string, string> {
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

function _loadVllmToolCallParsers(): Record<string, string> {
  const sharedPath = _sharedProfilePath();
  if (sharedPath) {
    try {
      const raw = fs.readFileSync(sharedPath, "utf8");
      const parsed = JSON.parse(raw) as Record<string, { tool_call_parser?: string }>;
      const out: Record<string, string> = {};
      for (const [prefix, profile] of Object.entries(parsed)) {
        if (profile && typeof profile.tool_call_parser === "string" && profile.tool_call_parser.trim()) {
          out[prefix.toLowerCase()] = profile.tool_call_parser.trim();
        }
      }
      if (Object.keys(out).length > 0) {
        return out;
      }
    } catch {
      // Fall back to built-in defaults when profile file is absent/invalid.
    }
  }
  return { ..._VLLM_TOOL_CALL_PARSERS_FALLBACK };
}

function _modelMatchCandidates(model: string): string[] {
  const normalized = model.toLowerCase().replace(/\\/g, "/").trim();
  if (!normalized) return [];
  const out: string[] = [normalized];
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length > 0) {
    out.push(parts[parts.length - 1]);
    for (const p of parts) out.push(p);
  }
  return [...new Set(out)];
}

/**
 * Returns `--enable-auto-tool-choice --tool-call-parser <parser>` for the
 * given model, or an empty string if no profile is found.
 * Matching is longest-prefix, case-insensitive.
 */
function _vllmToolCallFlags(model: string): string {
  const parsers = _loadVllmToolCallParsers();
  const keys = Object.keys(parsers);
  const candidates = _modelMatchCandidates(model);
  const match = keys
    .filter((k) => candidates.some((c) => c.startsWith(k)))
    .sort((a, b) => b.length - a.length)[0];
  if (!match) { return ""; }
  return `--enable-auto-tool-choice --tool-call-parser ${parsers[match]}`;
}

/** Loads the `reasoning_parser` field per model prefix from the shared profile. */
function _loadVllmReasoningParsers(): Record<string, string> {
  const sharedPath = _sharedProfilePath();
  if (!sharedPath) { return {}; }
  try {
    const raw = fs.readFileSync(sharedPath, "utf8");
    const parsed = JSON.parse(raw) as Record<string, { reasoning_parser?: string }>;
    const out: Record<string, string> = {};
    for (const [prefix, profile] of Object.entries(parsed)) {
      if (profile && typeof profile.reasoning_parser === "string" && profile.reasoning_parser.trim()) {
        out[prefix.toLowerCase()] = profile.reasoning_parser.trim();
      }
    }
    return out;
  } catch {
    return {};
  }
}

/**
 * Returns `--reasoning-parser <parser>` for the given model when its profile
 * declares one, so vLLM splits thinking into a separate `reasoning_content`
 * field (which the backend renders as a thinking block). Empty string if none.
 * Matching is longest-prefix, case-insensitive.
 */
function _vllmReasoningFlags(model: string): string {
  const parsers = _loadVllmReasoningParsers();
  const keys = Object.keys(parsers);
  const candidates = _modelMatchCandidates(model);
  const match = keys
    .filter((k) => candidates.some((c) => c.startsWith(k)))
    .sort((a, b) => b.length - a.length)[0];
  if (!match) { return ""; }
  return `--reasoning-parser ${parsers[match]}`;
}

// ── Virtual document provider for proposed file content (diff editor) ─────────

/** Stores proposed file content keyed by virtual URI path, for open_diff. */
const _diffProposedContent = new Map<string, string>();

/**
 * Reconstruct the "before" content by reverse-applying a unified diff patch.
 * Removes added lines (+) and restores removed lines (-) to recover the original.
 */
function _reversePatch(current: string, patch: string): string {
  const currentLines = current.split("\n");
  const result: string[] = [];
  const patchLines = patch.split("\n");

  let ci = 0; // index into currentLines

  for (let pi = 0; pi < patchLines.length; pi++) {
    const line = patchLines[pi];
    if (line.startsWith("---") || line.startsWith("+++") || line.startsWith("diff ")) continue;
    if (line.startsWith("@@")) {
      // @@ -oldStart,oldCount +newStart,newCount @@
      const m = line.match(/\+(\d+)(?:,(\d+))?/);
      if (m) {
        const newStart = parseInt(m[1], 10) - 1; // 0-based
        // Emit unchanged lines up to this hunk from current
        while (ci < newStart) {
          result.push(currentLines[ci++]);
        }
      }
      continue;
    }
    if (line.startsWith("+")) {
      // Line was added in the patch → skip it in the current (it exists there)
      ci++;
    } else if (line.startsWith("-")) {
      // Line was removed in the patch → restore it to the original
      result.push(line.slice(1));
    } else if (line.startsWith(" ")) {
      // Context line — present in both; advance current pointer
      result.push(currentLines[ci++]);
    }
  }
  // Emit any remaining lines after the last hunk
  while (ci < currentLines.length) {
    result.push(currentLines[ci++]);
  }
  return result.join("\n");
}

const _diffContentProvider = new class implements vscode.TextDocumentContentProvider {
  provideTextDocumentContent(uri: vscode.Uri): string {
    return _diffProposedContent.get(uri.path) ?? "";
  }
}();

// ── Plan preview: one virtual document, always read from disk ────────────────
//
// The Markdown preview renders a *cached* TextDocument, and on a network mount
// the file watcher may never fire, so previewing the plan file itself re-renders
// the previous revision. We preview a `mimir-plan:` document instead: its content
// provider re-reads the bytes on every request and we invalidate it on each
// update. Trade-off: read-only, and relative links inside the plan do not
// resolve; plans are prose, freshness matters more.
//
// The URI is CONSTANT — it names "the plan MIMIR is showing", not one plan file.
// Keying it on the plan's path meant a second plan (new title → new file) became
// a new resource, and switching an open preview's resource leaves it rendering
// the old one; with a fixed resource every plan reuses the same preview tab.
//
// Invalidating that resource is necessary but NOT sufficient. Nothing holds an
// editor on the virtual document, so VS Code drops it once it is unreferenced;
// firing the change event on a dropped document is a no-op, and the preview keeps
// rendering the plan it last saw — which is why a plan written minutes earlier (in
// this session or another) stayed on screen for every plan after it. Hence the
// three steps in _showPlanPreview: reopen the document (fresh provider read),
// invalidate it (fresh content if it was still open), then force the preview to
// re-render (revealing an existing preview re-renders nothing by itself).

const _planChanged = new vscode.EventEmitter<vscode.Uri>();

/** Absolute path of the plan the preview currently mirrors. */
let _currentPlanPath: string | undefined;

/** The single document the plan preview renders. `.md` types it as Markdown. */
const PLAN_URI = vscode.Uri.from({ scheme: "mimir-plan", path: "/MIMIR plan.md" });

const _planContentProvider = new class implements vscode.TextDocumentContentProvider {
  readonly onDidChange = _planChanged.event;
  provideTextDocumentContent(_uri: vscode.Uri): string {
    if (!_currentPlanPath) return "No plan yet.";
    try {
      return fs.readFileSync(_currentPlanPath, "utf8");
    } catch (e) {
      return `Cannot read ${_currentPlanPath}\n\n${e}`;
    }
  }
}();

/** Poller for the currently previewed markdown file, if any. */
let _previewWatch: { path: string; mtimeMs: number; timer: NodeJS.Timeout } | undefined;

function _stopPreviewWatch(): void {
  if (_previewWatch) {
    clearInterval(_previewWatch.timer);
    _previewWatch = undefined;
  }
}

/** Poll `abs` and re-render the plan document whenever the file changes on disk. */
function _watchPreviewedFile(abs: string): void {
  const stamp = (): number => {
    try {
      return fs.statSync(abs).mtimeMs;
    } catch {
      return 0;
    }
  };
  if (_previewWatch?.path === abs) {
    // Same plan reopened: keep the poller, just re-baseline it.
    _previewWatch.mtimeMs = stamp();
    return;
  }
  _stopPreviewWatch();
  const timer = setInterval(() => {
    if (!_previewWatch) return;
    const mtimeMs = stamp();
    if (mtimeMs !== _previewWatch.mtimeMs) {
      _previewWatch.mtimeMs = mtimeMs;
      void _invalidatePlanDocument();
    }
  }, 1000);
  _previewWatch = { path: abs, mtimeMs: stamp(), timer };
}

/** Re-render every open Markdown preview from its (re-read) document. */
async function _refreshPlanPreview(): Promise<void> {
  try {
    await vscode.commands.executeCommand("markdown.preview.refresh");
  } catch {
    /* command absent on this VS Code build — the invalidation is all we have */
  }
}

/** Re-read the plan document and push the new bytes to the preview. */
async function _invalidatePlanDocument(): Promise<void> {
  // Reopening resurrects the document if VS Code dropped it (no editor holds it),
  // which re-runs the content provider on the current path; the fire covers the
  // opposite case, a document still open on the previous plan's bytes.
  try {
    await vscode.workspace.openTextDocument(PLAN_URI);
  } catch {
    /* provider threw — the fire below still reaches an open document */
  }
  _planChanged.fire(PLAN_URI);
  await _refreshPlanPreview();
}

/** Point the plan preview at `abs` and open/refresh it. */
function _showPlanPreview(abs: string): void {
  _currentPlanPath = abs;
  // Before revealing: revealing an already-open preview does not re-request the
  // content, so an earlier plan would still be on screen.
  void _invalidatePlanDocument().then(() =>
    vscode.commands.executeCommand("markdown.showPreview", PLAN_URI).then(
      () => {
        _watchPreviewedFile(abs);
        // Again once the preview is up: a preview created by this very call
        // renders before the first invalidation can reach it.
        setTimeout(() => void _invalidatePlanDocument(), 120);
      },
      () => vscode.window.showWarningMessage(`MIMIR: cannot preview ${abs}`)
    )
  );
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider("mimir-diff", _diffContentProvider),
    vscode.workspace.registerTextDocumentContentProvider("mimir-plan", _planContentProvider),
    _planChanged
  );

  const provider = new MimirAgentViewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("mimir.chatView", provider, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  // Active-editor context: tell the webview which file (and selected line range)
  // is focused, so the user can attach it to a message with one click (opt-in chip).
  // Selection changes are debounced to avoid a flood while dragging.
  let selectionTimer: ReturnType<typeof setTimeout> | undefined;
  const pushEditor = () => provider.pushActiveEditor();
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(pushEditor),
    vscode.window.onDidChangeTextEditorSelection(() => {
      if (selectionTimer) clearTimeout(selectionTimer);
      selectionTimer = setTimeout(pushEditor, 120);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("mimir.openChat", () =>
      vscode.commands.executeCommand("mimir.chatView.focus")
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("mimir.startServer", () =>
      startServer(context)
    )
  );
}

export function deactivate(): void {
  _stopPreviewWatch();
  serverProcess?.kill();
  serverProcess = undefined;
  // Also tear down any SSH tunnel; otherwise it survives the reload still
  // listening on localhost:8765 and intercepts the next session's connection.
  sshTunnelProcess?.kill();
  sshTunnelProcess = undefined;
}

// ── Sidebar WebviewViewProvider ───────────────────────────────────────────────

class MimirAgentViewProvider implements vscode.WebviewViewProvider {
  private _ws: WebSocket | undefined;
  private _view: vscode.WebviewView | undefined;
  private _pendingMessages: string[] = [];

  constructor(private readonly extensionUri: vscode.Uri) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this._view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this.extensionUri, "dist"),
        vscode.Uri.joinPath(this.extensionUri, "images"),
      ],
    };
    view.webview.html = getWebviewHtml(view.webview, this.extensionUri);

    // Messages from the React app → forward to Python WS server
    view.webview.onDidReceiveMessage((msg: unknown) => {
      this._handleFromWebview(msg);
    });

    // Do NOT auto-connect — user must click Connect in the UI.
    // This avoids spawning connections on HPC front nodes.

    // Send config immediately so the webview can populate the model list.
    this._sendConfig();
    // Seed the active-file chip on (re)load.
    this.pushActiveEditor();
  }

  /**
   * Report the active editor's file + selected line range to the webview as an
   * ``active_editor`` message. The webview shows a click-to-attach chip. When no file
   * editor is active we keep the last value (webview focus / non-file panels shouldn't
   * clear the chip), so nothing is posted in that case.
   */
  pushActiveEditor(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.uri.scheme !== "file") return;

    const file = vscode.workspace.asRelativePath(editor.document.uri, false);
    const sel = editor.selection;
    let selection: string | null = null;
    if (sel && !sel.isEmpty) {
      const start = sel.start.line + 1;
      // A selection ending at column 0 of a line doesn't really include that line
      // (whole-line drag selection), so treat it as ending on the previous line.
      const last =
        sel.end.character === 0 && sel.end.line > sel.start.line
          ? sel.end.line
          : sel.end.line + 1;
      selection = start === last ? `${start}` : `${start}-${last}`;
    }
    this._view?.webview.postMessage({ type: "active_editor", file, selection });
  }

  private _sendConfig(): void {
    const cfg = vscode.workspace.getConfiguration("mimir");

    // Expand relative vLLM model entries using VLLM_MODELS_DIR env var or the
    // mimir.vllmModelsDir setting.  Absolute paths (starting with /) are
    // kept as-is so existing full-path entries still work.
    const vllmModelsDir = (
      cfg.get<string>("vllmModelsDir") ?? ""
    ).replace(/\/+$/, "");
    const rawVllmModels = cfg.get<string[]>("vllmAvailableModels") ?? [];
    const vllmModels = rawVllmModels.map(m =>
      m.startsWith("/") || !vllmModelsDir ? m : `${vllmModelsDir}/${m}`
    );

    this._view?.webview.postMessage({
      type: "config",
      models: cfg.get<string[]>("availableModels") ?? [],
      vllmModels,
      anthropicModels: cfg.get<string[]>("anthropicAvailableModels") ?? [],
      modelSizes: cfg.get<Record<string, number>>("modelSizes") ?? {},
      slurmEnabled: cfg.get<boolean>("slurmEnabled") ?? false,
      clusterConfig: cfg.get<unknown[]>("clusterConfig") ?? [],
      backend: cfg.get<string>("backend") ?? "vllm",
      vllmBaseUrl: cfg.get<string>("vllmBaseUrl") ?? "http://127.0.0.1:8000",
      vllmMode: cfg.get<string>("vllmMode") ?? "launch",
    });
  }

  private _connectToServer(retryCount = 0): void {
    const cfg = vscode.workspace.getConfiguration("mimir");
    const wsUrl = cfg.get<string>("wsUrl") ?? "ws://localhost:8765";
    const maxRetries = 40; // retry for up to ~40 seconds while server starts

    const ws = new WebSocket(wsUrl);
    this._ws = ws;

    ws.on("open", () => {
      // Flush any messages queued before the connection was ready
      for (const m of this._pendingMessages) {
        ws.send(m);
      }
      this._pendingMessages = [];
    });

    ws.on("message", (data: WebSocket.RawData) => {
      // Forward Python server messages → React webview
      const text = data.toString();
      this._view?.webview.postMessage({ type: "ws", payload: text });
      // Surface a native VS Code notification when the chat isn't in view.
      this._maybeNotify(text);
    });

    ws.on("close", () => {
      const stillRunning =
        (serverProcess && !serverProcess.killed) ||
        (sshTunnelProcess && !sshTunnelProcess.killed);
      if (retryCount < maxRetries && stillRunning) {
        // Server or tunnel still alive — retry after 2 seconds
        setTimeout(() => this._connectToServer(retryCount + 1), 2000);
      } else {
        // Notify webview — no auto-reconnect (user must click Connect again)
        this._view?.webview.postMessage({ type: "ws_closed" });
      }
    });

    ws.on("error", () => {
      // close fires immediately after error (which handles reconnect/teardown);
      // surface a distinct error signal so the webview can show an error state.
      this._view?.webview.postMessage({ type: "ws_error" });
    });
  }

  // Timestamp of the last notification, used to throttle bursts (e.g. several
  // approvals arriving back-to-back should not spam a stack of toasts).
  private _lastNotifyAt = 0;

  /**
   * Show a native VS Code notification for milestone events (task finished or
   * user action required) — but only when the user isn't already looking at the
   * chat. "Not looking" means the VS Code window is unfocused OR the chat view
   * is hidden. Clicking the notification focuses the chat.
   */
  private _maybeNotify(payload: string): void {
    const cfg = vscode.workspace.getConfiguration("mimir");
    if (!(cfg.get<boolean>("notifications.enabled") ?? true)) {
      return;
    }

    let msg: {
      type?: string; text?: string; cancelled?: boolean; summary?: string;
      job_key?: string; state?: string;
    };
    try {
      msg = JSON.parse(payload);
    } catch {
      return;
    }
    if (!msg || typeof msg.type !== "string") {
      return;
    }

    // Only notify when the chat isn't in front of the user.
    const chatInView = (this._view?.visible ?? false) && vscode.window.state.focused;
    if (chatInView) {
      return;
    }

    let title: string | undefined;
    let kind: "info" | "warn" = "info";
    switch (msg.type) {
      case "answer":
        // Final answer to the user = task completed (ignore cancelled turns).
        if (msg.cancelled) { return; }
        title = "MIMIR a terminé la tâche.";
        kind = "info";
        break;
      case "approval":
        title = "MIMIR attend votre approbation.";
        kind = "warn";
        break;
      case "continue_prompt":
        title = "MIMIR attend votre confirmation pour continuer.";
        kind = "warn";
        break;
      case "todo_prompt":
        title = "MIMIR a besoin de votre intervention.";
        kind = "warn";
        break;
      case "job_complete":
        // A detached background run finished; the agent auto-resumes.
        if (msg.state === "crashed") {
          title = `Tâche en arrière-plan « ${msg.job_key ?? ""} » terminée en échec.`;
          kind = "warn";
        } else {
          title = `Tâche en arrière-plan « ${msg.job_key ?? ""} » terminée.`;
          kind = "info";
        }
        break;
      default:
        return;
    }

    // Coalesce rapid-fire notifications into one (2s window).
    const now = Date.now();
    if (now - this._lastNotifyAt < 2000) {
      return;
    }
    this._lastNotifyAt = now;

    const open = "Ouvrir le chat";
    const show = kind === "warn"
      ? vscode.window.showWarningMessage
      : vscode.window.showInformationMessage;
    show(title, open).then((choice) => {
      if (choice === open) {
        vscode.commands.executeCommand("mimir.chatView.focus");
      }
    });
  }

  // Kill every locally-tracked piece of connection state (the ws_server process
  // and any SSH tunnel) so a fresh connect always starts from a clean local
  // slate. The actual port 8765 is freed by `fuser -k` in each spawn command;
  // this just clears the processes this extension owns. Called at the top of both
  // connect paths so local↔SLURM switches never leak a server or tunnel that
  // keeps intercepting localhost:8765.
  private _teardownLocalState(): void {
    if (serverProcess && !serverProcess.killed) {
      serverProcess.kill();
    }
    serverProcess = undefined;
    if (sshTunnelProcess && !sshTunnelProcess.killed) {
      sshTunnelProcess.kill();
    }
    sshTunnelProcess = undefined;
  }

  private _startLocalAndConnect(model: string, backend = "vllm", vllmBaseUrl = "http://127.0.0.1:8000", anthropicApiKey = ""): void {
    // Always start from a clean local slate (kills any prior server/tunnel,
    // including one left over from a SLURM session) before binding port 8765.
    this._teardownLocalState();

    const cfg = vscode.workspace.getConfiguration("mimir");
    // This path runs ws_server on the local host (the frontal), which may be a
    // different CPU arch than the SLURM compute nodes. Prefer localPythonPath so
    // the frontal can use its own (e.g. x86) interpreter; fall back to pythonPath.
    const pythonPath = cfg.get<string>("localPythonPath") || cfg.get<string>("pythonPath") || "python3";
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

    const outputChannel = vscode.window.createOutputChannel("MIMIR Server");
    outputChannel.show();

    // In vLLM "connect" mode the server runs here on the frontal and points at an
    // already-running vLLM (no SLURM allocation, no `vllm serve`).
    // Free port 8765 first (mirroring the SLURM path) so a stale tunnel or a
    // previous server process cannot intercept connections meant for this one.
    const backendArgs =
      backend === "vllm" ? ` --backend vllm --vllm-base-url ${vllmBaseUrl}`
      : backend === "anthropic" ? " --backend anthropic"
      : "";
    const spawnCmd = `fuser -k 8765/tcp 2>/dev/null || true && ${pythonPath} -m mimir.client.ui.ws.ws_server --port 8765${model ? ` --model ${model}` : ""}${backendArgs}`;
    // Log the command only — the Claude API key is injected via env below and is
    // deliberately kept out of this string so it never lands in the output channel.
    outputChannel.appendLine(`[Local] Starting server: ${spawnCmd}`);

    // Internal HTTPS vLLM routes are often served behind a private
    // CA; when the user disables cert verification, propagate VLLM_VERIFY_SSL so
    // /v1/models model-resolution and chat requests don't hit CERTIFICATE_VERIFY_FAILED.
    const verifyEnv = cfg.get<boolean>("vllmVerifySsl", true) ? {} : { VLLM_VERIFY_SSL: "0" };
    // An HTTP proxy silently swallows requests to an on-prem vLLM host,
    // so ws_server would hang on /v1/models before ever binding port 8765.
    const noProxyEnv = backend === "vllm" ? noProxyFor(vllmBaseUrl) : {};
    // Only override ANTHROPIC_API_KEY when the webview actually supplied one;
    // otherwise inherit whatever is already exported (so users who set the key in
    // their shell don't have to retype it in the form).
    const anthropicEnv =
      backend === "anthropic" && anthropicApiKey ? { ANTHROPIC_API_KEY: anthropicApiKey } : {};
    serverProcess = cp.spawn("bash", ["-c", spawnCmd], {
      cwd,
      // Anchor the agent's per-workspace state dir (.mimir) and the file-server
      // root to the opened workspace, regardless of the process cwd.
      env: { ...process.env, MCP_FILES_ROOT: cwd, ...noProxyEnv, ...verifyEnv, ...anthropicEnv },
      stdio: ["ignore", "pipe", "pipe"],
    });

    serverProcess.stdout?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
    serverProcess.stderr?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
    serverProcess.on("exit", (code) => {
      outputChannel.appendLine(`\nServer exited (code ${code})`);
      serverProcess = undefined;
      this._view?.webview.postMessage({ type: "ws_closed" });
    });

    // Give the server a moment to start, then begin connecting with retries
    setTimeout(() => this._connectToServer(), 1000);
  }

  private _startSlurmAndConnect(model: string, profile?: { loginNode?: string; slurmArgs?: string }, backend = "vllm", vllmBaseUrl = "http://127.0.0.1:8000", clusterVllmPath?: string, clusterOllamaPath?: string, vllmMode: "launch" | "connect" = "launch"): void {
    // Same clean-slate guarantee as the local path: drop any prior server/tunnel
    // (e.g. a leftover local server still bound to 8765) before allocating.
    this._teardownLocalState();
    const cfg = vscode.workspace.getConfiguration("mimir");
    const pythonPath = cfg.get<string>("pythonPath") || "python3";
    const slurmArgs  = profile?.slurmArgs ?? cfg.get<string>("slurmArgs") ?? "";
    const loginNode  = profile?.loginNode ?? cfg.get<string>("loginNode") ?? "";
    const ollamaSetup = cfg.get<string>("ollamaSetupScript") ?? "";
    // Cluster-level path takes priority over global setting
    const ollamaPath = clusterOllamaPath ?? cfg.get<string>("ollamaPath") ?? "ollama";
    const ollamaUrl  = cfg.get<string>("ollamaUrl") ?? "http://127.0.0.1:11434";
    const mimirPath = cfg.get<string>("mimirPath") ?? "";

    if (!slurmArgs) {
      vscode.window.showErrorMessage("MIMIR: set mimir.slurmArgs before connecting via SLURM (e.g. '-p my-partition --gres=gpu:1 --account my-account --mem 64G').");
      return;
    }
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

    const outputChannel = vscode.window.createOutputChannel("MIMIR Server");
    outputChannel.show();
    outputChannel.appendLine(`[SLURM] Allocating node: salloc ${slurmArgs}`);

    // srun runs on the compute node: cd to the user's workspace, add mimirPath to PYTHONPATH
    // so the package is importable without changing the working directory.
    // Note: ollama is backgrounded with & so the next command must use ; not &&
    // OLLAMA_ORIGINS=* is required to avoid 403 "incorrect service" rejections.
    const ollamaSetupCmd = ollamaSetup ? `source ${ollamaSetup} && ` : "";
    // Env prefix applied DIRECTLY to the foreground ws_server python. The vLLM/
    // ollama daemon and its preceding `cd`/`export` run in a backgrounded AND-list
    // (`… &`), so they do NOT reach this process. MCP_FILES_ROOT must be set here
    // so the agent's per-workspace state dir (.mimir) anchors to the workspace
    // instead of falling back to the compute node's $HOME (the srun default cwd).
    const mimirArg   = `MCP_FILES_ROOT=${cwd} ${mimirPath ? `PYTHONPATH=${mimirPath}:$PYTHONPATH ` : ""}`;

    let remoteCmd: string;
    if (backend === "vllm") {
      const vllmSetupScript = cfg.get<string>("vllmSetupScript") ?? "";
      // Cluster-level path takes priority over global setting
      const vllmPath       = clusterVllmPath ?? cfg.get<string>("vllmPath") ?? "vllm";
      const vllmExtraArgs  = cfg.get<string>("vllmExtraArgs") ?? "";
      const vllmServeOverride = cfg.get<string>("vllmServeCommand") ?? "";
      // Auto-generate serve command from selected model when no override is set
      const vllmPort = (() => { try { return new URL(vllmBaseUrl).port || "8000"; } catch { return "8000"; } })();
      // Auto-inject tool-call + reasoning parser flags unless the user supplied an override command.
      const autoToolFlags = model ? _vllmToolCallFlags(model) : "";
      const autoReasoningFlags = model ? _vllmReasoningFlags(model) : "";
      const autoFlags = [autoToolFlags, autoReasoningFlags].filter(Boolean).join(" ");
      const vllmServeCommand = vllmServeOverride ||
        (model ? `${vllmPath} serve ${model} --port ${vllmPort}${autoFlags ? " " + autoFlags : ""}${vllmExtraArgs ? " " + vllmExtraArgs : ""}` : "");
      const vllmSetupCmd = vllmSetupScript ? `source ${vllmSetupScript} && ` : "";
      const wsBackendArgs = `--backend vllm --vllm-base-url ${vllmBaseUrl}`;
      // Skip TLS verification for internal HTTPS routes behind a private CA so
      // /v1/models resolution and chat requests don't hit CERTIFICATE_VERIFY_FAILED.
      const verifyPrefix = cfg.get<boolean>("vllmVerifySsl", true) ? "" : "VLLM_VERIFY_SSL=0 ";
      // "launch" mode requires a serve command; never silently fall through to
      // pre-running (that path is handled by _startLocalAndConnect in connect mode).
      if (vllmMode === "launch" && !vllmServeCommand) {
        vscode.window.showErrorMessage("MIMIR: vLLM launch mode needs a model (or set mimir.vllmServeCommand). To attach to a running server, choose 'Connect to running server'.");
        return;
      }
      if (vllmServeCommand) {
        // Self-hosted: start vLLM daemon then wait for readiness before launching ws_server
        remoteCmd =
          `cd ${cwd} && ` +
          vllmSetupCmd +
          `fuser -k 8765/tcp 2>/dev/null || true && ` +
          `${vllmServeCommand} > vllm.log 2>&1 & ` +
          `timeout 1800 bash -c "until http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= curl -sf ${vllmBaseUrl}/v1/models >/dev/null 2>&1; do sleep 2; done"; ` +
          `${mimirArg}${verifyPrefix}${pythonPath} -m mimir.client.ui.ws.ws_server --host 0.0.0.0 --port 8765 --cwd ${cwd}${model ? ` --model ${model}` : ""} ${wsBackendArgs}`;
      } else {
        // Pre-running vLLM: skip daemon, just pass backend args to ws_server
        remoteCmd =
          `cd ${cwd} && ` +
          `fuser -k 8765/tcp 2>/dev/null || true && ` +
          `${mimirArg}${verifyPrefix}${pythonPath} -m mimir.client.ui.ws.ws_server --host 0.0.0.0 --port 8765 --cwd ${cwd}${model ? ` --model ${model}` : ""} ${wsBackendArgs}`;
      }
    } else {
      // Ollama (default)
      remoteCmd =
        `cd ${cwd} && ` +
        ollamaSetupCmd +
        `fuser -k 8765/tcp 2>/dev/null || true && ` +
        `OLLAMA_ORIGINS="*" ${ollamaPath} serve > ollama.log 2>&1 & ` +
        `timeout 300 bash -c "until curl -sf ${ollamaUrl}/api/tags >/dev/null 2>&1; do sleep 1; done"; ` +
        `no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost OLLAMA_BASE_URL=${ollamaUrl} ${mimirArg}${pythonPath} -m mimir.client.ui.ws.ws_server --host 0.0.0.0 --port 8765 --cwd ${cwd}${model ? ` --model ${model}` : ""}`;
    }

    // When tunnelling via a login node the remoteCmd is embedded inside a
    // double-quoted SSH argument.  Any literal " in remoteCmd (e.g. the
    // inner `bash -c "until … do … done"`) would break that outer double-
    // quoted string, so we escape them as \" before interpolation.
    const slurmProc = cp.spawn(
      "bash", ["-c", loginNode
        ? `ssh -tt ${loginNode} "salloc ${slurmArgs} srun --ntasks=1 bash -c '${remoteCmd.replace(/"/g, '\\"')}'"`
        : `salloc ${slurmArgs} srun --ntasks=1 bash -c '${remoteCmd}'`],
      { cwd, stdio: ["ignore", "pipe", "pipe"] }
    );

    serverProcess = slurmProc;
    let tunnelCreated = false;  // guard: only create the tunnel once

    slurmProc.stdout?.on("data", (d: Buffer) => {
      const text = d.toString();
      outputChannel.append(text);
      // Parse compute node hostname emitted by ws_server.py (only act on first match)
      const match = !tunnelCreated && text.match(/SLURM_NODE:(\S+)/);
      if (match) {
        tunnelCreated = true;
        const node = match[1];
        outputChannel.appendLine(`[SLURM] Compute node: ${node} — creating SSH tunnel...`);

        // Free local port 8765 first, then spawn tunnel once fuser completes
        cp.exec("fuser -k 8765/tcp 2>/dev/null; sleep 1", () => {
          const commonOpts = [
            "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3",
            "-o", "IPQoS=lowdelay throughput",
            "-o", "TCPKeepAlive=yes",
            "-o", "Compression=no",
            "-L", `8765:localhost:8765`,
          ];
          const tunnelArgs = loginNode
            ? [...commonOpts, "-J", loginNode, node]
            : [...commonOpts, node];
          outputChannel.appendLine(`[SLURM] ssh ${tunnelArgs.join(" ")}`);
          sshTunnelProcess = cp.spawn("ssh", tunnelArgs, { stdio: ["ignore", "pipe", "pipe"] });
          sshTunnelProcess.stdout?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
          sshTunnelProcess.stderr?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
          sshTunnelProcess.on("exit", (code) => {
            outputChannel.appendLine(`[SLURM] SSH tunnel exited (code ${code})`);
          });
          // Give the tunnel a moment to establish, then connect
          setTimeout(() => this._connectToServer(), 3000);
        });
      }
    });

    slurmProc.stderr?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
    slurmProc.on("exit", (code) => {
      outputChannel.appendLine(`\n[SLURM] Process exited (code ${code})`);
      sshTunnelProcess?.kill();
      sshTunnelProcess = undefined;
      serverProcess = undefined;
      this._view?.webview.postMessage({ type: "ws_closed" });
    });
  }

  private _handleFromWebview(msg: unknown): void {
    const m = msg as Record<string, unknown>;

    if (m.type === "open_file") {
      const rel = m.file as string | undefined;
      if (rel) {
        const roots = vscode.workspace.workspaceFolders;
        const base  = roots?.[0]?.uri.fsPath ?? process.cwd();
        const abs   = require("path").isAbsolute(rel)
          ? rel
          : require("path").join(base, rel);
        const uri = vscode.Uri.file(abs);
        vscode.workspace.openTextDocument(uri).then(
          (doc) => vscode.window.showTextDocument(doc, { preview: true }),
          () => vscode.window.showWarningMessage(`MIMIR: cannot open ${rel}`)
        );
      }
      return;
    }

    if (m.type === "open_preview") {
      // Open a file the agent wrote (e.g. a plan .md) for reading. Markdown
      // files render in VS Code's Markdown preview; anything else falls back to
      // a normal editor tab.
      const rel = m.file as string | undefined;
      if (rel) {
        const roots = vscode.workspace.workspaceFolders;
        const base  = roots?.[0]?.uri.fsPath ?? process.cwd();
        const abs   = require("path").isAbsolute(rel)
          ? rel
          : require("path").join(base, rel);
        if (/\.mdx?$/i.test(abs)) {
          _showPlanPreview(abs);
        } else {
          const uri = vscode.Uri.file(abs);
          vscode.workspace.openTextDocument(uri).then(
            (doc) => vscode.window.showTextDocument(doc, { preview: true }),
            () => vscode.window.showWarningMessage(`MIMIR: cannot open ${rel}`)
          );
        }
      }
      return;
    }

    if (m.type === "open_diff") {
      const rel        = m.file as string | undefined;
      const newContent = m.new_content as string | undefined;
      if (rel) {
        const roots = vscode.workspace.workspaceFolders;
        const base  = roots?.[0]?.uri.fsPath ?? process.cwd();
        const abs   = require("path").isAbsolute(rel)
          ? rel
          : require("path").join(base, rel);
        const basename = require("path").basename(rel);
        const currentUri = vscode.Uri.file(abs);

        if (newContent !== undefined) {
          // Store proposed content under the absolute path as key, then open a
          // diff editor: left = current file on disk (or empty if new), right = proposed content.
          _diffProposedContent.set(abs, newContent);
          const proposedUri = vscode.Uri.from({ scheme: "mimir-diff", path: abs });

          // Check if the file already exists; if not, use an empty virtual doc
          // as the left side so VS Code doesn't throw "nonexistent file".
          const fs = require("fs") as typeof import("fs");
          const fileExists = fs.existsSync(abs);
          if (fileExists) {
            vscode.commands.executeCommand(
              "vscode.diff",
              currentUri,
              proposedUri,
              `${basename}: Current ↔ Proposed`,
              { preview: true }
            );
          } else {
            // New file — open the proposed content directly (no empty-vs-new diff).
            vscode.workspace.openTextDocument(proposedUri).then(
              (doc) => vscode.window.showTextDocument(doc, { preview: true }),
              () => vscode.window.showWarningMessage(`MIMIR: cannot preview new file ${rel}`)
            );
          }
        } else if (m.patch) {
          // Post-write diff: reconstruct the "before" content by reverse-applying
          // the unified patch, then open left=before right=current.
          const patch = m.patch as string;
          const fs = require("fs") as typeof import("fs");
          let currentContent = "";
          try { currentContent = fs.readFileSync(abs, "utf8"); } catch { /* new file */ }
          const originalContent = _reversePatch(currentContent, patch);
          _diffProposedContent.set(abs + "__original", originalContent);
          const originalUri = vscode.Uri.from({ scheme: "mimir-diff", path: abs + "__original" });
          vscode.commands.executeCommand(
            "vscode.diff",
            originalUri,
            currentUri,
            `${basename}: Before ↔ After`,
            { preview: true }
          );
        } else {
          // No patch or proposed content — fall back to opening the file directly.
          vscode.workspace.openTextDocument(currentUri).then(
            (doc) => vscode.window.showTextDocument(doc, { preview: true }),
            () => vscode.window.showWarningMessage(`MIMIR: cannot open ${rel}`)
          );
        }
      }
      return;
    }

    if (m.type === "open_patch") {
      // Open a single syntax-highlighted .diff virtual document (review window style).
      const rel  = m.file as string | undefined;
      const patch = m.patch as string | undefined;
      if (rel && patch) {
        const roots = vscode.workspace.workspaceFolders;
        const base  = roots?.[0]?.uri.fsPath ?? process.cwd();
        const abs   = require("path").isAbsolute(rel) ? rel : require("path").join(base, rel);
        const basename = require("path").basename(rel);
        // Store patch under a .diff-suffixed key so the language server detects
        // the diff language and applies green/red syntax highlighting.
        const patchKey = abs + ".diff";
        _diffProposedContent.set(patchKey, patch);
        const patchUri = vscode.Uri.from({ scheme: "mimir-diff", path: patchKey });
        vscode.workspace.openTextDocument(patchUri).then(
          (doc) => vscode.window.showTextDocument(doc, { preview: true }),
          () => vscode.window.showWarningMessage(`MIMIR: cannot open patch for ${rel}`)
        );
      }
      return;
    }

    if (m.type === "disconnect") {
      this._ws?.close();
      this._ws = undefined;
      if (serverProcess && !serverProcess.killed) {
        serverProcess.kill();
        serverProcess = undefined;
      }
      sshTunnelProcess?.kill();
      sshTunnelProcess = undefined;
      this._view?.webview.postMessage({ type: "ws_closed" });
      return;
    }

    if (m.type === "get_config") {
      this._sendConfig();
      return;
    }

    if (m.type === "connect") {
      const model = (m.model as string | undefined) ?? "";
      const profile = m.profile as { loginNode?: string; slurmArgs?: string } | undefined;
      const backend = (m.backend as string | undefined) ?? "vllm";
      const vllmBaseUrl = (m.vllmBaseUrl as string | undefined) ?? "http://127.0.0.1:8000";
      const vllmMode = (m.vllmMode as "launch" | "connect" | undefined) ?? "launch";
      const clusterVllmPath = (m.vllmPath as string | undefined);
      const clusterOllamaPath = (m.ollamaPath as string | undefined);
      // Claude API key from the webview. Kept in-process only: passed to the
      // ws_server via the ANTHROPIC_API_KEY env var (never a CLI arg or setting),
      // so it never reaches the process list, the output channel, or disk.
      const anthropicApiKey = (m.anthropicApiKey as string | undefined) ?? "";
      const cfg = vscode.workspace.getConfiguration("mimir");
      const vllmConnect = backend === "vllm" && vllmMode === "connect";
      // The hosted Claude API runs no local daemon and needs no GPU/SLURM — it
      // runs the ws_server on the frontal and talks to api.anthropic.com, exactly
      // like vLLM connect mode.
      const anthropic = backend === "anthropic";

      // In vLLM connect mode the model is whatever the endpoint serves, so an
      // empty model is fine (ws_server resolves it from /v1/models). Every other
      // backend — including Anthropic — needs an explicit model.
      if (!model && !vllmConnect) {
        vscode.window.showErrorMessage("MIMIR: select a model before connecting.");
        return;
      }

      if (anthropic) {
        this._startLocalAndConnect(model, "anthropic", vllmBaseUrl, anthropicApiKey);
      } else if (vllmConnect) {
        // Connect to an already-running vLLM at vllmBaseUrl. No SLURM allocation
        // and no `vllm serve` — the ws_server runs locally on the frontal and
        // talks to the remote endpoint (regardless of slurmEnabled).
        this._startLocalAndConnect(model, "vllm", vllmBaseUrl);
      } else if (cfg.get<boolean>("slurmEnabled")) {
        this._startSlurmAndConnect(model, profile, backend, vllmBaseUrl, clusterVllmPath, clusterOllamaPath, vllmMode);
      } else {
        // Non-slurm: just connect to the configured wsUrl.
        // The user is responsible for starting the ws_server manually
        // (e.g. on a remote ARM login node where spawning locally is not possible).
        this._connectToServer();
      }
      return;
    }

    if (m.type === "ws_send") {
      // React wants to send a message to the Python server
      const payload = m.payload as string;
      if (this._ws?.readyState === WebSocket.OPEN) {
        this._ws.send(payload);
      } else {
        this._pendingMessages.push(payload);
      }
    }
  }
}

// ── Optional server lifecycle ─────────────────────────────────────────────────

let sshTunnelProcess: cp.ChildProcess | undefined;

function startServer(context: vscode.ExtensionContext): void {
  const cfg = vscode.workspace.getConfiguration("mimir");
  const pythonPath = cfg.get<string>("pythonPath") ?? "python3";
  const slurmEnabled = cfg.get<boolean>("slurmEnabled") ?? false;
  const slurmArgs = cfg.get<string>("slurmArgs") ?? "";

  if (serverProcess && !serverProcess.killed) {
    vscode.window.showInformationMessage("MIMIR WS server is already running.");
    return;
  }

  const outputChannel = vscode.window.createOutputChannel("MIMIR Server");
  outputChannel.show();

  const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

  let spawnCmd: string;
  if (slurmEnabled) {
    spawnCmd = `salloc ${slurmArgs} srun ${pythonPath} -m mimir.client.ui.ws.ws_server --host 0.0.0.0 --port 8765`;
    outputChannel.appendLine(`[SLURM] Allocating node: salloc ${slurmArgs}`);
  } else {
    // Free port 8765 first so a stale tunnel/server can't intercept connections.
    spawnCmd = `fuser -k 8765/tcp 2>/dev/null || true && ${pythonPath} -m mimir.client.ui.ws.ws_server --port 8765`;
  }

  serverProcess = cp.spawn("bash", ["-c", spawnCmd], {
    cwd,
    // Anchor the agent's per-workspace state dir (.mimir) and the file-server
    // root to the opened workspace, regardless of the process cwd.
    env: { ...process.env, MCP_FILES_ROOT: cwd },
    stdio: ["ignore", "pipe", "pipe"],
  });

  serverProcess.stdout?.on("data", (d: Buffer) => {
    const text = d.toString();
    outputChannel.append(text);

    // When SLURM mode is on, parse the compute node hostname and set up SSH tunnel
    if (slurmEnabled) {
      const match = text.match(/SLURM_NODE:(\S+)/);
      if (match) {
        const node = match[1];
        outputChannel.appendLine(`[SLURM] Compute node: ${node} — creating SSH tunnel...`);
        sshTunnelProcess?.kill();
        sshTunnelProcess = cp.spawn("ssh", ["-N", "-L", `8765:localhost:8765`, node], {
          stdio: "ignore",
        });
        sshTunnelProcess.on("exit", (code) => {
          outputChannel.appendLine(`[SLURM] SSH tunnel exited (code ${code})`);
        });
        outputChannel.appendLine(`[SLURM] SSH tunnel established → localhost:8765 → ${node}:8765`);
      }
    }
  });

  serverProcess.stderr?.on("data", (d: Buffer) =>
    outputChannel.append(d.toString())
  );
  serverProcess.on("exit", (code) => {
    outputChannel.appendLine(`\nServer exited (code ${code})`);
    sshTunnelProcess?.kill();
    sshTunnelProcess = undefined;
    serverProcess = undefined;
  });

  vscode.window.showInformationMessage(
    slurmEnabled ? "MIMIR WS server starting via SLURM…" : "MIMIR WS server started."
  );
  context.subscriptions.push({
    dispose: () => {
      serverProcess?.kill();
      sshTunnelProcess?.kill();
    },
  });
}

// ── Webview HTML ──────────────────────────────────────────────────────────────

function getWebviewHtml(
  webview: vscode.Webview,
  extensionUri: vscode.Uri
): string {
  const scriptUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "dist", "webview.js")
  );
  // Brand assets served from the extension's images/ folder. Exposed to the
  // React app via a small global so components (avatars, connect screen) can
  // reference them through proper webview URIs under the CSP.
  const logoUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "images", "mimir-logo.png")
  );
  const introVideoUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "images", "mimir-intro.mp4")
  );
  const nonce = getNonce();

  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             script-src 'nonce-${nonce}';
             style-src ${webview.cspSource} 'unsafe-inline';
             font-src data:;
             connect-src ws://localhost:* ws://127.0.0.1:*;
             media-src ${webview.cspSource};
             img-src ${webview.cspSource} data:;">
  <title>MIMIR</title>
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}">
    window.__MIMIR_ASSETS__ = {
      logo: "${logoUri}",
      introVideo: "${introVideoUri}"
    };
  </script>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
}

function getNonce(): string {
  let text = "";
  const possible =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
