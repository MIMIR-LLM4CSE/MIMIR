import * as vscode from "vscode";
import * as cp from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import WebSocket = require("ws");
import { fetchModels, type DiscoverableBackend } from "./modelList";

let serverProcess: cp.ChildProcess | undefined;

/** Global-state key holding the endpoint the user asked us to remember. */
const REMEMBERED_KEY = "mimir.rememberedEndpoint";

/** Endpoint we can reconnect to unattended — no secret is ever part of it. */
interface RememberedEndpoint {
  backend: string;
  baseUrl: string;
  model: string;
}

/**
 * Interpreter that runs the WS server.
 *
 * `bash -c` is neither a login nor an interactive shell, so it sources no profile:
 * the child gets this extension host's environment and nothing else. `python3` from
 * that PATH is therefore whatever the machine's default is — frequently older than
 * MIMIR's 3.10 minimum, and almost never the venv the package was installed into.
 *
 * Rather than making every user write a setting, resolve it in order:
 *   1. `mimir.pythonPath` — an explicit override still wins.
 *   2. `MIMIR_PYTHON` — for people who would rather export it than click.
 *   3. `<state home>/python` — written by install.sh, so a plain `./install.sh`
 *      needs no configuration at all.
 *   4. `python3` from PATH.
 */
function resolvePython(): string {
  const configured = vscode.workspace.getConfiguration("mimir").get<string>("pythonPath");
  if (configured) {
    return configured;
  }
  if (process.env.MIMIR_PYTHON) {
    return process.env.MIMIR_PYTHON;
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

  const provider = new MimirAgentViewProvider(context.extensionUri, context.globalState);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("mimir.chatView", provider, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  // `onStartupFinished` activates us with the chat panel possibly still closed,
  // so the remembered endpoint is probed here rather than on the first view
  // resolve: MIMIR is already connected by the time the user opens the panel.
  void provider.maybeAutoConnect();

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
}

// ── Sidebar WebviewViewProvider ───────────────────────────────────────────────

class MimirAgentViewProvider implements vscode.WebviewViewProvider {
  private _ws: WebSocket | undefined;
  private _view: vscode.WebviewView | undefined;
  private _pendingMessages: string[] = [];
  /** Auto-connect is a startup courtesy: once per window, never on a view reload. */
  private _autoConnectTried = false;
  /**
   * Endpoint we auto-connected to before any webview existed. The socket is live
   * but the React app has never seen the server's greeting, so the first view to
   * resolve re-attaches (see `resolveWebviewView`) to be told `ready`.
   */
  private _headlessConnect: RememberedEndpoint | undefined;
  /**
   * Address of the server this window talks to, learned from the server's own
   * "Listening on ws://…" line (it binds an OS-assigned port, so each VS Code
   * window gets its own) or copied from the `mimir.wsUrl` override. Every
   * reconnect goes through this — there is no fixed address to fall back on.
   */
  private _wsUrl: string | undefined;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly memento: vscode.Memento,
  ) {}

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

    // No blind auto-connect: nothing is started unless the user ticked "Remember
    // this address" and that address answers (see _maybeAutoConnect). This keeps
    // us from spawning connections on HPC front nodes.

    // Send config immediately so the webview can populate the model list.
    this._sendConfig();
    // Seed the active-file chip on (re)load.
    this.pushActiveEditor();

    if (this._headlessConnect) {
      // We connected at startup, before this webview existed. Its `ready` frame
      // was sent to nobody, so re-attach: the Python server greets every new
      // client, which is what moves the UI from "connecting" to "connected".
      const saved = this._headlessConnect;
      this._headlessConnect = undefined;
      view.webview.postMessage({ type: "auto_connect", ...saved });
      const stale = this._ws;
      if (stale && this._wsUrl) {
        this._ws = undefined;
        stale.close();
        this._connectToServer(this._wsUrl);
      }
      // No socket yet — the connect already in flight will attach on its own,
      // and this webview is here to receive the `ready` it brings back. Opening
      // a second one here would leave the server with two clients.
    } else {
      // …otherwise reconnect on our own to the address the user asked us to
      // remember (no-op when activation already did it, or nothing is stored).
      void this._maybeAutoConnect();
    }
  }

  /**
   * Reconnect to the remembered address, if it answers.
   *
   * The user opts in with the "Remember this address" checkbox in the connect
   * form; we store backend/address/model (never a key) in global state. Once per
   * window — at activation, so an unopened chat panel still comes up connected —
   * we probe the endpoint's model list, the same request the form makes, and
   * only start the server when it replies. An unreachable address (laptop off
   * the VPN, compute node released) is not an error here: the connect form
   * simply comes up as usual, pre-filled.
   */
  async maybeAutoConnect(): Promise<void> {
    return this._maybeAutoConnect();
  }

  private async _maybeAutoConnect(): Promise<void> {
    if (this._autoConnectTried) return;
    this._autoConnectTried = true;

    const saved = this._remembered();
    if (!saved) return;

    const verifySsl = vscode.workspace.getConfiguration("mimir").get<boolean>("vllmVerifySsl", true);
    try {
      await fetchModels(saved.backend as DiscoverableBackend, saved.baseUrl, verifySsl, 3000);
    } catch {
      return; // endpoint not up — leave the user on the connect form
    }
    // The user can have been impatient and connected by hand while we probed
    // (the server spawns before the socket exists, so check both).
    if (this._ws || (serverProcess && !serverProcess.killed)) return;

    if (this._view) {
      this._view.webview.postMessage({ type: "auto_connect", ...saved });
    } else {
      // No webview yet — remember to greet the first one that resolves.
      this._headlessConnect = saved;
    }
    // Startup connect: don't pop the server log over whatever the user opened.
    this._startServerAndConnect(saved.model, saved.backend, saved.baseUrl, "", { silent: true });
  }

  /** The remembered endpoint, or undefined when nothing valid is stored. */
  private _remembered(): RememberedEndpoint | undefined {
    const saved = this.memento.get<RememberedEndpoint>(REMEMBERED_KEY);
    if (!saved || !saved.baseUrl) return undefined;
    // Only endpoints the user gives an address for; Anthropic needs a key we
    // deliberately never persist, so it can't be auto-connected.
    if (saved.backend !== "vllm" && saved.backend !== "ollama") return undefined;
    return saved;
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

  /**
   * Seed the connect form with the defaults from settings.
   *
   * Only the starting values: the user edits the address in the form, and the
   * model list comes from the endpoint itself (see `_sendModels`), so a working
   * setup needs no `.vscode/settings.json` at all.
   */
  private _sendConfig(): void {
    const cfg = vscode.workspace.getConfiguration("mimir");

    this._view?.webview.postMessage({
      type: "config",
      backend: cfg.get<string>("backend") ?? "vllm",
      vllmBaseUrl: cfg.get<string>("vllmBaseUrl") ?? "http://127.0.0.1:8000",
      ollamaBaseUrl: cfg.get<string>("ollamaUrl") ?? "http://127.0.0.1:11434",
      anthropicModels: cfg.get<string[]>("anthropicAvailableModels") ?? [],
      remembered: this._remembered() ?? null,
    });
  }

  private _connectToServer(wsUrl: string, retryCount = 0): void {
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
      // A socket we replaced ourselves (headless re-attach) must not drive the
      // retry/teardown of the one that took its place.
      if (this._ws !== ws) return;
      const stillRunning = serverProcess && !serverProcess.killed;
      if (retryCount < maxRetries && stillRunning) {
        // Server still starting up — retry after 2 seconds
        setTimeout(() => this._connectToServer(wsUrl, retryCount + 1), 2000);
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

  // Kill the ws_server this window owns so a fresh connect always starts from a
  // clean slate. The spawn command `exec`s python, so the process we track is the
  // server itself and killing it frees its port — no other window is touched.
  // A server we only attached to (the `mimir.wsUrl` override) is not ours to kill,
  // and `serverProcess` is undefined in that case.
  private _teardownServer(): void {
    if (serverProcess && !serverProcess.killed) {
      serverProcess.kill();
    }
    serverProcess = undefined;
  }

  /**
   * Start the WS server here and connect to it.
   *
   * The server always runs on the machine VS Code runs on; *baseUrl* points it at
   * an LLM endpoint that is already serving (vLLM or Ollama), wherever that is.
   * Anthropic needs no URL at all — the hosted API is reached over the network
   * with the key from the form or the environment.
   */
  private _startServerAndConnect(
    model: string,
    backend = "vllm",
    baseUrl = "http://127.0.0.1:8000",
    anthropicApiKey = "",
    { silent = false }: { silent?: boolean } = {},
  ): void {
    const cfg = vscode.workspace.getConfiguration("mimir");

    // Attach mode: an explicit `mimir.wsUrl` means the user runs the server
    // themselves (by hand, or on another host), so we connect and start nothing.
    // That server is not ours, so it is never torn down either.
    const override = (cfg.get<string>("wsUrl") ?? "").trim();
    if (override) {
      this._teardownServer();
      this._wsUrl = override;
      this._connectToServer(override);
      return;
    }

    // Clean slate: a server left over from a previous connect (possibly on another
    // backend) is ours, and nothing else should be left holding a port.
    this._teardownServer();

    const pythonPath = resolvePython();
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

    const outputChannel = vscode.window.createOutputChannel("MIMIR Server");
    // A connect the user clicked shows its log; one we started at VS Code launch
    // stays out of the way (the channel is still there to open by hand).
    if (!silent) outputChannel.show();

    const backendArgs =
      backend === "vllm" ? ` --backend vllm --vllm-base-url ${baseUrl}`
      : backend === "ollama" ? ` --backend ollama --ollama-base-url ${baseUrl}`
      : backend === "anthropic" ? " --backend anthropic"
      : "";
    // `--port 0`: the OS picks a free port, so two VS Code windows never contend for
    // one — the server prints the port it actually bound and we connect to that.
    // `exec` replaces the shell with python, so `serverProcess.kill()` reaches the
    // server itself rather than orphaning it on its port.
    const spawnCmd = `exec ${pythonPath} -m mimir.client.ui.ws.ws_server --port 0${model ? ` --model ${model}` : ""}${backendArgs}`;
    // Log the command only — the Claude API key is injected via env below and is
    // deliberately kept out of this string so it never lands in the output channel.
    outputChannel.appendLine(`Starting server: ${spawnCmd}`);

    // Internal HTTPS vLLM routes are often served behind a private
    // CA; when the user disables cert verification, propagate VLLM_VERIFY_SSL so
    // /v1/models model-resolution and chat requests don't hit CERTIFICATE_VERIFY_FAILED.
    const verifyEnv = cfg.get<boolean>("vllmVerifySsl", true) ? {} : { VLLM_VERIFY_SSL: "0" };
    // An HTTP proxy silently swallows requests to an on-prem endpoint, so
    // ws_server would hang on model resolution before ever binding its port.
    const noProxyEnv = backend === "anthropic" ? {} : noProxyFor(baseUrl);
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

    // The address is not knowable in advance (the OS assigns the port), so the
    // server's own startup line is what triggers the connect. Everything before
    // that — MCP handshakes, model resolution — can take a while on a cold start.
    const spawned = serverProcess;
    let listening = false;
    const startupTimer = setTimeout(() => {
      if (listening || serverProcess !== spawned) return;
      outputChannel.appendLine(
        "\nServer did not report a listening address within 120s — giving up. " +
        "Check the endpoint above is reachable, then connect again."
      );
      this._view?.webview.postMessage({ type: "ws_closed" });
    }, 120_000);

    serverProcess.stdout?.on("data", (d: Buffer) => {
      const text = d.toString();
      outputChannel.append(text);
      if (listening) return;
      const m = /Listening on (ws:\/\/\S+)/.exec(text);
      if (m) {
        listening = true;
        clearTimeout(startupTimer);
        this._wsUrl = m[1];
        this._connectToServer(m[1]);
      }
    });
    serverProcess.stderr?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
    serverProcess.on("exit", (code) => {
      clearTimeout(startupTimer);
      outputChannel.appendLine(`\nServer exited (code ${code})`);
      serverProcess = undefined;
      this._view?.webview.postMessage({ type: "ws_closed" });
    });
  }

  /**
   * Ask the endpoint what models it serves and hand the list to the webview.
   *
   * Runs here rather than in React because the webview's CSP forbids HTTP.
   * A failure is reported as text under the address field, never as a modal: the
   * user can still connect (the server resolves the served model itself).
   */
  private async _sendModels(backend: string, baseUrl: string): Promise<void> {
    if (backend !== "vllm" && backend !== "ollama") {
      return;
    }
    const verifySsl = vscode.workspace.getConfiguration("mimir").get<boolean>("vllmVerifySsl", true);
    try {
      const models = await fetchModels(backend as DiscoverableBackend, baseUrl, verifySsl);
      this._view?.webview.postMessage({ type: "models", backend, models });
    } catch (err) {
      this._view?.webview.postMessage({
        type: "models",
        backend,
        models: [],
        error: err instanceof Error ? err.message : String(err),
      });
    }
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
      this._view?.webview.postMessage({ type: "ws_closed" });
      return;
    }

    if (m.type === "get_config") {
      this._sendConfig();
      return;
    }

    if (m.type === "fetch_models") {
      void this._sendModels(
        (m.backend as string | undefined) ?? "vllm",
        (m.baseUrl as string | undefined) ?? "",
      );
      return;
    }

    if (m.type === "connect") {
      const model = (m.model as string | undefined) ?? "";
      const backend = (m.backend as string | undefined) ?? "vllm";
      const baseUrl = (m.baseUrl as string | undefined) ?? "http://127.0.0.1:8000";
      // Claude API key from the webview. Kept in-process only: passed to the
      // ws_server via the ANTHROPIC_API_KEY env var (never a CLI arg or setting),
      // so it never reaches the process list, the output channel, or disk.
      const anthropicApiKey = (m.anthropicApiKey as string | undefined) ?? "";
      const remember = m.remember === true;

      // vLLM and Ollama resolve the served model themselves when none is picked,
      // so only the hosted Claude API needs an explicit one.
      if (!model && backend === "anthropic") {
        vscode.window.showErrorMessage("MIMIR: select a model before connecting.");
        return;
      }

      // "Remember this address" — store the endpoint (never the key) so the next
      // window can reconnect on its own; unchecking it forgets the stored one.
      if (remember && (backend === "vllm" || backend === "ollama")) {
        void this.memento.update(REMEMBERED_KEY, { backend, baseUrl, model });
      } else {
        void this.memento.update(REMEMBERED_KEY, undefined);
      }

      this._startServerAndConnect(model, backend, baseUrl, anthropicApiKey);
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

function startServer(context: vscode.ExtensionContext): void {
  const pythonPath = resolvePython();

  if (serverProcess && !serverProcess.killed) {
    vscode.window.showInformationMessage("MIMIR WS server is already running.");
    return;
  }

  const outputChannel = vscode.window.createOutputChannel("MIMIR Server");
  outputChannel.show();

  const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

  // A fixed port here on purpose: this command exists to run a server you then
  // point `mimir.wsUrl` at, and an OS-assigned port could not be written down in a
  // setting. If 8765 is taken, the bind error shows up in the output channel.
  // `exec` so the process we track — and later kill — is the server, not the shell.
  const spawnCmd = `exec ${pythonPath} -m mimir.client.ui.ws.ws_server --port 8765`;

  serverProcess = cp.spawn("bash", ["-c", spawnCmd], {
    cwd,
    // Anchor the agent's per-workspace state dir (.mimir) and the file-server
    // root to the opened workspace, regardless of the process cwd.
    env: { ...process.env, MCP_FILES_ROOT: cwd },
    stdio: ["ignore", "pipe", "pipe"],
  });

  serverProcess.stdout?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
  serverProcess.stderr?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
  serverProcess.on("exit", (code) => {
    outputChannel.appendLine(`\nServer exited (code ${code})`);
    serverProcess = undefined;
  });

  vscode.window.showInformationMessage("MIMIR WS server started.");
  context.subscriptions.push({
    dispose: () => {
      serverProcess?.kill();
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
