/**
 * The chat view in the side bar: the webview, and what connects it to a server. It
 * decides when to attach, spawn or reconnect; the socket, the server process, the model
 * probe, the file index and the notifications are components it is built from.
 */
import * as vscode from "vscode";
import { ServerConnection } from "./connection";
import { openDiff, openFile, openPatch, openPreview } from "./editors";
import {
  ADDRESSED_BACKENDS, REMEMBERED_KEY, validRemembered, type RememberedEndpoint,
} from "./endpoints";
import { LocalServer, type SpawnRequest } from "./localServer";
import { fetchModels, type DiscoverableBackend } from "./modelList";
import { ModelProbe } from "./modelProbe";
import { Notifier } from "./notifications";
import { serverChannel } from "./serverChannel";
import { findLiveServer, type ServerEntry } from "./serverRegistry";
import { DEFAULT_BACKEND, DEFAULT_OLLAMA_URL, DEFAULT_OPENAI_URL } from "./shared/defaults";
import type { HostMessage, HostMessageOf } from "./shared/hostMessages";
import { showCodeMap } from "./codeMapView";
import { getWebviewHtml } from "./webviewHtml";
import { workspaceRoot } from "./workspace";
import { WorkspaceFiles } from "./workspaceFiles";
import { showToast } from "./notifications";

export class MimirAgentViewProvider implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  /** An auto-connect is already running (or has run) — never start a second one. */
  private autoConnectStarted = false;
  /** A probe is in flight; a view resolving meanwhile must not fire its own. */
  private autoConnectProbing = false;
  /**
   * Endpoint we auto-connected to, kept until a webview has actually been told.
   * The React app is what shows the connecting state, and it mounts long after we
   * start — so this is replayed on every `get_config` (its mount handshake) until
   * the socket is up.
   */
  private pendingAutoConnect: RememberedEndpoint | undefined;
  /**
   * The model list the auto-connect probe already received, kept for the webview.
   *
   * On an auto-connect nothing else asks the endpoint what it serves, and the status
   * bar's model picker needs that list. Replayed on `get_config` rather than posted
   * once, because the React app usually does not exist yet when the probe answers.
   */
  private autoConnectModels: { backend: string; models: string[] } | undefined;

  private readonly connection = new ServerConnection({
    post: (m) => this.post(m),
    onServerText: (text) => {
      // Read by the host as well as forwarded: one of these messages changes who owns
      // the server process, which is this layer's business and not the webview's.
      this.noteOwnership(text);
      this.notifier.notify(text);
    },
    serverStarting: () => this.server.childRunning,
  });
  private readonly models = new ModelProbe((m) => this.post(m));
  private readonly files = new WorkspaceFiles();
  private readonly notifier = new Notifier(
    () => (this.view?.visible ?? false) && vscode.window.state.focused,
    // Straight down the socket rather than through the webview: the pane may not be
    // loaded, which is the situation that produced the notification. The webview learns
    // of the switch from the session_loaded that follows.
    (sessionId) => this.connection.send(JSON.stringify({ type: "switch_session", session_id: sessionId })),
  );

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly memento: vscode.Memento,
    private readonly server: LocalServer,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this.extensionUri, "dist"),
        vscode.Uri.joinPath(this.extensionUri, "images"),
      ],
    };
    view.webview.html = getWebviewHtml(view.webview, this.extensionUri);

    view.webview.onDidReceiveMessage((msg: unknown) => {
      this.handleFromWebview(msg);
    });

    // Config at once, so the webview can populate the connect form.
    void this.sendConfig();
    // Seed the active-file chip on (re)load.
    this.pushActiveEditor();

    // Reconnect on our own to the address the user asked us to remember (a no-op when
    // activation already started it, or nothing is stored); nothing is started blindly,
    // which keeps us from spawning connections on HPC front nodes. Catching a webview up
    // on a connect it never saw is *not* done here: nothing posted from
    // `resolveWebviewView` is guaranteed to be heard, because the React app has not
    // loaded its message listener yet. That happens on `get_config` (`resumeAutoConnect`).
    void this.maybeAutoConnect();
  }

  /**
   * Reconnect to the remembered address (opted into in the connect form; never a key),
   * once per window at activation, if its model list answers. An unreachable address
   * just leaves the form up, pre-filled.
   */
  async maybeAutoConnect(): Promise<void> {
    if (this.autoConnectStarted || this.autoConnectProbing) return;

    // A server this workspace already has comes first, and it does not depend on a
    // remembered address: a run left going when the window closed is still serving,
    // and the point of leaving it there is that reopening the window finds it. Asked
    // before the endpoint probe because it is both cheaper and more certain — the
    // probe asks whether an LLM endpoint answers, this asks whether the agent that was
    // already working is still working.
    if (await this.attachToRunningServer()) return;

    const saved = this.remembered();
    if (!saved) return;

    const verifySsl = vscode.workspace.getConfiguration("mimir").get<boolean>("vllmVerifySsl", true);
    this.autoConnectProbing = true;
    try {
      // Longer than the form's own probe: nothing waits on this (the connect form is
      // already usable), and the address is one that worked before, so a slow VPN or a
      // still-starting node is worth waiting out.
      const models = await fetchModels(
        saved.backend as DiscoverableBackend, saved.baseUrl, verifySsl, 30000);
      this.autoConnectModels = { backend: saved.backend, models };
    } catch {
      // Endpoint not up — leave the user on the connect form. The attempt is not
      // marked as started, so opening the chat panel later probes once more: at
      // VS Code startup the network (VPN, compute node) is often not up yet.
      return;
    } finally {
      this.autoConnectProbing = false;
    }
    // The user can have been impatient and connected by hand while we probed
    // (the server spawns before the socket exists, so check both).
    if (this.connection.active || this.server.childRunning) return;
    this.autoConnectStarted = true;

    // Replayed on the webview's `get_config` until the socket is up, because the
    // React app may not be listening yet (or may not exist at all).
    this.pendingAutoConnect = saved;
    this.announceAutoConnect();
    // Both here and on `get_config`, because neither alone covers it: the webview
    // may not exist yet when the probe answers, and the probe may not have answered
    // yet when the webview mounts. Posting the list twice costs a state set.
    this.announceModels();
    // Startup connect: don't pop the server log over whatever the user opened.
    void this.startServerAndConnect({
      model: saved.model, backend: saved.backend, baseUrl: saved.baseUrl,
      anthropicApiKey: "", silent: true,
    }).catch((err) => {
      // Reported rather than swallowed: a rejection here leaves the panel on
      // "connecting" with nothing said about why.
      serverChannel().appendLine(
        `Starting the server failed: ${err instanceof Error ? err.message : String(err)}`);
      this.pendingAutoConnect = undefined;
      this.post({ type: "ws_closed" });
    });
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
    this.post({ type: "active_editor", file, selection });
  }

  private post(message: Record<string, unknown>): void {
    this.view?.webview.postMessage(message);
  }

  /**
   * Attach to the server this workspace already has, if it has one; true when it did,
   * so the connect form is skipped. The registry pid stands in for a child process: the
   * server drops its claim on this socket, and now lives as long as this window.
   */
  private async attachToRunningServer(
    { silent = true }: { silent?: boolean } = {},
  ): Promise<boolean> {
    if (this.connection.active || this.server.childRunning) return false;
    // An explicit override is the user's own statement about where the server is, and
    // it wins: the spawn path defers to it for the same reason.
    const override = (vscode.workspace.getConfiguration("mimir")
      .get<string>("wsUrl") ?? "").trim();
    if (override) return false;

    let entry: ServerEntry | undefined;
    try {
      entry = await findLiveServer(workspaceRoot());
    } catch {
      return false;   // advisory, never authoritative
    }
    if (!entry) return false;

    const log = serverChannel();
    if (!silent) log.show();
    log.appendLine(
      `Attaching to the MIMIR server already running for this workspace ` +
      `(${entry.url}, pid ${entry.pid}). This window takes it over: closing it stops ` +
      `the server, unless the run is left going again. Starting a second server for ` +
      `one workspace would have both write the same session journal.`
    );
    // The handle a window that spawns nothing would otherwise lack. Set before the
    // connect, so a socket that closes immediately still leaves something to end.
    this.server.adopt(entry.pid);
    this.connection.attachLog = log;
    this.autoConnectStarted = true;
    // So a webview that mounts after this shows "connecting" rather than the form.
    this.pendingAutoConnect = { backend: "", baseUrl: entry.url, model: entry.model ?? "" };
    this.announceAutoConnect();
    this.connection.connect(entry.url);
    return true;
  }

  /**
   * Notice the server telling us it has detached.
   *
   * Nothing is rendered: the point is the ownership change. The server has re-pointed
   * its output away from the pipe we hold and left our process group, so from here on
   * closing this window leaves it working — which is what the user asked for. Read in
   * the host rather than the webview because it is the host that does the killing.
   */
  private noteOwnership(text: string): void {
    let msg: {
      type?: string; pid?: number; autonomy?: string; log?: string | null;
      detached?: boolean;
    };
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    if (msg?.type !== "detached") return;
    const log = serverChannel();
    // Absent means detached: a server that sends no flag sends the message only to
    // announce a detachment.
    if (msg.detached === false) {
      this.server.noteDetached(false);
      log.appendLine(
        `This server belongs to this window again (pid ${msg.pid ?? "?"}); closing ` +
        `the window stops it.`
      );
      return;
    }
    this.server.noteDetached(true);
    log.appendLine(
      `This server has detached (pid ${msg.pid ?? "?"}, autonomy ` +
      `${msg.autonomy ?? "?"}). It is left running when the window closes; its ` +
      `output continues in ${msg.log ?? "its log"}.`
    );
  }

  /** Hand the webview the model list the auto-connect probe already has. */
  private announceModels(): void {
    if (!this.view || !this.autoConnectModels) return;
    this.post({ type: "models", ...this.autoConnectModels });
  }

  /** Tell the webview, if there is one, which endpoint we are connecting to. */
  private announceAutoConnect(): void {
    if (!this.view || !this.pendingAutoConnect) return;
    this.post({ type: "auto_connect", ...this.pendingAutoConnect });
  }

  /**
   * Bring a freshly mounted webview up to date (on `get_config`, the first moment it
   * listens): replay `auto_connect` for a connect in flight, or re-attach a live socket
   * so the server greets the webview with `ready`.
   */
  private resumeAutoConnect(): void {
    if (!this.pendingAutoConnect) return; // nothing was missed — leave the socket alone
    this.announceAutoConnect();
    this.connection.reopen();
    // The webview has now heard it — replaying again would throw a connected UI
    // back into "connecting" and needlessly cycle a healthy socket.
    this.pendingAutoConnect = undefined;
  }

  /** The remembered endpoint, or undefined when nothing valid is stored. */
  private remembered(): RememberedEndpoint | undefined {
    return validRemembered(this.memento.get<RememberedEndpoint>(REMEMBERED_KEY));
  }

  /**
   * Seed the connect form with the defaults from settings.
   *
   * Only the starting values: the user edits the address in the form, and the
   * model list comes from the endpoint itself (see `ModelProbe`), so a working
   * setup needs no `.vscode/settings.json` at all.
   */
  private async sendConfig(): Promise<void> {
    const cfg = vscode.workspace.getConfiguration("mimir");

    // Whether this workspace already has a server, so a disconnected panel can offer
    // to rejoin it rather than asking for an endpoint it will not use. A run left
    // going is the one thing the user came back for, and a connect form is a poor way
    // to say "it is still here".
    let running: ServerEntry | undefined;
    try {
      running = await findLiveServer(workspaceRoot());
    } catch {
      running = undefined;
    }

    this.post({
      type: "config",
      running: running
        ? { url: running.url, pid: running.pid, model: running.model ?? "",
            autonomy: running.autonomy ?? "", detached: running.detached === true }
        : null,
      backend: cfg.get<string>("backend") ?? DEFAULT_BACKEND,
      vllmBaseUrl: cfg.get<string>("vllmBaseUrl") ?? DEFAULT_OPENAI_URL,
      rayBaseUrl: cfg.get<string>("rayBaseUrl") ?? DEFAULT_OPENAI_URL,
      ollamaBaseUrl: cfg.get<string>("ollamaUrl") ?? DEFAULT_OLLAMA_URL,
      anthropicModels: cfg.get<string[]>("anthropicAvailableModels") ?? [],
      remembered: this.remembered() ?? null,
    });
  }

  /**
   * Release the server this window is responsible for, so a fresh connect starts from
   * a clean slate. The spawn command `exec`s python, so killing the tracked process
   * frees its port and touches no other window. A detached server and one reached
   * through `mimir.wsUrl` are not ours to end: the socket is dropped, the process left.
   */
  private teardownServer(): void {
    this.connection.retire();
    this.server.release();
  }

  /**
   * Connect, starting a server only if this workspace does not already have one.
   *
   * Two servers on one workspace would interleave one journal and its `seq`. The server
   * refuses a second claim itself; this keeps the ordinary path from relying on that.
   */
  private async startServerAndConnect(req: SpawnRequest): Promise<void> {
    if (await this.attachToRunningServer({ silent: req.silent })) return;

    // Attach mode: an explicit `mimir.wsUrl` means the user runs the server
    // themselves (by hand, or on another host), so we connect and start nothing.
    // That server is not ours, so it is never torn down either.
    const override = (vscode.workspace.getConfiguration("mimir").get<string>("wsUrl") ?? "").trim();
    if (override) {
      // Said in the server channel anyway: a stray `mimir.wsUrl` (easily left behind
      // in a workspace's .vscode/settings.json) otherwise silently disables the
      // whole spawn path, and the connect form just reappears with no trace of why.
      const attachLog = serverChannel();
      if (!req.silent) attachLog.show();
      attachLog.appendLine(
        `Attaching to ${override} — no server is started here because the ` +
        `"mimir.wsUrl" setting is set. Clear it to let MIMIR start its own server.`
      );
      this.connection.attachLog = attachLog;
      this.teardownServer();
      this.connection.connect(override);
      return;
    }
    // A spawned connect is not attached to anything.
    this.connection.attachLog = undefined;

    // Clean slate: a server left over from a previous connect (possibly on another
    // backend) is ours, and nothing else should be left holding a port.
    this.teardownServer();

    const outcome = await this.server.spawn(req, () => {
      // Nothing left to catch a webview up on — don't replay "connecting", and
      // don't offer the models of an endpoint that is no longer being served.
      this.pendingAutoConnect = undefined;
      this.autoConnectModels = undefined;
      this.post({ type: "ws_closed" });
    });
    if (outcome.kind === "superseded") return;
    if (outcome.kind === "gave_up") {
      this.pendingAutoConnect = undefined;
      this.post({ type: "ws_closed" });
      return;
    }
    this.connection.connect(outcome.url);
  }

  /** One handler per message the webview posts (`HostMessage`). */
  private readonly webviewHandlers: {
    [T in HostMessage["type"]]: (m: HostMessageOf<T>) => void;
  } = {
    open_file: openFile,
    open_preview: openPreview,
    open_diff: openDiff,
    open_patch: openPatch,
    disconnect: () => this.disconnect(),
    // The "@" dropdown, asking which workspace files match what has been typed. The
    // query is echoed back, so the webview drops an answer the typing has overtaken.
    file_search: (m) => {
      const query = typeof m.query === "string" ? m.query : "";
      void this.files.matching(query).then((files) => this.post({ type: "file_results", query, files }));
    },
    get_config: () => this.onWebviewMounted(),
    fetch_models: (m) => void this.models.probe(m.backend ?? DEFAULT_BACKEND, m.baseUrl ?? ""),
    connect: (m) => this.connect(m),
    ws_send: (m) => this.forwardToServer(m.payload),
    show_code_map: (m) => showCodeMap(m.map, m.session, this.extensionUri,
      () => this.forwardToServer(JSON.stringify({ type: "code_map" }))),
  };

  private handleFromWebview(msg: unknown): void {
    const m = msg as HostMessage;
    const handler = this.webviewHandlers[m?.type] as ((m: HostMessage) => void) | undefined;
    handler?.(m);
  }

  private disconnect(): void {
    this.connection.close();
    // A detached server is not this window's to end: disconnecting leaves it running.
    this.server.release();
    this.post({ type: "ws_closed" });
  }

  /** The webview's mount handshake: the one point where it is certainly listening, so
   *  it is also where a connect it missed is replayed. */
  private onWebviewMounted(): void {
    void this.sendConfig();
    this.announceModels();
    this.resumeAutoConnect();
  }

  private connect(m: HostMessageOf<"connect">): void {
    const model = m.model ?? "";
    const backend = m.backend ?? DEFAULT_BACKEND;
    const baseUrl = m.baseUrl ?? DEFAULT_OPENAI_URL;
    // The Claude API key stays in-process: it reaches the server through its
    // environment, never a CLI arg, a setting or the disk.
    const anthropicApiKey = m.anthropicApiKey ?? "";
    // The addressed endpoints resolve the served model themselves.
    if (!model && backend === "anthropic") {
      showToast("error", "MIMIR: select a model before connecting.");
      return;
    }
    // "Remember this address": the endpoint (never the key), for the next window.
    if (m.remember === true && ADDRESSED_BACKENDS.includes(backend)) {
      void this.memento.update(REMEMBERED_KEY, { backend, baseUrl, model });
    } else {
      void this.memento.update(REMEMBERED_KEY, undefined);
    }
    // A hand-typed connect supersedes any auto-connect still being replayed.
    this.pendingAutoConnect = undefined;
    this.autoConnectModels = undefined;
    this.autoConnectStarted = true;
    void this.startServerAndConnect({ model, backend, baseUrl, anthropicApiKey, silent: false })
      .catch((err) => {
        serverChannel().appendLine(
          `Starting the server failed: ${err instanceof Error ? err.message : String(err)}`);
        this.post({ type: "ws_closed" });
      });
  }

  /** A message for the Python server, queued while the socket is not open. */
  private forwardToServer(payload: string): void {
    // Asking to detach stops the kill at once: closing the editor during the round trip
    // must not end a run the user asked to keep. Taking it back (`enabled: false`)
    // waits for the server's answer.
    if (payload.includes('"detach"')) {
      try {
        const parsed = JSON.parse(payload) as { type?: string; enabled?: boolean };
        if (parsed.type === "detach" && parsed.enabled !== false) {
          this.server.noteDetached(true);
        }
      } catch {
        // Not ours to parse; leave the flag alone.
      }
    }
    this.connection.send(payload);
  }
}
