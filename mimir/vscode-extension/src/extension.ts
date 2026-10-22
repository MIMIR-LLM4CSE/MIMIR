/**
 * The extension's entry: registers the chat view, the `mimir-diff:` documents and the
 * commands, and on deactivation ends the server this window is responsible for.
 */
import * as vscode from "vscode";
import { MimirAgentViewProvider } from "./chatView";
import { diffContentProvider } from "./editors";
import { LocalServer } from "./localServer";
import { stopPlanWatch } from "./planPreview";
import { maybeSelfUpdate } from "./selfUpdate";
import { disposeServerChannel } from "./serverChannel";
import { affectsServerEnv } from "./serverEnv";

/** The server this window is responsible for, reachable from `deactivate()`. */
let localServer: LocalServer | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const server = new LocalServer(context.extensionUri.fsPath);
  localServer = server;
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider("mimir-diff", diffContentProvider)
  );

  const provider = new MimirAgentViewProvider(context.extensionUri, context.globalState, server);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("mimir.chatView", provider, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  // `onStartupFinished` activates us with the chat panel possibly still closed,
  // so the remembered endpoint is probed here rather than on the first view
  // resolve: MIMIR is already connected by the time the user opens the panel.
  void provider.maybeAutoConnect();

  // Shared-release mode: offer to pick up the current release's extension when
  // it is newer than the one installed. No-op where the release tree is absent.
  maybeSelfUpdate(context);

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
    ),
    vscode.commands.registerCommand("mimir.startServer", () => server.startOnFixedPort(context)),
    vscode.commands.registerCommand("mimir.stopServer", () => server.stopFromCommand()),
  );

  // The server reads these settings once, at start: say so, once per burst of edits.
  let noticeShown = false;
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (event) => {
      if (noticeShown || !affectsServerEnv(event)) return;
      noticeShown = true;
      const choice = await vscode.window.showInformationMessage(
        "MIMIR passes this setting to its server when the server starts. " +
        "A server already running keeps the old value until it is stopped.",
        "Stop Server"
      );
      noticeShown = false;
      if (choice === "Stop Server") await server.stopFromCommand();
    })
  );
}

export function deactivate(): void {
  stopPlanWatch();
  // The server is left running when detached; the timer mirroring its log is the one
  // part of that which must not survive a disposed extension host.
  localServer?.release();
  disposeServerChannel();
}
