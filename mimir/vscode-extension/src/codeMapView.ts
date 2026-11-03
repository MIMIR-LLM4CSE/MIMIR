/**
 * The code map: one editor tab drawing the workspace's architecture, from the map the
 * server answers (`shared.code_map`). The page is its own bundle (`dist/codemap.js`); it
 * asks for the map once loaded, opens files at a line through the host, and asks the
 * server again on its ⟳ or for another proxy's steps (or every proxy's).
 */
import * as vscode from "vscode";
import * as fs from "fs";
import { openFile } from "./editors";
import { getNonce } from "./webviewHtml";

/** The tab, while it is open. */
let mapPanel: vscode.WebviewPanel | undefined;

/** Set while the tab's own refresh is on its way: its answer does not reveal the tab. */
let quiet = false;

/** The last answer, for a page that loads after it arrived. */
let last: { map: unknown; session?: unknown; focus?: string } = { map: undefined };

/** What the tab can ask of the host. */
type TabMessage =
  | { type: "ready" }
  | { type: "refresh"; proxy?: string; combined?: boolean }
  | { type: "open_file"; file: string; rel?: string; line?: number }
  | { type: "prompt"; text: string };

/** Whose steps the tab asks for: a proxy's, or every proxy's; the active session's without. */
export interface StepsPick { proxy?: string; combined?: boolean }

/** What the tab can ask of the chat: the map again, or a request put to the agent. */
export interface MapActions {
  refresh: (pick: StepsPick) => void;
  prompt: (text: string) => void;
}

/** Show *map* (and the session's steps) in the tab, opening it if needed, on the
 *  component *focus* when given; *actions* ask the server again or put a request to the agent. */
export function showCodeMap(map: unknown, session: unknown, focus: string | undefined,
                            extensionUri: vscode.Uri, actions: MapActions): void {
  last = { map, session, focus };
  if (!mapPanel) {
    mapPanel = vscode.window.createWebviewPanel(
      "mimir.codeMap",
      "Code map",
      { viewColumn: vscode.ViewColumn.Active, preserveFocus: false },
      { enableScripts: true, retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, "dist")] }
    );
    mapPanel.webview.html = pageHtml(mapPanel.webview, extensionUri);
    mapPanel.webview.onDidReceiveMessage((m: TabMessage) => {
      if (m?.type === "ready") void mapPanel?.webview.postMessage({ type: "map", ...last });
      else if (m?.type === "refresh") {
        // The answer to the tab's own question updates it where it is: the tab asks again
        // on its own while a request is at work, and must not take the focus each time.
        quiet = true;
        actions.refresh({
          ...(typeof m.proxy === "string" && m.proxy ? { proxy: m.proxy } : {}),
          ...(m.combined === true ? { combined: true } : {}),
        });
      }
      else if (m?.type === "prompt" && typeof m.text === "string") actions.prompt(m.text);
      else if (m?.type === "open_file" && typeof m.file === "string") {
        // The server's absolute path when it exists on this machine, else the path
        // relative to the map's root, resolved against the open folder.
        const file = fs.existsSync(m.file) || !m.rel ? m.file : m.rel;
        openFile({ type: "open_file", file, line: m.line });
      }
    });
    mapPanel.onDidDispose(() => { mapPanel = undefined; });
  } else {
    if (!quiet) mapPanel.reveal(undefined, false);
    quiet = false;
    void mapPanel.webview.postMessage({ type: "map", ...last });
  }
}

/** Tell an open tab that the server had no map to give, so a ⟳ it waits on says so
 *  instead of turning for ever. */
export function codeMapUnavailable(reason?: string): void {
  quiet = false;
  void mapPanel?.webview.postMessage({ type: "unavailable", ...(reason ? { reason } : {}) });
}

/** Tell an open tab that what it draws may have moved; it asks for the map again itself,
 *  with its own pick of proxy. Nothing when no tab is open. */
export function codeMapStale(): void {
  void mapPanel?.webview.postMessage({ type: "stale" });
}

function pageHtml(webview: vscode.Webview, extensionUri: vscode.Uri): string {
  const script = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "dist", "codemap.js"));
  const nonce = getNonce();
  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src 'nonce-${nonce}'; style-src ${webview.cspSource} 'unsafe-inline';">
  <title>Code map</title>
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}" src="${script}"></script>
</body>
</html>`;
}
