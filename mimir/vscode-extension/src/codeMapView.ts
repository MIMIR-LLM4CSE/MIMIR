/**
 * The code map: one editor tab drawing the workspace's architecture, from the map the
 * server answers (`shared.code_map`). The page is its own bundle (`dist/codemap.js`); it
 * asks for the map once loaded, opens files at a line through the host, and asks the
 * server again on its ⟳.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import { openFile } from "./editors";
import { getNonce } from "./webviewHtml";

/** The tab, while it is open. */
let mapPanel: vscode.WebviewPanel | undefined;

/** The last answer, for a page that loads after it arrived. */
let last: { map: unknown; session?: unknown; focus?: string } = { map: undefined };

/** What the tab can ask of the host. */
type TabMessage =
  | { type: "ready" }
  | { type: "refresh" }
  | { type: "open_file"; file: string; rel?: string; line?: number }
  | { type: "prompt"; text: string };

/** What the tab can ask of the chat: the map again, or a request put to the agent. */
export interface MapActions {
  refresh: () => void;
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
      else if (m?.type === "refresh") actions.refresh();
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
    mapPanel.reveal(undefined, false);
    void mapPanel.webview.postMessage({ type: "map", ...last });
  }
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
