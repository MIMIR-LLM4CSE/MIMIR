/**
 * The code map: one editor tab drawing the workspace's architecture, from the map the
 * server answers (`shared.code_map`). The page is its own bundle (`dist/codemap.js`); it
 * asks for the map once loaded, opens files at a line through the host, and asks the
 * server again on its ⟳.
 */
import * as vscode from "vscode";
import { openFile } from "./editors";
import { getNonce } from "./webviewHtml";

/** The tab, while it is open. */
let mapPanel: vscode.WebviewPanel | undefined;

/** The last map the server answered, for a page that loads after it arrived. */
let lastMap: unknown;

/** What the tab can ask of the host. */
type TabMessage =
  | { type: "ready" }
  | { type: "refresh" }
  | { type: "open_file"; file: string; line?: number };

/** Show *map* in the tab, opening it if needed; *refresh* asks the server again. */
export function showCodeMap(map: unknown, extensionUri: vscode.Uri, refresh: () => void): void {
  lastMap = map;
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
      if (m?.type === "ready") void mapPanel?.webview.postMessage({ type: "map", map: lastMap });
      else if (m?.type === "refresh") refresh();
      else if (m?.type === "open_file" && typeof m.file === "string") openFile(m);
    });
    mapPanel.onDidDispose(() => { mapPanel = undefined; });
  } else {
    mapPanel.reveal(undefined, false);
    void mapPanel.webview.postMessage({ type: "map", map });
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
