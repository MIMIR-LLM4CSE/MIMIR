import * as vscode from "vscode";

/** The chat webview's page: the React bundle, under a nonce-locked CSP. */
export function getWebviewHtml(
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
