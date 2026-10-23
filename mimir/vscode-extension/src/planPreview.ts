/**
 * The plan preview: our own panel, rendered from the bytes on disk.
 *
 * VS Code's Markdown preview renders a cached document and can show a stale plan, so
 * each show reads the file, renders it with the Markdown extension's engine
 * (`markdown.api.render`) into a panel we own: one tab for every plan. Relative image
 * paths become webview URIs; link clicks come back here to resolve.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { showToast } from "./notifications";

/** The plan panel, while it is open. */
let planPanel: vscode.WebviewPanel | undefined;

/** Absolute path of the plan the panel currently mirrors. */
let currentPlanPath: string | undefined;

/** Bumped on every render, so a slow render cannot overwrite a newer one. */
let planRenderSeq = 0;

/** Poller for the plan the panel shows, if any. */
let previewWatch: { path: string; mtimeMs: number; size: number; timer: NodeJS.Timeout } | undefined;

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
}

/** The Markdown extension's folder, for its preview stylesheet. */
function markdownMediaUri(): vscode.Uri | undefined {
  const ext = vscode.extensions.getExtension("vscode.markdown-language-features");
  return ext ? vscode.Uri.joinPath(ext.extensionUri, "media") : undefined;
}

/** True for `https:`, `data:`, `vscode-resource:`… and protocol-relative URLs. */
function hasScheme(ref: string): boolean {
  return /^[a-z][\w+.-]*:/i.test(ref) || ref.startsWith("//");
}

/** `ref` from the plan, as an absolute path (fragment and query dropped). */
function resolveFromPlan(ref: string, planAbs: string): string {
  let p = ref.replace(/[?#].*$/, "");
  try {
    p = decodeURI(p);
  } catch {
    /* keep it as written */
  }
  return path.isAbsolute(p) ? p : path.join(path.dirname(planAbs), p);
}

/**
 * Point every local `<img src>` at a webview URI. Returns the rewritten HTML and
 * the folders those images live in, which the panel must be allowed to read.
 */
function rewriteImages(html: string, planAbs: string, webview: vscode.Webview): { html: string; dirs: string[] } {
  const dirs = new Set<string>();
  const out = html.replace(/(<img\b[^>]*?\bsrc=")([^"]*)(")/gi, (all, pre: string, src: string, post: string) => {
    if (!src || hasScheme(src)) return all;
    const abs = resolveFromPlan(src.replace(/&amp;/g, "&"), planAbs);
    dirs.add(path.dirname(abs));
    return pre + webview.asWebviewUri(vscode.Uri.file(abs)).toString() + post;
  });
  return { html: out, dirs: [...dirs] };
}

/** Follow a link clicked in the plan panel. In-page anchors never get here. */
function openPlanLink(href: string): void {
  const planAbs = currentPlanPath;
  if (hasScheme(href) && !/^file:/i.test(href)) {
    void vscode.env.openExternal(vscode.Uri.parse(href));
    return;
  }
  if (!planAbs) return;
  const abs = /^file:/i.test(href) ? vscode.Uri.parse(href).fsPath : resolveFromPlan(href, planAbs);
  if (!fs.existsSync(abs)) {
    void showToast("warn", `MIMIR: ${abs} not found`);
    return;
  }
  if (fs.statSync(abs).isDirectory()) {
    void vscode.commands.executeCommand("revealInExplorer", vscode.Uri.file(abs));
    return;
  }
  vscode.workspace.openTextDocument(vscode.Uri.file(abs)).then(
    (doc) => vscode.window.showTextDocument(doc, { preview: true, viewColumn: vscode.ViewColumn.One }),
    () => vscode.commands.executeCommand("vscode.open", vscode.Uri.file(abs))
  );
}

/** Hands link clicks to the extension; scrolls in-page anchors itself. */
const PLAN_SCRIPT = `
const vscode = acquireVsCodeApi();
document.addEventListener("click", (e) => {
  const a = e.target.closest && e.target.closest("a[href]");
  if (!a) return;
  const href = a.getAttribute("href");
  e.preventDefault();
  if (href.startsWith("#")) {
    const id = decodeURIComponent(href.slice(1));
    const el = document.getElementById(id) || document.getElementsByName(id)[0];
    if (el) el.scrollIntoView();
    return;
  }
  vscode.postMessage({ type: "open", href });
});`;

/** Read the current plan from disk and write it into the panel. */
async function renderPlan(): Promise<void> {
  const panel = planPanel;
  const abs = currentPlanPath;
  if (!panel || !abs) return;
  const seq = ++planRenderSeq;
  let body: string;
  try {
    const text = fs.readFileSync(abs, "utf8");
    try {
      body = await vscode.commands.executeCommand<string>("markdown.api.render", text);
    } catch {
      // Markdown extension disabled: the plan is still readable as text.
      body = `<pre>${escapeHtml(text)}</pre>`;
    }
  } catch (e) {
    body = `<p>Cannot read ${escapeHtml(abs)}</p><pre>${escapeHtml(String(e))}</pre>`;
  }
  if (seq !== planRenderSeq || panel !== planPanel) return;
  const media = markdownMediaUri();
  const images = rewriteImages(body, abs, panel.webview);
  // Roots before html: the new page must be allowed to load what it references.
  panel.webview.options = {
    enableScripts: true,
    localResourceRoots: [
      ...(media ? [media] : []),
      vscode.Uri.file(path.dirname(abs)),
      ...images.dirs.map((d) => vscode.Uri.file(d)),
    ],
  };
  const sheets = media
    ? ["markdown.css", "highlight.css"]
        .map((f) => `<link rel="stylesheet" href="${panel.webview.asWebviewUri(vscode.Uri.joinPath(media, f))}">`)
        .join("\n")
    : "";
  const csp = panel.webview.cspSource;
  const nonce = Math.random().toString(36).slice(2) + Date.now().toString(36);
  panel.title = `Plan — ${path.basename(abs, path.extname(abs))}`;
  panel.webview.html = `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${csp} 'unsafe-inline'; img-src ${csp} https: data:; script-src 'nonce-${nonce}';">
${sheets}
<style>body { padding: 0 26px; font-family: var(--vscode-markdown-font-family, var(--vscode-font-family)); font-size: var(--vscode-markdown-font-size, 14px); }</style>
</head><body class="vscode-body">${images.html}
<script nonce="${nonce}">${PLAN_SCRIPT}</script></body></html>`;
}

/** Stop polling the previewed plan. */
export function stopPlanWatch(): void {
  if (previewWatch) {
    clearInterval(previewWatch.timer);
    previewWatch = undefined;
  }
}

/** Poll `abs` and re-render the panel whenever the file changes on disk. */
function watchPreviewedFile(abs: string): void {
  const stamp = (): { mtimeMs: number; size: number } => {
    try {
      const st = fs.statSync(abs);
      return { mtimeMs: st.mtimeMs, size: st.size };
    } catch {
      return { mtimeMs: 0, size: -1 };
    }
  };
  stopPlanWatch();
  const timer = setInterval(() => {
    if (!previewWatch) return;
    const now = stamp();
    if (now.mtimeMs !== previewWatch.mtimeMs || now.size !== previewWatch.size) {
      Object.assign(previewWatch, now);
      void renderPlan();
    }
  }, 1000);
  previewWatch = { path: abs, ...stamp(), timer };
}

/** Show `abs` in the plan panel, opening the panel if needed. */
export function showPlanPreview(abs: string): void {
  currentPlanPath = abs;
  if (!planPanel) {
    const media = markdownMediaUri();
    planPanel = vscode.window.createWebviewPanel(
      "mimir.plan",
      "Plan",
      // Active, not Beside: the chat lives in the side bar, so "beside" means beside the
      // editor group, splitting the editor area and leaving the plan half a window wide.
      // The plan is a document to read, so it opens as a tab in the group already there
      // and fills it. Focus comes with it, or the tab would open behind the file the
      // user was reading and look like nothing had happened.
      { viewColumn: vscode.ViewColumn.Active, preserveFocus: false },
      { enableScripts: true, localResourceRoots: media ? [media] : [] }
    );
    planPanel.webview.onDidReceiveMessage((m) => {
      if (m?.type === "open" && typeof m.href === "string") openPlanLink(m.href);
    });
    planPanel.onDidDispose(() => {
      planPanel = undefined;
      stopPlanWatch();
    });
  } else {
    // Same reasoning: an existing panel is brought to the front of its group rather
    // than left behind whatever tab is on top of it.
    planPanel.reveal(undefined, false);
  }
  // Re-baseline the poller before the read, so a write landing in between is
  // caught on the next tick rather than lost.
  watchPreviewedFile(abs);
  void renderPlan();
}
