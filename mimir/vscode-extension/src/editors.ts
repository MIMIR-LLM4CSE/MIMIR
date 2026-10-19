/**
 * What the webview asks to open in an editor: a file at a line range, a plan in its
 * preview, a diff of an edit, a patch. Proposed and reconstructed contents are served
 * as virtual `mimir-diff:` documents.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { reversePatch } from "./patch";
import { showPlanPreview } from "./planPreview";
import type { HostMessageOf } from "./shared/hostMessages";
import { workspacePath } from "./workspace";

/** The contents of the `mimir-diff:` documents, keyed by URI path. */
const virtualContent = new Map<string, string>();

/** Serves the `mimir-diff:` documents; registered once at activation. */
export const diffContentProvider = new class implements vscode.TextDocumentContentProvider {
  provideTextDocumentContent(uri: vscode.Uri): string {
    return virtualContent.get(uri.path) ?? "";
  }
}();

export function openFile(m: HostMessageOf<"open_file">): void {
  if (!m.file) return;
  const uri = vscode.Uri.file(workspacePath(m.file));
  const line = typeof m.line === "number" ? m.line : undefined;
  const endLine = typeof m.end_line === "number" ? m.end_line : line;
  vscode.workspace.openTextDocument(uri).then(
    (doc) => {
      if (line === undefined) {
        void vscode.window.showTextDocument(doc, { preview: true });
        return;
      }
      // Clamped: the file may have changed since the tool touched it.
      const last = doc.lineCount - 1;
      const start = Math.min(Math.max(line - 1, 0), last);
      const end = Math.min(Math.max((endLine ?? line) - 1, start), last);
      const range = new vscode.Range(start, 0, end, doc.lineAt(end).text.length);
      void vscode.window.showTextDocument(doc, {
        preview: true,
        viewColumn: vscode.ViewColumn.One,
        selection: range,
      }).then((editor) => {
        editor.revealRange(range, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
      });
    },
    () => vscode.window.showWarningMessage(`MIMIR: cannot open ${m.file}`)
  );
}

/** A file the agent wrote (a plan .md) opened for reading: Markdown in its preview. */
export function openPreview(m: HostMessageOf<"open_preview">): void {
  if (!m.file) return;
  const abs = workspacePath(m.file);
  if (/\.mdx?$/i.test(abs)) {
    showPlanPreview(abs);
    return;
  }
  vscode.workspace.openTextDocument(vscode.Uri.file(abs)).then(
    (doc) => vscode.window.showTextDocument(doc, { preview: true }),
    () => vscode.window.showWarningMessage(`MIMIR: cannot open ${m.file}`)
  );
}

/** A pending edit (current ↔ proposed), a written one (before ↔ after), else the file. */
export function openDiff(m: HostMessageOf<"open_diff">): void {
  const rel = m.file;
  if (!rel) return;
  const abs = workspacePath(rel);
  const basename = path.basename(rel);
  const currentUri = vscode.Uri.file(abs);
  if (m.new_content !== undefined) {
    virtualContent.set(abs, m.new_content);
    const proposedUri = vscode.Uri.from({ scheme: "mimir-diff", path: abs });
    if (fs.existsSync(abs)) {
      vscode.commands.executeCommand("vscode.diff", currentUri, proposedUri,
        `${basename}: Current ↔ Proposed`, { preview: true });
    } else {
      // A new file: the proposed content alone, not an empty-vs-new diff.
      vscode.workspace.openTextDocument(proposedUri).then(
        (doc) => vscode.window.showTextDocument(doc, { preview: true }),
        () => vscode.window.showWarningMessage(`MIMIR: cannot preview new file ${rel}`)
      );
    }
    return;
  }
  if (m.patch) {
    // The "before" content is the current file with the patch reversed.
    let currentContent = "";
    try { currentContent = fs.readFileSync(abs, "utf8"); } catch { /* new file */ }
    virtualContent.set(abs + "__original", reversePatch(currentContent, m.patch));
    const originalUri = vscode.Uri.from({ scheme: "mimir-diff", path: abs + "__original" });
    vscode.commands.executeCommand("vscode.diff", originalUri, currentUri,
      `${basename}: Before ↔ After`, { preview: true });
    return;
  }
  vscode.workspace.openTextDocument(currentUri).then(
    (doc) => vscode.window.showTextDocument(doc, { preview: true }),
    () => vscode.window.showWarningMessage(`MIMIR: cannot open ${rel}`)
  );
}

/** A patch as one highlighted .diff document (the .diff key picks the language). */
export function openPatch(m: HostMessageOf<"open_patch">): void {
  if (!m.file || !m.patch) return;
  const patchKey = workspacePath(m.file) + ".diff";
  virtualContent.set(patchKey, m.patch);
  const patchUri = vscode.Uri.from({ scheme: "mimir-diff", path: patchKey });
  vscode.workspace.openTextDocument(patchUri).then(
    (doc) => vscode.window.showTextDocument(doc, { preview: true }),
    () => vscode.window.showWarningMessage(`MIMIR: cannot open patch for ${m.file}`)
  );
}
