import * as vscode from "vscode";
import * as path from "path";

/** The first workspace folder, or the host's working directory without one. */
export function workspaceRoot(): string {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();
}

/** *rel* resolved against the first workspace folder (an absolute path is kept). */
export function workspacePath(rel: string): string {
  return path.isAbsolute(rel) ? rel : path.join(workspaceRoot(), rel);
}
