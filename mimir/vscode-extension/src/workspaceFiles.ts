/**
 * The workspace's files for the "@" autocomplete: scanned once per window, dropped when
 * a file is created or deleted, ranked per query by fileIndex.
 */
import * as vscode from "vscode";
import { isNoisePath, rankFiles } from "./fileIndex";

export class WorkspaceFiles {
  /**
   * The workspace's file list, as a promise so concurrent keystrokes share one scan.
   * Undefined means "not scanned, or invalidated by a file appearing or disappearing".
   */
  private scan: Promise<string[]> | undefined;
  /** Watcher behind that invalidation; created with the first scan, never replaced. */
  private watcher: vscode.FileSystemWatcher | undefined;

  /**
   * The files matching *query*, best first. Empty when there is no workspace or the
   * host refused the scan: the dropdown still has the MCP resources, and a path typed
   * by hand resolves server-side either way.
   */
  async matching(query: string): Promise<string[]> {
    let files: string[];
    try {
      files = await this.files();
    } catch {
      files = [];
    }
    return rankFiles(files, query);
  }

  /** The cached workspace-relative file list, scanning on the first ask. */
  private files(): Promise<string[]> {
    if (!this.scan) {
      this.watch();
      // Capped: past a few tens of thousands of files the ranking is the wrong tool
      // and the user is better served by typing more of the path. `undefined` as the
      // exclude keeps the workspace's own `files.exclude`; `isNoisePath` adds the
      // trees every repo has and nobody attaches.
      this.scan = Promise.resolve(
        vscode.workspace.findFiles("**/*", undefined, 40000),
      ).then((uris) =>
        uris
          .map((uri) => vscode.workspace.asRelativePath(uri, false))
          .filter((rel) => !isNoisePath(rel)),
      );
    }
    return this.scan;
  }

  /**
   * Drop the cached list when the workspace gains or loses a file. Content changes are
   * ignored (the index holds paths), and the watcher is created with the first scan, so
   * a window whose user never types "@" installs nothing.
   */
  private watch(): void {
    if (this.watcher) return;
    const watcher = vscode.workspace.createFileSystemWatcher("**/*", false, true, false);
    const invalidate = () => { this.scan = undefined; };
    watcher.onDidCreate(invalidate);
    watcher.onDidDelete(invalidate);
    this.watcher = watcher;
  }
}
