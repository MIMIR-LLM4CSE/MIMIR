import * as vscode from "vscode";

let channel: vscode.OutputChannel | undefined;

/** The one "MIMIR Server" output channel of this window, created on first use. */
export function serverChannel(): vscode.OutputChannel {
  return (channel ??= vscode.window.createOutputChannel("MIMIR Server"));
}

/** Dispose of the channel, when the extension is deactivated. */
export function disposeServerChannel(): void {
  channel?.dispose();
  channel = undefined;
}
