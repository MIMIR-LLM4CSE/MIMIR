/**
 * Asking an endpoint what models it serves, for the connect form. Runs in the host
 * because the webview's CSP forbids HTTP. A failure is reported as text under the
 * address field, never as a modal: the user can still connect (the server resolves
 * the served model itself).
 */
import * as vscode from "vscode";
import { ADDRESSED_BACKENDS } from "./endpoints";
import { explainFetchError, fetchModels, type DiscoverableBackend } from "./modelList";

export class ModelProbe {
  /**
   * Which probe the webview is still waiting on. A probe waits up to 30 s, and the form
   * fires one per address the user pauses on: a slow reply for an address since retyped
   * would otherwise land last and replace a good list with its own failure. Only the
   * newest probe may post.
   */
  private seq = 0;
  /**
   * Log channel for the results, created on first use; its own, apart from the server's,
   * since the connect form probes before any server is started.
   */
  private log: vscode.OutputChannel | undefined;

  constructor(private readonly post: (message: Record<string, unknown>) => void) {}

  /** Ask *baseUrl* for its models and post them (or why not) to the webview. */
  async probe(backend: string, baseUrl: string): Promise<void> {
    if (!ADDRESSED_BACKENDS.includes(backend)) {
      return;
    }
    const verifySsl = vscode.workspace.getConfiguration("mimir").get<boolean>("vllmVerifySsl", true);
    const seq = ++this.seq;
    const current = () => seq === this.seq;
    try {
      const models = await fetchModels(backend as DiscoverableBackend, baseUrl, verifySsl);
      this.channel().appendLine(
        `Model list from ${baseUrl}: ${models.length ? models.join(", ") : "(none named)"}`,
      );
      if (!current()) return;
      this.post({ type: "models", backend, models });
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      // Also to the output channel: the panel shows one line under the address,
      // which the user may have moved past by the time they come looking for why
      // the list was empty.
      this.channel().appendLine(`Model list from ${baseUrl} failed: ${reason}`);
      if (!current()) return;
      // The panel gets the plain-words version; the raw reason stays in the log.
      this.post({
        type: "models", backend, models: [], error: explainFetchError(err, baseUrl),
      });
    }
  }

  private channel(): vscode.OutputChannel {
    return (this.log ??= vscode.window.createOutputChannel("MIMIR Model Discovery"));
  }
}
