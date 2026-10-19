/** The socket to the MIMIR server: connecting with retries, and what is sent before it opens. */
import * as vscode from "vscode";
import WebSocket = require("ws");

export interface ConnectionHooks {
  /** A message for the webview. */
  post(message: Record<string, unknown>): void;
  /** Every frame the server sends, after it is forwarded to the webview. */
  onServerText(text: string): void;
  /** True while a server this window spawned may still be coming up. */
  serverStarting(): boolean;
}

export class ServerConnection {
  private ws: WebSocket | undefined;
  private pending: string[] = [];
  /**
   * Bumped by every new connect attempt. A retry chain carries the generation it was
   * born with, so the one a superseded attempt left running goes inert instead of
   * reclaiming the socket and then reporting the connection closed.
   */
  private gen = 0;
  /**
   * Address of the server this window talks to, learned from the registry the server
   * publishes (it binds an OS-assigned port, so each window gets its own) or copied
   * from the `mimir.wsUrl` override. Every reconnect goes through this.
   */
  private url: string | undefined;
  /**
   * Log channel of an attach (to `mimir.wsUrl`, or to a server the workspace already
   * had), or undefined when this window spawned the server. Kept so a failed attach can
   * name the setting responsible.
   */
  attachLog: vscode.OutputChannel | undefined;

  constructor(private readonly hooks: ConnectionHooks) {}

  /** A socket exists, open or still connecting. */
  get active(): boolean {
    return this.ws !== undefined;
  }

  /** Connect to *url*, superseding any attempt still retrying. */
  connect(url: string): void {
    this.url = url;
    this.attempt(url, 0);
  }

  /**
   * Replace a live socket with a fresh one to the same address, so the server greets
   * a newly mounted webview with `ready`. The old socket's close drives nothing.
   */
  reopen(): void {
    const live = this.ws;
    if (!live || !this.url) return;
    this.ws = undefined;
    live.close();
    this.attempt(this.url, 0);
  }

  /** Send *payload* now, or once the socket opens. */
  send(payload: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(payload);
    } else {
      this.pending.push(payload);
    }
  }

  /** Close the socket. */
  close(): void {
    this.ws?.close();
    this.ws = undefined;
  }

  /**
   * Close the socket and stop its retry chain: the process it belongs to is being
   * released, and a chain outliving it would hunt a port nobody serves.
   */
  retire(): void {
    this.gen++;
    this.close();
  }

  /** Open the socket, retrying while the server is still coming up. */
  private attempt(url: string, retryCount: number, gen?: number): void {
    const maxRetries = 40; // retry for up to ~40 seconds while server starts
    const myGen = gen ?? ++this.gen;
    if (myGen !== this.gen) return; // superseded while this retry waited

    const ws = new WebSocket(url);
    this.ws = ws;

    ws.on("open", () => {
      if (myGen !== this.gen) {
        ws.close();
        return;
      }
      // Flush any messages queued before the connection was ready
      for (const m of this.pending) {
        ws.send(m);
      }
      this.pending = [];
    });

    ws.on("message", (data: WebSocket.RawData) => {
      if (myGen !== this.gen) return;
      const text = data.toString();
      this.hooks.post({ type: "ws", payload: text });
      this.hooks.onServerText(text);
    });

    ws.on("close", () => {
      // A socket replaced on purpose (`reopen`) must not drive the retry/teardown of
      // the one that took its place.
      if (myGen !== this.gen || this.ws !== ws) return;
      if (retryCount < maxRetries && this.hooks.serverStarting()) {
        // Server still starting up — retry after 2 seconds
        setTimeout(() => this.attempt(url, retryCount + 1, myGen), 2000);
      } else {
        // In attach mode there is no server log to consult, so the reason the
        // connection never came up would otherwise go unrecorded entirely.
        this.attachLog?.appendLine(
          `Nothing answered at ${url}. That address comes from the "mimir.wsUrl" ` +
          `setting; clear it (check the workspace's .vscode/settings.json) to have ` +
          `MIMIR start its own server instead.`
        );
        // No auto-reconnect: the user connects again.
        this.hooks.post({ type: "ws_closed" });
      }
    });

    ws.on("error", () => {
      if (myGen !== this.gen) return;
      // close fires right after error and handles reconnect/teardown; this is the
      // distinct signal the webview shows an error state on.
      this.hooks.post({ type: "ws_error" });
    });
  }
}
