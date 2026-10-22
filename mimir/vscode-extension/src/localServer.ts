/**
 * The MIMIR server this window is responsible for: a child it spawned, or the pid of one
 * it attached to, and whether it has detached (then it is not this window's to end).
 */
import * as vscode from "vscode";
import * as cp from "child_process";
import * as fs from "fs";
import { serverChannel } from "./serverChannel";
import { findLiveServer, logPath, waitForServer } from "./serverRegistry";
import { mimirServerEnv, noProxyFor, resolvePython, settingsEnv } from "./serverEnv";
import { workspaceRoot } from "./workspace";

/** What a connect asks the server to run on. */
export interface SpawnRequest {
  model: string;
  backend: string;
  baseUrl: string;
  /** Reaches the server through its environment only, never its command line. */
  anthropicApiKey: string;
  /** A startup connect: the log channel is not brought to the front. */
  silent: boolean;
}

/** How a spawn ended: listening at *url*, given up on, or replaced while it started. */
export type SpawnOutcome =
  | { kind: "listening"; url: string }
  | { kind: "gave_up" }
  | { kind: "superseded" };

export class LocalServer {
  private child: cp.ChildProcess | undefined;
  /**
   * Set once the server reports it has detached. From then on it is not ours to end: it
   * re-pointed its output away from us and left our process group precisely so that
   * closing this window leaves it working.
   */
  private detached = false;
  /**
   * The pid of a server this window attached to rather than started (from this
   * workspace's registry entry), so leaving the window can end it. A server reached
   * through `mimir.wsUrl` is the user's own, and never recorded here.
   */
  private attachedPid: number | undefined;
  /** Stops the timer mirroring the server's log file into its output channel. */
  private logTail: vscode.Disposable | undefined;

  constructor(private readonly extensionRoot: string) {}

  /** A child this window spawned is still alive. */
  get childRunning(): boolean {
    return this.child !== undefined && !this.child.killed;
  }

  /** The server said it detached (or took that back). */
  noteDetached(detached: boolean): void {
    this.detached = detached;
  }

  /** This window is now responsible for the server at *pid*, which it did not spawn. */
  adopt(pid: number): void {
    this.attachedPid = pid;
  }

  /**
   * End the server this window is responsible for. SIGTERM, so the server's own handler
   * reaps the MCP servers. A detached server is exempt: it was kept going on purpose.
   */
  endUnlessKeptRunning(): void {
    if (!this.detached) {
      if (this.child && !this.child.killed) {
        this.child.kill();
      } else if (this.attachedPid !== undefined) {
        try {
          process.kill(this.attachedPid, "SIGTERM");
        } catch {
          // Already gone, or not ours to signal. Either way there is nothing to end,
          // and a window closing must not fail over it.
        }
      }
    }
    this.child = undefined;
    this.attachedPid = undefined;
  }

  /** Stop mirroring the log, and end the server unless it is kept running. */
  release(): void {
    this.logTail?.dispose();
    this.logTail = undefined;
    this.endUnlessKeptRunning();
  }

  /**
   * Spawn a server for this workspace and wait for the address it publishes.
   *
   * *onExit* runs when the child exits while still the current one.
   */
  async spawn(req: SpawnRequest, onExit: () => void): Promise<SpawnOutcome> {
    const pythonPath = resolvePython(this.extensionRoot);
    const cwd = workspaceRoot();

    const outputChannel = serverChannel();
    // A connect the user clicked shows its log; one we started at VS Code launch
    // stays out of the way (the channel is still there to open by hand).
    if (!req.silent) outputChannel.show();

    const { backend, baseUrl } = req;
    const backendArgs =
      backend === "vllm" ? ` --backend vllm --vllm-base-url ${baseUrl}`
      : backend === "ray" ? ` --backend ray --ray-base-url ${baseUrl}`
      : backend === "ollama" ? ` --backend ollama --ollama-base-url ${baseUrl}`
      : backend === "anthropic" ? " --backend anthropic"
      : "";
    // `--port 0`: the OS picks a free port, which the server publishes. `exec` makes
    // `kill()` reach python itself. Output goes to a file: a pipe held by this host
    // would SIGPIPE a server meant to outlive the window.
    const serverLog = logPath(cwd, new Date().toISOString().replace(/[:.]/g, "-"));
    const quotedLog = `'${serverLog.replace(/'/g, "'\\''")}'`;
    const spawnCmd =
      `mkdir -p -- "$(dirname -- ${quotedLog})" && ` +
      `exec ${pythonPath} -m mimir.client.ui.ws.server.main --port 0` +
      `${req.model ? ` --model ${req.model}` : ""}${backendArgs} >> ${quotedLog} 2>&1`;
    // The command only: the Claude API key goes through the environment below, so it
    // never lands in the output channel.
    outputChannel.appendLine(`Starting server: ${spawnCmd}`);

    // An HTTP proxy silently swallows requests to an on-prem endpoint, so
    // ws_server would hang on model resolution before ever binding its port.
    const noProxyEnv = backend === "anthropic" ? {} : noProxyFor(baseUrl);
    // Only override ANTHROPIC_API_KEY when the webview actually supplied one;
    // otherwise inherit whatever is already exported (so users who set the key in
    // their shell don't have to retype it in the form).
    const anthropicEnv =
      backend === "anthropic" && req.anthropicApiKey ? { ANTHROPIC_API_KEY: req.anthropicApiKey } : {};
    // Our own child supersedes any attached pid.
    this.attachedPid = undefined;
    const spawned = cp.spawn("bash", ["-c", spawnCmd], {
      cwd,
      // Anchor the agent's per-workspace state dir (.mimir) and the file-server
      // root to the opened workspace, regardless of the process cwd. The base
      // env is mimirServerEnv(): the user's PYTHONHOME/PYTHONPATH never reach
      // the server's interpreter (see there).
      env: {
        ...mimirServerEnv(), MCP_FILES_ROOT: cwd, ...noProxyEnv, ...settingsEnv(),
        ...anthropicEnv,
      },
      // Its own process session, from birth. Nothing about this window's teardown can
      // then reach it implicitly — not a process-group signal, not a SIGHUP, not the
      // editor tidying up its tree — so the only thing that ends it is the explicit
      // SIGTERM of `endUnlessKeptRunning`, skipped once it has detached. Arranging this
      // after the fact, by having the server leave the group itself, depends on it
      // not already being a group leader, and a detachment that quietly does not hold
      // is worse than none.
      detached: true,
      stdio: "ignore",
    });
    this.child = spawned;
    // So this window's exit is not waited on by Node for a process it no longer owns.
    spawned.unref();

    outputChannel.appendLine(`Output: ${serverLog}`);
    this.tailLog(outputChannel, serverLog);

    spawned.on("exit", (code) => {
      if (this.child !== spawned) return;
      outputChannel.appendLine(`\nServer exited (code ${code}) — see ${serverLog}`);
      this.child = undefined;
      onExit();
    });

    // The address is not knowable in advance (the OS assigns the port) and there is no
    // pipe to read it from, so it is taken from the registry the server publishes once
    // its socket is bound. Everything before that — MCP handshakes, model resolution —
    // can take a while on a cold start.
    const entry = await waitForServer(cwd, spawned.pid);
    if (this.child !== spawned && entry === undefined) return { kind: "superseded" };
    if (!entry) {
      outputChannel.appendLine(
        "\nServer did not publish a listening address within 120s — giving up. " +
        `Check the endpoint above is reachable, then connect again. Its log: ${serverLog}`
      );
      return { kind: "gave_up" };
    }
    if (entry.pid !== this.child?.pid) {
      // Ours stood down: this workspace already had a server, and starting a second
      // would have both write the same session journal. We are responsible for the one
      // that won, so its pid takes the place of the child we no longer have.
      outputChannel.appendLine(
        `\nThis workspace already had a server (pid ${entry.pid}); attached to it.`);
      this.child = undefined;
      this.attachedPid = entry.pid;
    }
    return { kind: "listening", url: entry.url };
  }

  /**
   * The `mimir.startServer` command: a server on the fixed port 8765, to point
   * `mimir.wsUrl` at (an OS-assigned port could not be written down in a setting). If
   * 8765 is taken, the bind error shows up in the output channel.
   */
  startOnFixedPort(context: vscode.ExtensionContext): void {
    const pythonPath = resolvePython(this.extensionRoot);

    if (this.childRunning) {
      vscode.window.showInformationMessage("MIMIR WS server is already running.");
      return;
    }

    const outputChannel = serverChannel();
    outputChannel.show();

    const cwd = workspaceRoot();
    // `exec` so the process we track — and later kill — is the server, not the shell.
    const spawnCmd = `exec ${pythonPath} -m mimir.client.ui.ws.server.main --port 8765`;

    // A child of our own from here on: whatever pid we were attached to before is not
    // the server this window is responsible for any more.
    this.attachedPid = undefined;
    const child = cp.spawn("bash", ["-c", spawnCmd], {
      cwd,
      // Anchor the agent's per-workspace state dir (.mimir) and the file-server
      // root to the opened workspace, regardless of the process cwd. Same
      // python-env isolation and settings as the connect path (see mimirServerEnv).
      env: { ...mimirServerEnv(), MCP_FILES_ROOT: cwd, ...settingsEnv() },
      stdio: ["ignore", "pipe", "pipe"],
    });
    this.child = child;

    child.stdout?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
    child.stderr?.on("data", (d: Buffer) => outputChannel.append(d.toString()));
    child.on("exit", (code) => {
      outputChannel.appendLine(`\nServer exited (code ${code})`);
      this.child = undefined;
    });

    vscode.window.showInformationMessage("MIMIR WS server started.");
    // Not left running on window close: ended through the one path out, like every
    // other departure.
    context.subscriptions.push({ dispose: () => this.endUnlessKeptRunning() });
  }

  /**
   * The `mimir.stopServer` command: stop the server this workspace has, detached or
   * not, after confirming with what is running named. Signalled, since the window
   * asking may not be connected to it; SIGTERM, so its exit reaps the MCP servers.
   */
  async stopFromCommand(): Promise<void> {
    const entry = await findLiveServer(workspaceRoot());
    if (!entry) {
      vscode.window.showInformationMessage(
        "No MIMIR server is running for this workspace."
      );
      return;
    }
    const what = entry.detached
      ? `the detached MIMIR server (pid ${entry.pid})`
      : `the MIMIR server (pid ${entry.pid})`;
    const answer = await vscode.window.showWarningMessage(
      `Stop ${what}? Any turn still running, and any background job it is waiting on, ` +
      `ends with it.`,
      { modal: true },
      "Stop server"
    );
    if (answer !== "Stop server") return;
    try {
      process.kill(entry.pid, "SIGTERM");
    } catch (err) {
      vscode.window.showErrorMessage(
        `Could not stop pid ${entry.pid}: ${(err as Error).message}`
      );
      return;
    }
    this.detached = false;
    this.child = undefined;
    this.attachedPid = undefined;
    vscode.window.showInformationMessage(`Asked pid ${entry.pid} to stop.`);
  }

  /**
   * Mirror a server's log file into its output channel.
   *
   * A detached server writes to a file rather than a pipe, so the channel reads that
   * file. Polled on a timer: the file is appended to by a process in another session,
   * and a watcher adds nothing a cheap stat does not already give.
   */
  private tailLog(channel: vscode.OutputChannel, file: string): void {
    this.logTail?.dispose();
    let offset = 0;
    const timer = setInterval(() => {
      try {
        const size = fs.statSync(file).size;
        if (size < offset) offset = 0;      // truncated or replaced
        if (size === offset) return;
        const fd = fs.openSync(file, "r");
        try {
          const buf = Buffer.alloc(size - offset);
          fs.readSync(fd, buf, 0, buf.length, offset);
          offset = size;
          channel.append(buf.toString());
        } finally {
          fs.closeSync(fd);
        }
      } catch {
        // Not written yet, or gone. Either way there is nothing to mirror.
      }
    }, 500);
    this.logTail = { dispose: () => clearInterval(timer) };
  }
}
