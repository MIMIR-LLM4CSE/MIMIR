/**
 * Native VS Code notifications for milestones (a task finished, a card waiting on the
 * user, a background run ending), shown only while the chat is not in front of the
 * user: the window unfocused, or the chat view hidden.
 */
import * as vscode from "vscode";

interface ServerEvent {
  type?: string; text?: string; cancelled?: boolean; summary?: string;
  job_key?: string; state?: string; id?: string;
  session_id?: string; session_title?: string;
}

export class Notifier {
  /** Last notification, so a burst of progress does not become a stack of toasts. */
  private lastNotifyAt = 0;
  /**
   * Cards already announced, so two conversations asking together get a toast each and
   * neither is announced twice. Keyed by card id, which is a uuid.
   */
  private notifiedPrompts = new Set<string>();

  constructor(
    private readonly chatInView: () => boolean,
    /** Open a conversation from the notification's own action. */
    private readonly openConversation: (sessionId: string) => void,
  ) {}

  /** Notify about the server frame *payload*, when it is a milestone and the chat is away. */
  notify(payload: string): void {
    const cfg = vscode.workspace.getConfiguration("mimir");
    if (!(cfg.get<boolean>("notifications.enabled") ?? true)) {
      return;
    }

    let msg: ServerEvent;
    try {
      msg = JSON.parse(payload);
    } catch {
      return;
    }
    if (!msg || typeof msg.type !== "string") {
      return;
    }

    if (this.chatInView()) {
      return;
    }

    let title: string | undefined;
    let kind: "info" | "warn" = "info";
    switch (msg.type) {
      case "answer":
        // Final answer to the user = task completed (ignore cancelled turns).
        if (msg.cancelled) { return; }
        title = "MIMIR a terminé la tâche.";
        kind = "info";
        break;
      // A card names the conversation that raised it: several conversations run at once,
      // and one may be asking while the user reads another — or while the pane is closed,
      // which is the case this notification exists for.
      case "approval":
        title = msg.session_title
          ? `« ${msg.session_title} » attend votre approbation.`
          : "MIMIR attend votre approbation.";
        kind = "warn";
        break;
      case "user_question":
        title = msg.session_title
          ? `« ${msg.session_title} » a une question pour vous.`
          : "MIMIR a une question pour vous.";
        kind = "warn";
        break;
      case "job_complete":
        // A detached background run reached a terminal state; the session that
        // launched it auto-resumes, which is not necessarily the one on screen.
        if (msg.state === "crashed") {
          title = `Tâche en arrière-plan « ${msg.job_key ?? ""} » terminée en échec.`;
          kind = "warn";
        } else if (msg.state === "unknown") {
          // Not a success reported quietly: the run stopped being trackable, and
          // saying "terminée" here would claim an outcome nobody observed.
          title = `Tâche en arrière-plan « ${msg.job_key ?? ""} » : suivi perdu.`;
          kind = "warn";
        } else {
          title = `Tâche en arrière-plan « ${msg.job_key ?? ""} » terminée.`;
          kind = "info";
        }
        break;
      case "job_checkin":
        // A periodic bulletin, not an ending: the runs are still going. Deliberately
        // quiet — the point of a check-in is that nothing has happened yet, and a
        // notification every time would train the user to ignore the ones that matter.
        return;
      default:
        return;
    }

    // A card is announced once per card, not once per 2s window. The window stops a burst
    // of progress turning into a burst of toasts; applied to cards it would swallow the
    // second of two conversations asking together, and that one waits for ever with
    // nothing said about it.
    const isPrompt = msg.type === "approval" || msg.type === "user_question";
    if (isPrompt) {
      const key = `${msg.type}:${msg.id ?? ""}:${msg.session_id ?? ""}`;
      if (this.notifiedPrompts.has(key)) {
        return;
      }
      this.notifiedPrompts.add(key);
      // Bounded: the ids are uuids and never recur, so nothing ever leaves on its own.
      if (this.notifiedPrompts.size > 200) {
        this.notifiedPrompts.clear();
      }
    } else {
      const now = Date.now();
      if (now - this.lastNotifyAt < 2000) {
        return;
      }
      this.lastNotifyAt = now;
    }

    // Two ways in, because a card raised elsewhere needs more than the pane: focusing
    // the chat shows whichever conversation was last open, which may not be the one
    // asking.
    const open = "Ouvrir le chat";
    const goThere = "Ouvrir la conversation";
    const actions = isPrompt && msg.session_id ? [goThere, open] : [open];
    const show = kind === "warn"
      ? vscode.window.showWarningMessage
      : vscode.window.showInformationMessage;
    show(title, ...actions).then((choice) => {
      if (choice === goThere && msg.session_id) {
        this.openConversation(msg.session_id);
      }
      if (choice === goThere || choice === open) {
        vscode.commands.executeCommand("mimir.chatView.focus");
      }
    });
  }
}
