import React from "react";
import type { ApprovalMode } from "../types";
import { APPROVAL_OPTIONS } from "./ApprovalSwitcher";

interface Props {
  /** The autonomy the run will have while unattended — whatever is set right here. */
  mode: ApprovalMode;
  /** True once the server has reported it detached; the control stops being an action. */
  detached: boolean;
  onDetach: () => void;
}

/**
 * "Continue without me" — the moment the user leaves.
 *
 * Deliberately not a setting chosen when connecting. The question is whether anything
 * is worth leaving running, and that is only answerable once something is: at connect
 * time there is nothing to decide about.
 *
 * It carries no autonomy picker of its own. The level is whatever the approval switcher
 * immediately above it already says, which is the control that exists for exactly this
 * question and already explains what each level does not lift. A second picker here
 * would be two places to answer one question, and the two would disagree.
 *
 * The title spells out the consequence rather than naming the level, because at
 * `manual` a detached run parks at its first sensitive tool and does almost nothing
 * overnight — a user who has not been told that reads "detached" as "it will finish".
 */
export const DetachButton: React.FC<Props> = ({ mode, detached, onDetach }) => {
  if (detached) {
    return (
      <div
        className="detach-state"
        title="This server keeps working when the window closes. Reopening the workspace reattaches to it."
      >
        ⛓️‍💥 detached
      </div>
    );
  }

  const option = APPROVAL_OPTIONS.find((o) => o.value === mode);
  const consequence =
    mode === "manual"
      ? "it will park at the first call that needs you, and wait for your return"
      : `it may act on its own: ${option?.desc ?? mode}`;

  return (
    <button
      className="detach-btn"
      onClick={onDetach}
      title={`Keep working without me. At "${option?.label ?? mode}" ${consequence}.`}
      aria-label="Keep working without me"
    >
      ⛓️‍💥
    </button>
  );
};
