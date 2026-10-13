import React from "react";
import type { ApprovalMode } from "../types";
import { APPROVAL_OPTIONS } from "./ApprovalSwitcher";

interface Props {
  /** The autonomy the run will have while unattended — whatever is set right here. */
  mode: ApprovalMode;
  /** True once the server has reported it detached. */
  detached: boolean;
  onDetach: () => void;
  /** Take the decision back: this window owns the server again. */
  onReattach: () => void;
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
 *
 * Detaching does not disconnect: the socket stays open and the turn goes on in front of
 * the user. So the state is a toggle rather than a destination — what it changed is who
 * owns the process, and that is a decision, revocable until the window actually closes.
 */
export const DetachButton: React.FC<Props> = ({
  mode, detached, onDetach, onReattach,
}) => {
  if (detached) {
    return (
      <button
        className="detach-btn detached"
        onClick={onReattach}
        title="MIMIR keeps working when this window closes, and reopening the workspace comes back to it. Click to take that back: the window owns the server again and closing it stops it."
        aria-label="Stop keeping the server alive without this window"
      >
        ⛓️‍💥
      </button>
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
