import React from "react";
import type { ApprovalMode } from "../types";
import { APPROVAL_OPTIONS } from "./ApprovalSwitcher";
import { DetachIcon } from "./icons";

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
 * "Continue without me": a toggle of who owns the process, revocable until the window
 * closes. The socket stays open.
 *
 * It takes the level of the approval switcher above it, and its title says the
 * consequence: at `manual` a detached run parks at its first sensitive tool.
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
        <DetachIcon />
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
      <DetachIcon />
    </button>
  );
};
