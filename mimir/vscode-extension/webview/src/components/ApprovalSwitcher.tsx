import React, { useEffect, useRef, useState } from "react";
import type { ApprovalMode } from "../types";

interface Props {
  mode: ApprovalMode;
  onModeChange: (mode: ApprovalMode) => void;
}

// Mirrors ApprovalManager.APPROVAL_MODES in client/guardrails/policy/approval.py.
// Each option says what stops asking, because that is the whole decision — and the
// descriptions say what an auto mode does *not* lift: the shell denylist, the
// servers' sandbox and the verification guards refuse in every mode, so "auto" must
// never read as "unguarded".
export const APPROVAL_OPTIONS: {
  value: ApprovalMode;
  icon: string;
  label: string;
  desc: string;
}[] = [
  {
    value: "manual",
    icon: "🔒",
    label: "manual",
    desc: "Every sensitive call, and every step outside the workspace, comes back to you",
  },
  {
    value: "auto",
    icon: "⚡",
    label: "auto",
    desc: "Sensitive tools run unasked — leaving the workspace still asks",
  },
  {
    value: "auto_all",
    icon: "⚡⚡",
    label: "all",
    desc: "Nothing asks. Guardrails (denylist, sandbox) still apply",
  },
];

/** Approval-mode picker, stacked directly above the send button.
 *
 *  Its own control rather than a section of the settings popover: an auto mode
 *  answers for the user, and it is switched mid-run — both of which argue for a
 *  state visible without opening anything, at the spot where the user is already
 *  looking when they send work off. Icon-only, because the column it shares with
 *  the send button is width the message box would otherwise have; the icon and its
 *  colour carry the state, and the tooltip and the picker rows carry the words.
 */
export const ApprovalSwitcher: React.FC<Props> = ({ mode, onModeChange }) => {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const active = APPROVAL_OPTIONS.find((o) => o.value === mode) ?? APPROVAL_OPTIONS[0];

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="approval-wrapper" ref={wrapRef}>
      <button
        className={`settings-btn approval-btn${mode !== "manual" ? " approval-btn--auto" : ""}`}
        title={`Approvals: ${active.label} — ${active.desc}`}
        aria-label={`Approval mode: ${active.label}`}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="approval-btn-icon">{active.icon}</span>
      </button>
      {open && (
        <div className="settings-popover approval-popover">
          <div className="settings-section-label">Approvals</div>
          <div className="settings-mode-list">
            {APPROVAL_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                data-approval={opt.value}
                className={`settings-option-btn settings-mode-btn ${mode === opt.value ? "active" : ""}`}
                onClick={() => {
                  onModeChange(opt.value);
                  setOpen(false);
                }}
              >
                <span className="settings-mode-btn-label">
                  {opt.icon} {opt.label}
                </span>
                <span className="settings-mode-btn-desc">{opt.desc}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
