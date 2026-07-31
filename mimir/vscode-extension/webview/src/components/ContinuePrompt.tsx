import React from "react";

interface Props {
  summary: string;
  onChoice: (cont: boolean) => void;
}

export const ContinuePrompt: React.FC<Props> = ({ summary, onChoice }) => {
  return (
    <div className="resume-overlay">
      <div className="resume-modal">
        <div className="resume-title">⏸ Step budget reached</div>
        <div className="resume-subtitle">
          The agent has used its step budget for this run. Continue working?
        </div>
        {summary && <div className="resume-list">{summary}</div>}
        <div className="resume-actions">
          <button className="resume-btn resume-btn--yes" onClick={() => onChoice(true)}>
            Continue
          </button>
          <button className="resume-btn resume-btn--no" onClick={() => onChoice(false)}>
            Stop
          </button>
        </div>
      </div>
    </div>
  );
};
