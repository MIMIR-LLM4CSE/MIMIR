import React from "react";

/** Monochrome line icons for the composer's controls: they paint in currentColor, so
 *  they follow the theme and the mode colour where a colour emoji would not. */
const Svg: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <svg
    className="ui-icon"
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.4}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    focusable="false"
  >
    {children}
  </svg>
);

export const GearIcon = () => (
  <Svg>
    <circle cx="8" cy="8" r="2.2" />
    <path d="M8 1.8v1.6M8 12.6v1.6M14.2 8h-1.6M3.4 8H1.8M12.4 3.6l-1.1 1.1M4.7 11.3l-1.1 1.1M12.4 12.4l-1.1-1.1M4.7 4.7L3.6 3.6" />
    <circle cx="8" cy="8" r="4.6" />
  </Svg>
);

export const BlocksIcon = () => (
  <Svg>
    <rect x="2" y="2" width="5" height="5" rx="1" />
    <rect x="9" y="2" width="5" height="5" rx="1" />
    <rect x="2" y="9" width="5" height="5" rx="1" />
    <path d="M11.5 9v5M9 11.5h5" />
  </Svg>
);

export const PlanIcon = () => (
  <Svg>
    <path d="M6 4h8M6 8h8M6 12h8" />
    <path d="M2.5 4h.01M2.5 8h.01M2.5 12h.01" strokeWidth={2} />
  </Svg>
);

export const AskIcon = () => (
  <Svg>
    <path d="M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2z" />
  </Svg>
);

export const LockIcon = () => (
  <Svg>
    <rect x="3" y="7" width="10" height="7" rx="1.5" />
    <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
  </Svg>
);

export const BoltIcon = () => (
  <Svg>
    <path d="M9 1.5L3.5 9H8l-1 5.5L12.5 7H8z" />
  </Svg>
);

export const BoltCircleIcon = () => (
  <Svg>
    <circle cx="8" cy="8" r="6.3" />
    <path d="M8.8 3.8L5.5 8.6h2.6l-.9 3.6 3.3-4.8H7.9z" />
  </Svg>
);

export const SendIcon = () => (
  <Svg>
    <path d="M8 13V3M3.5 7.5L8 3l4.5 4.5" />
  </Svg>
);

export const StopIcon = () => (
  <Svg>
    <rect x="4" y="4" width="8" height="8" rx="1.5" fill="currentColor" />
  </Svg>
);

export const DetachIcon = () => (
  <Svg>
    <path d="M6.5 9.5l-2 2a2.1 2.1 0 0 1-3-3l2-2M9.5 6.5l2-2a2.1 2.1 0 0 1 3 3l-2 2" />
    <path d="M10 13.5v1M13.5 10h1M6 2.5v-1M2.5 6h-1" />
  </Svg>
);
