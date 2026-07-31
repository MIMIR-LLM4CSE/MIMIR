import React from "react";

/** Brand assets injected by the extension host (see getWebviewHtml). */
function logoUrl(): string | undefined {
  return (window as unknown as { __MIMIR_ASSETS__?: { logo?: string } })
    .__MIMIR_ASSETS__?.logo;
}

/**
 * MIMIR brand mark. When the extension provides the logo image (the norse-myth
 * "M" glyph), it is rendered as an <img>; sizing lives in CSS on `.mimir-mark`
 * (1em by default, so it scales with the parent's font-size — avatars at 16px,
 * the empty-state header at 36px — and call sites can scale it up where it sits
 * next to emoji). Falls back to an inline radiant-eye SVG (evoking Mímir's well
 * of wisdom) when the asset is unavailable (e.g. outside a webview).
 */
export function MimirMark({ className }: { className?: string }): JSX.Element {
  const src = logoUrl();
  if (src) {
    return (
      <img
        className={`mimir-mark ${className ?? ""}`.trim()}
        src={src}
        alt="MIMIR"
        aria-hidden="true"
      />
    );
  }
  return (
    <svg
      className={`mimir-mark ${className ?? ""}`.trim()}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      role="img"
    >
      {/* eye almond */}
      <path d="M2.5 12S6 6.2 12 6.2 21.5 12 21.5 12 18 17.8 12 17.8 2.5 12 2.5 12z" />
      {/* iris */}
      <circle cx="12" cy="12" r="2.9" />
      {/* pupil */}
      <circle cx="12" cy="12" r="0.7" fill="currentColor" stroke="none" />
      {/* radiant rays — divine halo */}
      <path d="M12 2.2v1.7M12 20.1v1.7M3.9 5.1l1.2 1.2M18.9 17.7l1.2 1.2M20.1 5.1l-1.2 1.2M5.1 17.7l-1.2 1.2" />
    </svg>
  );
}
