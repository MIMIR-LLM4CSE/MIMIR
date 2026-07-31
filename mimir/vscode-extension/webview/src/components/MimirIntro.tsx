import React, { useEffect, useRef } from "react";
import { MimirMark } from "./MimirMark";

/** Brand assets injected by the extension host (see getWebviewHtml). */
function introVideoUrl(): string | undefined {
  return (window as unknown as { __MIMIR_ASSETS__?: { introVideo?: string } })
    .__MIMIR_ASSETS__?.introVideo;
}

/**
 * MIMIR brand mark shown on the connection screen.
 *
 * - `loop` (while connecting): plays the intro clip (norse-myth "M" logo)
 *   continuously as a loading animation.
 * - otherwise (idle / disconnected / connected): shows the static transparent
 *   logo, no video.
 *
 * Falls back to the static mark when the video asset is unavailable (e.g.
 * outside a webview).
 */
export function MimirIntro({ loop = false }: { loop?: boolean }): JSX.Element {
  const src = introVideoUrl();
  const videoRef = useRef<HTMLVideoElement>(null);

  // React's `muted` JSX attribute is unreliable (it doesn't always set the DOM
  // property), which makes Chromium block muted-autoplay. Set it imperatively
  // and kick off playback once the element is mounted.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = true;
    v.defaultMuted = true;
    const p = v.play();
    if (p && typeof p.catch === "function") {
      // Ignore autoplay rejections — the static mark fallback still shows.
      p.catch(() => undefined);
    }
  }, [src, loop]);

  if (!src || !loop) {
    return <div className="empty-icon mimir-intro-static"><MimirMark /></div>;
  }
  return (
    <video
      ref={videoRef}
      className="mimir-intro-video"
      src={src}
      autoPlay
      muted
      playsInline
      preload="auto"
      loop
      aria-hidden="true"
    />
  );
}


