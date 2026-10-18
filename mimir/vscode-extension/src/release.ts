/**
 * Shared-release helpers: a site deployment publishes immutable `releases/<version>`
 * trees behind a `current` symlink, activated by a `site.json` next to the bundle.
 * Dependency-free pieces only; the vscode/fs glue is in extension.ts.
 */

/** Site configuration a deployment may ship as site.json beside the extension. */
export interface SiteConfig {
  /** Shared release root — the tree whose `current` exposes the running release. */
  releaseHome?: string;
}

/**
 * Read `<extensionRoot>/site.json` when present.
 *
 * Returns the parsed configuration, or null when the file is absent or
 * malformed — the common case for a distribution without a shared release.
 * `readFileSync` is injected so this stays testable without an fs backend; the
 * extension host passes `fs.readFileSync`.
 */
export function loadSiteConfig(
  extensionRoot: string,
  readFileSync: (path: string) => string = () => ""
): SiteConfig | null {
  try {
    const raw = readFileSync(`${extensionRoot}/site.json`);
    if (!raw) return null;
    const cfg = JSON.parse(raw) as SiteConfig;
    return cfg && typeof cfg === "object" ? cfg : null;
  } catch {
    return null;
  }
}

/**
 * The release root to probe: a non-empty `mimir.releaseHome` setting, else site.json.
 *
 * Empty and unset are the same (VS Code returns `""` for an untouched setting). A path
 * without a `current` release disables the mode; the caller's existence probe decides.
 */
export function resolveReleaseRoot(
  configured: string | undefined,
  fromSite: string | undefined
): string {
  const explicit = (configured ?? "").trim();
  return (explicit || fromSite || "").replace(/\/+$/, "");
}

/**
 * Compare two dotted numeric versions ("1.0.0" vs "1.2.0"), left to right.
 * Returns a negative number when `a < b`, 0 when equal, positive when `a > b`.
 * A leading "v" is ignored; a missing segment counts as 0; a non-numeric
 * segment (e.g. a pre-release tag) counts as -1, so it sorts below the numeric
 * segment at the same position.
 */
export function compareVersions(a: string, b: string): number {
  const seg = (v: string): number => (v === "" ? 0 : /^\d+$/.test(v) ? parseInt(v, 10) : -1);
  const pa = (a || "").replace(/^v/, "").split(".").map(seg);
  const pb = (b || "").replace(/^v/, "").split(".").map(seg);
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i++) {
    const x = i < pa.length ? pa[i] : 0;
    const y = i < pb.length ? pb[i] : 0;
    if (x !== y) return x < y ? -1 : 1;
  }
  return 0;
}
