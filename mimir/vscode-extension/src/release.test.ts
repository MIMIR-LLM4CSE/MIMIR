import { describe, it, expect } from "vitest";
import { compareVersions, loadSiteConfig, resolveReleaseRoot } from "./release";

describe("compareVersions", () => {
  it("orders identical versions as equal", () => {
    expect(compareVersions("1.2.3", "1.2.3")).toBe(0);
  });

  it("orders a newer patch release above", () => {
    expect(compareVersions("1.2.4", "1.2.3")).toBeGreaterThan(0);
    expect(compareVersions("1.2.3", "1.2.4")).toBeLessThan(0);
  });

  it("orders a newer minor release above (not lexically)", () => {
    expect(compareVersions("1.10.0", "1.9.9")).toBeGreaterThan(0);
    expect(compareVersions("1.9.9", "1.10.0")).toBeLessThan(0);
  });

  it("orders a newer major release above", () => {
    expect(compareVersions("2.0.0", "1.99.99")).toBeGreaterThan(0);
  });

  it("counts a missing segment as zero", () => {
    expect(compareVersions("1.2", "1.2.0")).toBe(0);
    expect(compareVersions("1.2.1", "1.2")).toBeGreaterThan(0);
    expect(compareVersions("2", "1.9.9")).toBeGreaterThan(0);
  });

  it("treats a pre-release tag as older than the same numeric", () => {
    expect(compareVersions("1.2.0", "1.2.0-beta.1")).toBeGreaterThan(0);
    expect(compareVersions("1.2.0-beta.1", "1.2.0")).toBeLessThan(0);
  });

  it("ignores an optional leading v", () => {
    expect(compareVersions("v1.2.0", "1.2.0")).toBe(0);
  });
});

describe("loadSiteConfig", () => {
  const read = (raw: string | null) => (() => {
    if (raw === null) throw new Error("ENOENT: no such file");
    return raw;
  });

  it("reads releaseHome from a present site.json", () => {
    const cfg = loadSiteConfig("/opt/mimir", read(JSON.stringify({ releaseHome: "/shared/mimir" })));
    expect(cfg?.releaseHome).toBe("/shared/mimir");
  });

  it("returns null when the file is absent", () => {
    expect(loadSiteConfig("/opt/mimir", read(null))).toBeNull();
  });

  it("returns null on malformed JSON", () => {
    expect(loadSiteConfig("/opt/mimir", read("not json {"))).toBeNull();
  });

  it("does not fall back to a default when the file is absent (generic distribution)", () => {
    expect(loadSiteConfig("/opt/mimir", read(""))).toBeNull();
  });
});

describe("resolveReleaseRoot", () => {
  it("falls back to site.json when the setting holds its declared default (\"\")", () => {
    // VS Code returns the declared default — "" — for every user who never
    // touched mimir.releaseHome. This is the 1.1.1 bug: ?? kept that "" and
    // site.json was never consulted.
    expect(resolveReleaseRoot("", "/shared/mimir")).toBe("/shared/mimir");
  });

  it("falls back to site.json when the setting is unset (undefined)", () => {
    expect(resolveReleaseRoot(undefined, "/shared/mimir")).toBe("/shared/mimir");
  });

  it("lets an explicit setting win over site.json", () => {
    expect(resolveReleaseRoot("/other/release", "/shared/mimir")).toBe("/other/release");
  });

  it("yields an empty root when neither source has a path", () => {
    expect(resolveReleaseRoot("", undefined)).toBe("");
    expect(resolveReleaseRoot(undefined, undefined)).toBe("");
  });

  it("trims whitespace and trailing slashes before the existence probe", () => {
    expect(resolveReleaseRoot("  /shared/mimir/  ", undefined)).toBe("/shared/mimir");
  });
});
