import { describe, it, expect } from "vitest";
import { compareVersions, loadSiteConfig } from "./release";

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
