import { describe, it, expect } from "vitest";
import { isNoisePath, rankFiles, scoreFilePath } from "./fileIndex";

describe("isNoisePath", () => {
  it("drops dependency and cache trees", () => {
    expect(isNoisePath("node_modules/react/index.js")).toBe(true);
    expect(isNoisePath("mimir/client/__pycache__/loop.pyc")).toBe(true);
    expect(isNoisePath("mimir_agent.egg-info/PKG-INFO")).toBe(true);
    expect(isNoisePath(".venv/lib/site-packages/x.py")).toBe(true);
  });

  it("keeps ordinary sources, including a file named like a noise directory", () => {
    expect(isNoisePath("mimir/client/loop.py")).toBe(false);
    expect(isNoisePath("scripts/build")).toBe(false);
  });
});

describe("scoreFilePath", () => {
  it("rejects a query that is not a subsequence", () => {
    expect(scoreFilePath("webview/src/App.tsx", "zzz")).toBeNull();
  });

  it("accepts scattered initials", () => {
    expect(scoreFilePath("webview/src/App.tsx", "apptsx")).not.toBeNull();
  });

  it("prefers a match in the file name over one in the directory", () => {
    const inName = scoreFilePath("client/context/resource_context.py", "resource")!;
    const inDir = scoreFilePath("resource/deep/down/other.py", "resource")!;
    expect(inName).toBeGreaterThan(inDir);
  });
});

describe("rankFiles", () => {
  const paths = [
    "README.md",
    "mimir/client/loop.py",
    "mimir/client/context/resource_context.py",
    "mimir/vscode-extension/webview/src/App.tsx",
    "mimir/vscode-extension/src/extension.ts",
  ];

  it("puts the closest match first", () => {
    expect(rankFiles(paths, "app")[0]).toBe("mimir/vscode-extension/webview/src/App.tsx");
    expect(rankFiles(paths, "loop")[0]).toBe("mimir/client/loop.py");
  });

  it("matches a path fragment the user typed with a slash", () => {
    expect(rankFiles(paths, "context/resource")).toEqual([
      "mimir/client/context/resource_context.py",
    ]);
  });

  it("opens on the shallowest paths when nothing is typed yet", () => {
    expect(rankFiles(paths, "", 2)).toEqual(["README.md", "mimir/client/loop.py"]);
  });

  it("honours the limit", () => {
    expect(rankFiles(paths, "p", 2)).toHaveLength(2);
  });
});
