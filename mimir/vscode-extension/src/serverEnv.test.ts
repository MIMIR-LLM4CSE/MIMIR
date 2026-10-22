import * as fs from "fs";
import * as path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";

/** Settings as the user set them, by key under `mimir.`; anything absent is left at its default. */
let userSettings: Record<string, unknown> = {};

vi.mock("vscode", () => ({
  workspace: {
    getConfiguration: () => ({
      inspect: (key: string) => ({ key, globalValue: userSettings[key] }),
      get: (key: string) => userSettings[key],
    }),
  },
}));

import { EXTRA_ENV_SETTING, SERVER_ENV_SETTINGS, settingsEnv } from "./serverEnv";

const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "package.json"), "utf8"));
const declared: Record<string, { markdownDescription?: string; description?: string }> =
  Object.assign({}, ...manifest.contributes.configuration.map((c: any) => c.properties));

beforeEach(() => { userSettings = {}; });

describe("the settings the server takes as environment", () => {
  it("are each declared in the manifest", () => {
    for (const { key } of [...SERVER_ENV_SETTINGS, { key: EXTRA_ENV_SETTING }]) {
      expect(declared, key).toHaveProperty([`mimir.${key}`]);
    }
  });

  it("cover every declared setting that says it sets a variable", () => {
    const mapped = new Set(SERVER_ENV_SETTINGS.map((s) => `mimir.${s.key}`));
    for (const [key, spec] of Object.entries(declared)) {
      const says = spec.markdownDescription ?? spec.description ?? "";
      if (/Sets `?[A-Z_]+/.test(says)) expect(mapped, key).toContain(key);
    }
  });

  it("set nothing while every setting is at its default", () => {
    expect(settingsEnv()).toEqual({});
  });

  it("turn what the user set into the server's variables", () => {
    userSettings = {
      "maxModelLen": 32768,
      "vllmTokenize": false,
      "anthropic.thinking": false,
      "embeddings.model": " BAAI/bge-m3 ",
      "workspace.extraReadRoots": ["/data", "", "/opt/lib"],
    };
    expect(settingsEnv()).toEqual({
      MIMIR_VLLM_MAX_MODEL_LEN: "32768",
      MIMIR_RAY_MAX_MODEL_LEN: "32768",
      MIMIR_VLLM_TOKENIZE: "0",
      MIMIR_ANTHROPIC_THINKING: "off",
      MIMIR_EMBED_MODEL: "BAAI/bge-m3",
      MIMIR_EXTRA_FILE_ROOTS: ["/data", "/opt/lib"].join(path.delimiter),
    });
  });

  it("leave a value the server already defaults to unset", () => {
    userSettings = { "vllmVerifySsl": true, "maxModelLen": 0, "embeddings.model": "  " };
    expect(settingsEnv()).toEqual({});
  });

  it("let the free-form variables win over a setting", () => {
    userSettings = {
      "sessions.maxLive": 2,
      [EXTRA_ENV_SETTING]: { MIMIR_MAX_LIVE_SESSIONS: "5", MIMIR_PROXY_BENCH_DIR: "/scratch" },
    };
    expect(settingsEnv()).toEqual({ MIMIR_MAX_LIVE_SESSIONS: "5", MIMIR_PROXY_BENCH_DIR: "/scratch" });
  });
});
