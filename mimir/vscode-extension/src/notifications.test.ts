import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const window = {
  state: { focused: true },
  focusListeners: [] as ((state: { focused: boolean }) => void)[],
};
const settings: Record<string, unknown> = {};
const shown: string[] = [];
const commands: string[] = [];

vi.mock("vscode", () => {
  const show = (message: string) => {
    shown.push(message);
    return Promise.resolve(undefined);
  };
  return {
    workspace: {
      getConfiguration: () => ({ get: (key: string) => settings[key] }),
    },
    window: {
      get state() { return window.state; },
      showInformationMessage: show,
      showWarningMessage: show,
      showErrorMessage: show,
      onDidChangeWindowState: (listener: (state: { focused: boolean }) => void) => {
        window.focusListeners.push(listener);
        return {
          dispose: () => {
            window.focusListeners = window.focusListeners.filter((l) => l !== listener);
          },
        };
      },
    },
    commands: {
      executeCommand: (command: string) => {
        commands.push(command);
        return Promise.resolve();
      },
    },
  };
});

import { showToast } from "./notifications";

beforeEach(() => {
  vi.useFakeTimers();
  window.state = { focused: true };
  window.focusListeners = [];
  for (const key of Object.keys(settings)) delete settings[key];
  shown.length = 0;
  commands.length = 0;
});

afterEach(() => {
  vi.useRealTimers();
});

describe("a MIMIR notification", () => {
  it("leaves the screen after five seconds by default", () => {
    void showToast("error", "disk full");
    expect(shown).toEqual(["disk full"]);
    vi.advanceTimersByTime(4999);
    expect(commands).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(commands).toEqual(["notifications.hideToasts"]);
  });

  it("restarts the countdown when another is shown, so the newer one gets its full time", () => {
    void showToast("info", "first");
    vi.advanceTimersByTime(4000);
    void showToast("info", "second");
    vi.advanceTimersByTime(4000);
    expect(commands).toEqual([]);
    vi.advanceTimersByTime(1000);
    expect(commands).toEqual(["notifications.hideToasts"]);
  });

  it("only counts down once the window has focus", () => {
    window.state = { focused: false };
    void showToast("warn", "waiting on you");
    vi.advanceTimersByTime(60000);
    expect(commands).toEqual([]);
    window.focusListeners.forEach((l) => l({ focused: true }));
    vi.advanceTimersByTime(5000);
    expect(commands).toEqual(["notifications.hideToasts"]);
  });

  it("takes its duration from the setting, and 0 leaves it on screen", () => {
    settings["notifications.duration"] = 2;
    void showToast("info", "short");
    vi.advanceTimersByTime(2000);
    expect(commands).toEqual(["notifications.hideToasts"]);

    commands.length = 0;
    settings["notifications.duration"] = 0;
    void showToast("info", "kept");
    vi.advanceTimersByTime(60000);
    expect(commands).toEqual([]);
  });
});
