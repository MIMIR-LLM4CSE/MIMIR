// What the webview asks of the extension host, posted with `vscodePostMessage`. The
// host answers on the same channel; messages for the Python server travel as `ws_send`.

export type HostMessage =
  | { type: "open_file"; file: string; line?: number; end_line?: number }
  | { type: "open_preview"; file: string }
  | { type: "open_diff"; file: string; new_content?: string; patch?: string }
  | { type: "open_patch"; file: string; patch: string }
  | { type: "disconnect" }
  | { type: "file_search"; query: string }
  | { type: "get_config" }
  | { type: "fetch_models"; backend?: string; baseUrl?: string }
  | { type: "connect"; model?: string; backend?: string; baseUrl?: string;
      anthropicApiKey?: string; remember?: boolean }
  | { type: "ws_send"; payload: string }
  | { type: "show_code_map"; map: unknown };

export type HostMessageOf<T extends HostMessage["type"]> = Extract<HostMessage, { type: T }>;
