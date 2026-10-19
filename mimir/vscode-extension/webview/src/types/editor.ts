// The editor's side: the active file, the workspace files and resources to attach, a file
// the agent asks to open.

/** An MCP resource that can be @-attached to a query. */
export interface ResourceItem {
  uri: string;
  name: string;
  description: string;
  mimeType?: string | null;
}

export interface ResourcesMessage {
  type: "resources";
  resources: ResourceItem[];
}

/** Workspace files matching an "@" query, ranked by the extension host.
 *  `query` echoes the request so a reply overtaken by further typing is dropped. */
export interface FileResultsMessage {
  type: "file_results";
  query: string;
  files: string[]; // workspace-relative paths, best match first
}

/** The file (and optional selected line range) focused in the VS Code editor. */
export interface ActiveEditorMessage {
  type: "active_editor";
  file: string;
  selection: string | null; // "10" | "10-20" | null
}

/** Ask the host to open a file in a real editor tab (e.g. a written plan .md).
 *  Emitted by the agent on a tool result carrying open_in_editor + path. */
export interface OpenEditorMessage {
  type: "open_editor";
  path: string;
}

export interface ListResourcesMessage {
  type: "list_resources";
}
