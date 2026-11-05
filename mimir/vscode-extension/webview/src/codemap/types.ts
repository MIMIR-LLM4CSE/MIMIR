// The code map as the server answers it (`shared.code_map`).

export interface Component { id: string; name: string; tier: number; declared: string[]; external: Record<string, number>; tests: boolean }
export interface Module { id: string; component: string }
export interface File { id: string; lang: string; lines: number; module: string; component: string }
export interface Edge {
  from: string; to: string; line: number; kind: string; via: string;
  sure: "exact" | "by_name"; deferred: boolean; breaks: string;
}
export interface CodeMap {
  root: string; source_root: string; components: Component[]; modules: Module[]; files: File[];
  edges: Edge[]; cycles: string[][]; unresolved: Record<string, number>; truncated: boolean;
}

export type Selection = { kind: "comp" | "module" | "file" | "request"; id: string } | null;
