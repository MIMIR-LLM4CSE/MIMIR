import React, { useState, useMemo, useEffect } from "react";
import type { ClusterConfig } from "../types";

// Human-readable memory bus type per GPU model
const GPU_MEM_TYPE: Record<string, string> = {
  // NVIDIA Hopper / Grace-Hopper
  gh200:     "HBM3",
  h100:      "HBM3",
  h200:      "HBM3e",
  // NVIDIA Blackwell
  b100:      "HBM3e",
  b200:      "HBM3e",
  b300:      "HBM4",
  gb200:     "HBM3e",
  gb300:     "HBM4",
  // NVIDIA Ampere
  a100:      "HBM2e",
  a30:       "HBM2",
  a10:       "GDDR6",
  a6000:     "GDDR6",
  // NVIDIA Turing / Volta / Pascal
  v100:      "HBM2",
  t4:        "GDDR6",
  p100:      "HBM2",
  // NVIDIA RTX (workstation/consumer)
  rtx:       "GDDR6",
  rtx4090:   "GDDR6X",
  rtx3090:   "GDDR6X",
  // NVIDIA MIG slices
  "1g.24gb": "HBM3 MIG",
  "2g.48gb": "HBM3 MIG",
  "3g.96gb": "HBM3 MIG",
  // AMD Instinct
  mi300x:    "HBM3",
  mi325x:    "HBM3E",
  mi350x:    "HBM3E",
  mi400x:    "HBM4",
  // AMD Radeon
  rx7900:    "GDDR6",
};

function modelLabel(name: string, sizes: Record<string, number>): string {
  const gb = sizes[name];
  return gb !== undefined ? `${name} (${gb} GB)` : name;
}

/** For vLLM paths like /foo/bar/gpt-oss-20b, show only the basename with size. */
function vllmModelLabel(path: string, sizes: Record<string, number>): string {
  const base = path.split("/").pop() ?? path;
  const gb = sizes[base] ?? sizes[path];
  return gb !== undefined ? `${base} (${gb} GB)` : base;
}

function gpuCountOptions(max: number): number[] {
  if (max <= 1) return [1];
  const s = new Set<number>();
  for (let n = 1; n <= max; n *= 2) s.add(n);
  s.add(max);
  return [...s].sort((a, b) => a - b);
}

interface Props {
  clusters: ClusterConfig[];
  availableModels: string[];
  vllmModels?: string[];
  anthropicModels?: string[];
  modelSizes?: Record<string, number>;
  backend?: string;
  vllmBaseUrl?: string;
  vllmMode?: "launch" | "connect";
  onConnect: (model: string, loginNode: string | undefined, slurmArgs: string, backend: string, vllmBaseUrl: string, vllmPath?: string, ollamaPath?: string, vllmMode?: "launch" | "connect", anthropicApiKey?: string) => void;
}

export const ConnectForm: React.FC<Props> = ({ clusters, availableModels, vllmModels = [], anthropicModels = [], modelSizes = {}, backend: initBackend = "ollama", vllmBaseUrl: initVllmUrl = "http://127.0.0.1:8000", vllmMode: initVllmMode = "launch", onConnect }) => {
  const [backend, setBackend]           = useState(initBackend);
  const [vllmUrl, setVllmUrl]           = useState(initVllmUrl);
  // Claude API key — kept in webview state only; forwarded on connect and never
  // persisted. Left blank means "use whatever ANTHROPIC_API_KEY the host exports".
  const [apiKey, setApiKey]             = useState("");
  // For vLLM: "launch" = start `vllm serve` on a compute node (allocates SLURM);
  // "connect" = point the agent at an already-running vLLM at vllmUrl (no SLURM).
  const [vllmMode, setVllmMode]         = useState<"launch" | "connect">(initVllmMode);
  // In "connect" mode vLLM already runs elsewhere: no SLURM allocation and no model
  // pick (the agent uses whatever the endpoint serves), so those controls are hidden.
  const vllmConnectOnly = backend === "vllm" && vllmMode === "connect";
  // The hosted Claude API needs no GPU/SLURM/cluster — only a model + key.
  const anthropicBackend = backend === "anthropic";
  // Controls that only make sense for a self-hosted model (cluster, node type,
  // GPUs, memory, salloc preview) are hidden for both vLLM-connect and Anthropic.
  const localOnly = vllmConnectOnly || anthropicBackend;
  const activeModels = backend === "vllm" ? vllmModels : anthropicBackend ? anthropicModels : availableModels;
  const [model, setModel]           = useState(activeModels[0] ?? "");
  const [clusterIdx, setClusterIdx] = useState(0);
  const [nodeTypeIdx, setNodeTypeIdx] = useState(0);
  const [gpuCount, setGpuCount]     = useState<number>(1);
  const [memGB, setMemGB]           = useState<number | null>(null);

  // Sync model when availableModels arrives asynchronously
  useEffect(() => {
    if (activeModels.length > 0 && !model) setModel(activeModels[0]);
  }, [activeModels]);

  const cluster  = clusters[clusterIdx];
  const nodeType = cluster?.nodeTypes[nodeTypeIdx];
  const gpuSpec  = nodeType?.gpu ?? null;
  const memOptions = nodeType?.memOptionsGB ?? [64];
  const selectedMem = memGB ?? memOptions[memOptions.length - 1];

  const slurmArgs = useMemo(() => {
    if (!nodeType) return "";
    const parts: string[] = [
      `-p ${nodeType.partition}`,
      `-n ${nodeType.cpusPerNode}`,
    ];
    if (gpuSpec && gpuCount > 0)
      parts.push(`--gres=gpu:${gpuSpec.type}:${gpuCount}`);
    if (cluster?.account)
      parts.push(`--account ${cluster.account}`);
    parts.push(`--mem ${selectedMem}G`);
    return parts.join(" ");
  }, [cluster, nodeType, gpuSpec, gpuCount, selectedMem]);

  const handleClusterChange = (idx: number) => {
    setClusterIdx(idx);
    setNodeTypeIdx(0);
    setGpuCount(1);
    setMemGB(null);
  };

  const handleNodeTypeChange = (idx: number) => {
    setNodeTypeIdx(idx);
    setGpuCount(1);
    setMemGB(null);
  };

  // No cluster config: simple model + connect
  if (!clusters || clusters.length === 0) {
    return (
      <div className="connect-form">
        <div className="connect-field">
          <label className="connect-label">Backend</label>
          <select className="connect-select" value={backend} onChange={e => { setBackend(e.target.value); setModel(""); }}>
            <option value="ollama">Ollama</option>
            <option value="vllm">vLLM</option>
            <option value="anthropic">Anthropic (Claude)</option>
          </select>
        </div>
        {anthropicBackend && (
          <div className="connect-field">
            <label className="connect-label">API key</label>
            <input className="connect-input" type="password" autoComplete="off" placeholder="sk-ant-…" value={apiKey} onChange={e => setApiKey(e.target.value)} />
            <div className="connect-field-hint">Leave blank to use <code>$ANTHROPIC_API_KEY</code></div>
          </div>
        )}
        {backend === "vllm" && (
          <div className="connect-field">
            <label className="connect-label">vLLM mode</label>
            <select className="connect-select" value={vllmMode} onChange={e => setVllmMode(e.target.value as "launch" | "connect")}>
              <option value="launch">Launch on compute node</option>
              <option value="connect">Connect to running server</option>
            </select>
          </div>
        )}
        {backend === "vllm" && (
          <div className="connect-field">
            <label className="connect-label">{vllmMode === "connect" ? "vLLM address" : "vLLM URL"}</label>
            <input className="connect-input" type="text" value={vllmUrl} onChange={e => setVllmUrl(e.target.value)} />
          </div>
        )}
        {!vllmConnectOnly && activeModels.length > 0 && (
          <div className="connect-field">
            <label className="connect-label">Model</label>
            <select className="connect-select" value={model} onChange={e => setModel(e.target.value)}>
              {activeModels.map(m => <option key={m} value={m}>{backend === "vllm" ? vllmModelLabel(m, modelSizes) : modelLabel(m, modelSizes)}</option>)}
            </select>
          </div>
        )}
        <button className="connect-btn" onClick={() => onConnect(vllmConnectOnly ? "" : model, undefined, "", backend, vllmUrl, undefined, undefined, vllmMode, anthropicBackend ? apiKey : undefined)}>  {/* no cluster — paths come from global settings */}
          Connect
        </button>
      </div>
    );
  }

  const gpuOptions = gpuSpec ? gpuCountOptions(gpuSpec.maxCount) : [];

  return (
    <div className="connect-form">

      {/* ── Backend ── */}
      <div className="connect-field">
        <label className="connect-label">Backend</label>
        <select className="connect-select" value={backend} onChange={e => { setBackend(e.target.value); setModel(""); }}>
          <option value="ollama">Ollama</option>
          <option value="vllm">vLLM</option>
          <option value="anthropic">Anthropic (Claude)</option>
        </select>
      </div>

      {/* ── API key (only when backend=anthropic) ── */}
      {anthropicBackend && (
        <div className="connect-field">
          <label className="connect-label">API key</label>
          <input className="connect-input" type="password" autoComplete="off" placeholder="sk-ant-…" value={apiKey} onChange={e => setApiKey(e.target.value)} />
          <div className="connect-field-hint">Leave blank to use <code>$ANTHROPIC_API_KEY</code></div>
        </div>
      )}

      {/* ── vLLM mode (only when backend=vllm) ── */}
      {backend === "vllm" && (
        <div className="connect-field">
          <label className="connect-label">vLLM mode</label>
          <select className="connect-select" value={vllmMode} onChange={e => setVllmMode(e.target.value as "launch" | "connect")}>
            <option value="launch">Launch on compute node</option>
            <option value="connect">Connect to running server</option>
          </select>
        </div>
      )}

      {/* ── vLLM Base URL (only when backend=vllm) ── */}
      {backend === "vllm" && (
        <div className="connect-field">
          <label className="connect-label">{vllmConnectOnly ? "vLLM address" : "vLLM URL"}</label>
          <input className="connect-input" type="text" value={vllmUrl} onChange={e => setVllmUrl(e.target.value)} />
        </div>
      )}

      {/* ── Model (hidden in vLLM connect mode — the endpoint's model is used) ── */}
      {!vllmConnectOnly && activeModels.length > 0 && (
        <div className="connect-field">
          <label className="connect-label">Model</label>
          <select className="connect-select" value={model} onChange={e => setModel(e.target.value)}>
            {activeModels.map(m => <option key={m} value={m}>{backend === "vllm" ? vllmModelLabel(m, modelSizes) : modelLabel(m, modelSizes)}</option>)}
          </select>
        </div>
      )}

      {/* ── Cluster ── */}
      {!localOnly && (
      <div className="connect-field">
        <label className="connect-label">Cluster</label>
        <select
          className="connect-select"
          value={clusterIdx}
          onChange={e => handleClusterChange(Number(e.target.value))}
        >
          {clusters.map((c, i) => (
            <option key={c.name} value={i}>{c.name}</option>
          ))}
        </select>
      </div>
      )}

      {/* ── Node type (hidden when only one option) ── */}
      {!localOnly && cluster && cluster.nodeTypes.length > 1 && (
        <div className="connect-field">
          <label className="connect-label">Node type</label>
          <select
            className="connect-select"
            value={nodeTypeIdx}
            onChange={e => handleNodeTypeChange(Number(e.target.value))}
          >
            {cluster.nodeTypes.map((nt, i) => (
              <option key={nt.partition} value={i}>{nt.label}</option>
            ))}
          </select>
        </div>
      )}

      {/* ── GPU count (only for GPU node types) ── */}
      {!localOnly && gpuSpec && gpuOptions.length > 0 && (
        <div className="connect-field">
          <label className="connect-label">GPUs</label>
          <select
            className="connect-select"
            value={gpuCount}
            onChange={e => setGpuCount(Number(e.target.value))}
          >
            {gpuOptions.map(n => (
              <option key={n} value={n}>
                {n} × {gpuSpec.type.toUpperCase()} ({gpuSpec.memGB} GB {GPU_MEM_TYPE[gpuSpec.type] ?? ""})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* ── Memory ── */}
      {!localOnly && (
      <div className="connect-field">
        <label className="connect-label">Memory</label>
        <select
          className="connect-select"
          value={selectedMem}
          onChange={e => setMemGB(Number(e.target.value))}
        >
          {memOptions.map(m => (
            <option key={m} value={m}>{m} GB</option>
          ))}
        </select>
      </div>
      )}

      {/* ── salloc preview ── */}
      {!localOnly && nodeType && (
        <div className="connect-preview">
          <span className="connect-preview-label">salloc</span>
          <code className="connect-preview-args">{slurmArgs}</code>
          {cluster?.loginNode && (
            <span className="connect-preview-node">via {cluster.loginNode}</span>
          )}
        </div>
      )}

      <button
        className="connect-btn connect-btn-primary"
        onClick={() => onConnect(vllmConnectOnly ? "" : model, anthropicBackend ? undefined : cluster?.loginNode, anthropicBackend ? "" : slurmArgs, backend, vllmUrl, cluster?.vllmPath, cluster?.ollamaPath, vllmMode, anthropicBackend ? apiKey : undefined)}
      >
        Connect
      </button>
    </div>
  );
};
