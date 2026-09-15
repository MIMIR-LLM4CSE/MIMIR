import { describe, it, expect } from "vitest";
import * as http from "http";
import type { AddressInfo } from "net";
import { fetchModels, modelsUrl, parseModels } from "./modelList";

describe("modelsUrl", () => {
  it("builds the OpenAI models path for vLLM", () => {
    expect(modelsUrl("vllm", "http://10.0.0.4:8000")).toBe("http://10.0.0.4:8000/v1/models");
  });

  it("does not double the /v1 the user already typed", () => {
    expect(modelsUrl("vllm", "https://gpu.internal/v1/")).toBe("https://gpu.internal/v1/models");
  });

  it("builds the OpenAI models path for Ray Serve", () => {
    expect(modelsUrl("ray", "http://head:8000")).toBe("http://head:8000/v1/models");
  });

  it("keeps a Ray app's route prefix ahead of /v1", () => {
    expect(modelsUrl("ray", "http://head:8000/llm/")).toBe("http://head:8000/llm/v1/models");
  });

  it("builds the tags path for Ollama", () => {
    expect(modelsUrl("ollama", "http://127.0.0.1:11434/")).toBe("http://127.0.0.1:11434/api/tags");
  });
});

describe("parseModels", () => {
  it("reads data[].id from a vLLM response", () => {
    const body = { object: "list", data: [{ id: "Qwen3-32B" }, { id: "/models/devstral" }] };
    expect(parseModels("vllm", body)).toEqual(["Qwen3-32B", "/models/devstral"]);
  });

  it("reads data[].id from a Ray Serve response", () => {
    const body = { object: "list", data: [{ id: "Qwen3-32B" }] };
    expect(parseModels("ray", body)).toEqual(["Qwen3-32B"]);
  });

  it("reads models[].name from an Ollama response", () => {
    const body = { models: [{ name: "qwen3:8b", size: 5 }, { name: "llama3:70b" }] };
    expect(parseModels("ollama", body)).toEqual(["qwen3:8b", "llama3:70b"]);
  });

  it("drops blank entries and duplicates", () => {
    const body = { data: [{ id: "a" }, { id: "" }, { id: "a" }, { nope: 1 }] };
    expect(parseModels("vllm", body)).toEqual(["a"]);
  });

  it("returns [] for a shape it does not recognise", () => {
    expect(parseModels("vllm", { error: "not found" })).toEqual([]);
    expect(parseModels("ollama", null)).toEqual([]);
    expect(parseModels("vllm", "plain text")).toEqual([]);
  });
});

describe("fetchModels always settles", () => {
  /** Start a server running *handler*, and return its base URL. */
  async function serving(
    handler: http.RequestListener,
  ): Promise<{ url: string; close: () => Promise<void> }> {
    const server = http.createServer(handler);
    await new Promise<void>(done => server.listen(0, "127.0.0.1", done));
    const { port } = server.address() as AddressInfo;
    return {
      url: `http://127.0.0.1:${port}`,
      close: () => new Promise<void>(done => { server.closeAllConnections?.(); server.close(() => done()); }),
    };
  }

  it("rejects an endpoint that accepts the connection and never answers", async () => {
    // The form waits on this promise. A request that neither resolves nor rejects
    // leaves it saying "asking…" forever, with no cause to show — the failure the
    // deadline exists to prevent.
    const { url, close } = await serving(() => { /* deliberately no response */ });
    try {
      await expect(fetchModels("vllm", url, true, 150)).rejects.toThrow(/timed out/);
    } finally {
      await close();
    }
  });

  it("names the phase the request reached, so a hang points somewhere", async () => {
    const { url, close } = await serving(() => {});
    try {
      // Connected and silent is the endpoint's fault; never getting a socket would
      // be the route's or the proxy's. The message has to tell those apart.
      await expect(fetchModels("vllm", url, true, 150)).rejects.toThrow(/waiting for a response/);
    } finally {
      await close();
    }
  });

  it("reads the list when the endpoint answers", async () => {
    const { url, close } = await serving((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ object: "list", data: [{ id: "a" }, { id: "b" }] }));
    });
    try {
      await expect(fetchModels("vllm", url, true, 2000)).resolves.toEqual(["a", "b"]);
    } finally {
      await close();
    }
  });

  it("reports a refusal rather than waiting out the deadline", async () => {
    const { url, close } = await serving(() => {});
    await close();  // nothing is listening on that port any more
    await expect(fetchModels("vllm", url, true, 2000)).rejects.toThrow(/ECONNREFUSED/);
  });
});
