import { describe, it, expect } from "vitest";
import { modelsUrl, parseModels } from "./modelList";

describe("modelsUrl", () => {
  it("builds the OpenAI models path for vLLM", () => {
    expect(modelsUrl("vllm", "http://10.0.0.4:8000")).toBe("http://10.0.0.4:8000/v1/models");
  });

  it("does not double the /v1 the user already typed", () => {
    expect(modelsUrl("vllm", "https://gpu.internal/v1/")).toBe("https://gpu.internal/v1/models");
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
