import { describe, expect, it } from "vitest";
import { validRemembered } from "./endpoints";

describe("validRemembered", () => {
  it("keeps an addressed backend with an address", () => {
    const saved = { backend: "vllm", baseUrl: "http://node:8000/v1", model: "m" };
    expect(validRemembered(saved)).toEqual(saved);
  });

  it("drops a backend that needs a key, and an endpoint without an address", () => {
    expect(validRemembered({ backend: "anthropic", baseUrl: "x", model: "m" })).toBeUndefined();
    expect(validRemembered({ backend: "ollama", baseUrl: "", model: "m" })).toBeUndefined();
    expect(validRemembered(undefined)).toBeUndefined();
  });
});
