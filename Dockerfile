# MIMIR — reproducible container image.
#
# Ships the agent and every MCP server that is registered by default. Uses the
# vLLM "connect" backend (attach to an already-running OpenAI-compatible endpoint)
# or Ollama reachable over the network. The SLURM "launch" mode is NOT supported
# inside the container (it needs SSH access to a login node) — run that on a
# frontal node instead.
#
# Build:   docker build -t mimir:latest .
# Run WS:  docker run --rm -p 8765:8765 -v "$PWD":/workspace \
#              -e LLM_BACKEND=vllm -e VLLM_BASE_URL=http://<node>:8000 mimir:latest
# Run CLI: docker run --rm -it -v "$PWD":/workspace \
#              -e LLM_BACKEND=vllm -e VLLM_BASE_URL=http://<node>:8000 mimir:latest mimir

FROM python:3.11-slim AS base

# git is used by the localgit MCP server and general repo operations.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN useradd --create-home --uid 1000 mimir

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MCP_FILES_ROOT=/workspace

WORKDIR /opt/mimir

# Install dependencies first (better layer caching), then the package itself.
COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY mimir ./mimir
RUN pip install --upgrade pip && pip install ".[vllm]"

# The user's project is mounted here and used as MCP_FILES_ROOT (the sandbox root).
WORKDIR /workspace
RUN chown mimir:mimir /workspace
USER mimir

EXPOSE 8765

# Default: start the WebSocket server the VS Code extension connects to.
# Override the trailing command (e.g. `... mimir`) to run the interactive CLI.
CMD ["mimir-server", "--host", "0.0.0.0", "--port", "8765"]
