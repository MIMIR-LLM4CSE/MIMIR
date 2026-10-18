from __future__ import annotations

import re
from functools import lru_cache


# These two classify nothing on their own: a keyword over a request is a guess about
# intent, and a nudge rests on facts. They are ingredients of QUERY_DISCOVERY_SIGNALS
# below, whose single consumer reads the union as a coarse exit filter.
QUERY_EDIT_SIGNALS: tuple[str, ...] = (
    "improve", "ameliore", "améliore", "modify", "modifie", "update",
    "edit", "patch", "refactor", "fix", "correct",
)

QUERY_CREATE_SIGNALS: tuple[str, ...] = (
    "create", "new", "add", "nouveau", "nouvelle", "ajoute",
    "scaffold", "generate",
)

# HPC / performance / hardware intents: performance and architecture, not theory. Like
# the two sets above, these are ingredients of QUERY_DISCOVERY_SIGNALS and nothing else.
QUERY_HPC_SIGNALS: tuple[str, ...] = (
    "benchmark", "profile", "profiling",
    "optimize", "optimise", "optimization", "optimisation",
    "speed up", "speedup", "faster", "accelerate", "accélère",
    "parallelize", "parallelise", "parallélise", "parallel",
    "vectorize", "vectorise", "simd", "openmp", "mpi", "gpu", "cuda",
    "kernel", "flops", "throughput", "latency", "numa", "cache",
    "thread", "threads", "core", "cores",
    "slurm", "hpc", "cluster", "compiler", "scaling", "performance", "perf",
)


# Discovery-only terms (file/repo orientation) not already implied by edit/create/science.
_QUERY_DISCOVERY_ONLY: tuple[str, ...] = (
    "fichier", "file", "files", "repo", "repository", "arbo", "tree",
    "codebase", "project", "projet", "structure",
    "function", "fonction", "class", "classe", "method", "module",
    "scan", "search", "read", "analyze", "analyse", "inspect", "locate", "where",
    "cherche", "trouve", "montre", "liste", "lis",
    "explain", "understand", "describe", "show",
    "explique", "explication", "comprendre", "decris", "décris",
    "plan", "suggest", "propose", "recommend", "conseil",
    "server", "serveur",
)

# An exit filter, not a detector: broad enough to fit almost any repository request,
# it only excludes pure theory, bibliography and chit-chat. Its one consumer is the
# plan-mode explore phase.
QUERY_DISCOVERY_SIGNALS: tuple[str, ...] = tuple(
    dict.fromkeys(
        QUERY_EDIT_SIGNALS
        + QUERY_CREATE_SIGNALS
        + QUERY_HPC_SIGNALS
        + _QUERY_DISCOVERY_ONLY
    )
)


# What counts as source: an edit to one of these is recorded as produced work and owes
# a check. Every source language belongs here whatever this machine has installed: the
# check itself is performed in-process (guardrails.verification.builtin_check), so what a file owes
# does not depend on a binary being on PATH. Every spelling of a language belongs here
# too (`.f`, `.f90`, `.f03`, …): a missing one is a file never recorded as modified.
SOURCE_FILE_EXTENSIONS: tuple[str, ...] = (
    # Python
    ".py", ".pyi", ".pyx", ".pxd",
    # C / C++ (sources and headers)
    ".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".h++", ".inl",
    # CUDA / HIP — first-class HPC source
    ".cu", ".cuh", ".hip",
    # Fortran, every spelling: fixed form, free form, and the preprocessed variants
    ".f", ".for", ".ftn", ".f77", ".f90", ".f95", ".f03", ".f08", ".f18",
    # JVM / .NET
    ".java", ".kt", ".kts", ".scala", ".groovy", ".cs",
    # JavaScript / TypeScript
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    # Systems / general purpose
    ".go", ".rs", ".swift", ".zig", ".ml", ".hs",
    # Scripting
    ".sh", ".bash", ".zsh", ".ksh", ".pl", ".pm", ".rb", ".php", ".lua", ".tcl",
    # Scientific / array languages
    ".jl", ".r", ".m",
    # Hardware description
    ".v", ".sv", ".vhd", ".vhdl",
    # Structured data and configuration. Not code, but a broken `pyproject.toml` or a
    # malformed manifest breaks a project exactly as a broken module does, and these
    # are the extensions the built-in floor holds a *real parser* for — the check costs
    # nothing and cannot be wrong. Formats with no stdlib parser (YAML above all) are
    # deliberately absent: recording them would only add files the floor can say
    # nothing precise about.
    ".json", ".toml", ".ini", ".cfg",
    ".xml", ".xsd", ".xsl", ".xslt", ".plist",
)


# ---------------------------------------------------------------------------
# Query-signal matching anchors each token at ``\b`` (``query_matches_any``), so "add"
# does not match "address"; ``\w`` is Unicode-aware.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _compiled_signal_pattern(tokens: tuple[str, ...]) -> "re.Pattern[str] | None":
    """Compile a word-boundary alternation for *tokens* (cached per token tuple).

    Longer tokens are listed first so the alternation prefers the most specific
    match. Returns ``None`` when there is nothing to match.
    """
    parts = sorted(
        (re.escape(t.strip()) for t in tokens if t and t.strip()),
        key=len,
        reverse=True,
    )
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def query_matches_any(query: str, tokens: tuple[str, ...]) -> bool:
    """True if any of *tokens* appears as a whole word/phrase in *query*."""
    if not query or not tokens:
        return False
    pattern = _compiled_signal_pattern(tokens)
    if pattern is None:
        return False
    return pattern.search(query) is not None


# ── query-signal reading ───────────────────────────────────────────────────────
# Used only by the plan-mode explore phase, as the coarse exit filter above.

def query_requires_repo_discovery(query: str) -> bool:
    return query_matches_any(query, QUERY_DISCOVERY_SIGNALS)
