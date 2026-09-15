from __future__ import annotations

import re
from functools import lru_cache


# These two no longer classify anything on their own: the nudges that asked "did the
# user word this as an edit / as a creation?" were removed, because a keyword over a
# natural-language request is a guess about intent and a nudge must rest on a fact. They
# survive as ingredients of QUERY_DISCOVERY_SIGNALS below, whose single consumer reads
# the union as a coarse exit filter and nothing more.
QUERY_EDIT_SIGNALS: tuple[str, ...] = (
    "improve", "ameliore", "améliore", "modify", "modifie", "update",
    "edit", "patch", "refactor", "fix", "correct",
)

QUERY_CREATE_SIGNALS: tuple[str, ...] = (
    "create", "new", "add", "nouveau", "nouvelle", "ajoute",
    "scaffold", "generate",
)

# HPC / performance / hardware intents. A query matching one of these benefits from
# hardware-aware context (the cached platform profile) during plan discovery. This is a
# narrower set than QUERY_SCIENCE_SIGNALS — performance and architecture, not theory.
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


# Scientific-computing intents ("From Math, to HPC"): theory/derivation, performance,
# and bibliography. These trigger evidence gathering just like code-discovery terms.
QUERY_SCIENCE_SIGNALS: tuple[str, ...] = (
    "derive", "derivation", "dérive", "prove", "proof", "prouve",
    "theorem", "théorème", "lemma",
    "integrate", "integral", "intègre",
    "differentiate", "derivative", "dérivée", "simplify", "solve", "résous",
    "benchmark", "profile", "profiling",
    "optimize", "optimise", "optimisation", "optimization",
    "speed up", "accelerate", "accélère",
    "parallelize", "parallelise", "parallélise", "vectorize", "vectorise",
    "simd", "openmp", "mpi", "gpu", "cuda", "flops",
    "complexity", "complexité",
    "reference", "référence", "cite", "citation",
    "paper", "article", "bibliography", "bibliographie",
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

# Composed so each term lives in exactly one source set; dedup preserves order.
#
# Read this as an EXIT filter, not a detector: the union is broad enough to be true for
# almost any repo-touching request, and that is intended. Its job is to exclude pure
# theory, bibliography and chit-chat, not to discriminate among coding tasks — its one
# consumer (the plan-mode explore phase) must not read a positive as more than "this
# query plausibly touches the workspace".
#
# Hence QUERY_SCIENCE_SIGNALS (derive/prove/cite/theorem…) is excluded: a derivation or
# literature query needs no *repository* discovery. QUERY_HPC_SIGNALS
# (optimize/benchmark/parallelize…) does touch code, so it stays in.
QUERY_DISCOVERY_SIGNALS: tuple[str, ...] = tuple(
    dict.fromkeys(
        QUERY_EDIT_SIGNALS
        + QUERY_CREATE_SIGNALS
        + QUERY_HPC_SIGNALS
        + _QUERY_DISCOVERY_ONLY
    )
)

# Tool-classification sets (search / edit / validate) are no longer defined here:
# each server declares its tools' capabilities and consumers read them from the
# per-agent live registry (agent.tool_caps) via has_cap()/names_with_cap().




# What counts as source: an edit to one of these is recorded as produced work and owes
# a check. Every source language belongs here whatever this machine has installed: the
# check itself is performed in-process (guardrails.builtin_check), so what a file owes
# no longer depends on a binary being on PATH. Every spelling of a language belongs
# here too — `.f03` was once missing, and a Fortran 2003 file was then never even
# recorded as modified.
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
# Query-signal matching. Intent detection matches a *word*, not a raw substring — a
# naive ``token in text`` fires on "create" inside "creative", "add" inside "address".
# Everything goes through ``query_matches_any``, which anchors tokens at ``\b``.
# Multi-word phrases ("speed up") work; ``\w`` is Unicode-aware, so accented French
# tokens ("améliore") get correct boundaries too.
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
# One predicate, one consumer. The guardrail layer used to host four of these, asking
# whether the user's wording meant "edit", "create" or "just a question", and the nudges
# read the answers. That is a guess about intent, and a nudge must rest on something the
# code can check — so those predicates went, along with the nudge conditions that read
# them. What the nudges ask now is what actually happened: a file was edited, a target
# was declared, a run failed.
#
# This one survives for the plan-mode explore phase (``query_engine/plan_loop``), where
# it is used as the coarse exit filter the union above describes — "does this request
# plausibly touch the workspace at all" — and never to discriminate between tasks.

def query_requires_repo_discovery(query: str) -> bool:
    return query_matches_any(query, QUERY_DISCOVERY_SIGNALS)
