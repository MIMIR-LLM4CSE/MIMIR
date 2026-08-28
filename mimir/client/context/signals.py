from __future__ import annotations

import re
from functools import lru_cache


QUERY_EDIT_SIGNALS: tuple[str, ...] = (
    "improve", "ameliore", "améliore", "modify", "modifie", "update",
    "edit", "patch", "refactor", "fix", "correct",
)

QUERY_CREATE_SIGNALS: tuple[str, ...] = (
    "create", "new", "add", "nouveau", "nouvelle", "ajoute",
    "scaffold", "generate",
)

# The weakest create signals: they qualify a noun far more often than they order a
# creation ("a new machine", "what's new"), so they count as create intent only when
# qualifying an artifact we could actually write. Otherwise the strong verbs carry it.
_CREATE_WEAK_SIGNALS: frozenset[str] = frozenset({"new", "nouveau", "nouvelle"})

_CREATE_ARTIFACT_NOUNS: tuple[str, ...] = (
    "file", "fichier", "script", "module", "package", "class", "classe",
    "function", "fonction", "method", "methode", "méthode", "test", "tests",
    "server", "serveur", "tool", "outil", "command", "commande", "endpoint",
    "component", "composant", "directory", "folder", "dossier", "repertoire",
    "répertoire", "branch", "branche", "entry", "entree", "entrée",
)

# How many words after a weak create signal are scanned for an artifact noun.
_CREATE_ARTIFACT_WINDOW = 3

# Question-shaped openers. A query that asks *about* the repo is answered, not acted
# on: mutation-intent classifiers stay off so the workflow nudges never push the model
# into writing something the user only asked to be told about.
_INTERROGATIVE_OPENERS: tuple[str, ...] = (
    "what", "which", "where", "when", "why", "how", "who", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "any",
    "quel", "quelle", "quels", "quelles", "que", "quoi", "qui", "où",
    "quand", "pourquoi", "comment", "combien", "est", "sont", "peux", "peut",
    "y",  # "y a-t-il"
)

# Imperative mutation verbs. Their presence means the user asked for an action even
# if the sentence is phrased as a question ("can you add a test?"), so the
# informational gate must not swallow it.
_ACTION_VERBS: tuple[str, ...] = QUERY_EDIT_SIGNALS + (
    "create", "add", "write", "implement", "generate", "scaffold", "build",
    "make", "remove", "delete", "rename", "move",
    "cree", "crée", "ajoute", "ecris", "écris", "implemente", "implémente",
    "supprime", "renomme", "deplace", "déplace",
)
# Deliberately absent: "install"/"run"/"lance" — they describe the environment, not a
# repo mutation, so "how do I install X on a new machine?" stays informational.

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
# theory, bibliography and chit-chat, not to discriminate among coding tasks — its two
# consumers (the plan-mode explore phase, the discovery nudge) must not read a positive
# as more than "this query plausibly touches the workspace".
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


# ---------------------------------------------------------------------------
# Domain tool groups — query-gated pruning of heavy, specialized tool families whose
# JSON schemas dominate per-step prompt cost (HPC, platform, benchmarking, finetune,
# proxy). Each entry maps a tuple of tool-name *prefixes* to the query keywords that
# activate the group; a group whose keywords are all absent is pruned from the tool
# list (see query_engine.toollist.inactive_domain_prefixes).
#
# Keyword lists are deliberately generous and overlapping, so a task that legitimately
# needs a domain is never starved of its tools. Core families (files, search, code,
# math, memory, todo, web, system) carry no prefix and are never pruned.
DOMAIN_TOOL_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # HPC cluster control (Slurm / Lmod / allocations) — only for real cluster ops.
    (
        ("slurm_", "salloc_"),
        (
            "slurm", "sbatch", "salloc", "srun", "module", "lmod", "cluster",
            "partition", "hpc", "batch job", "queue", "allocation",
            "compute node", "login node", "supercomputer",
        ),
    ),
    # Hardware / platform advisory.
    (
        ("platform_",),
        (
            "platform", "hardware", "architecture", "microarchitecture",
            "compiler", "cpu", "gpu", "simd", "avx", "vectorize", "vectorise",
            "target", "isa", "numa", "optimize", "optimise", "optimization",
            "optimisation", "performance", "perf", "speedup", "speed up",
            "faster", "accelerate",
        ),
    ),
    # Benchmarking.
    (
        ("benchmark_",),
        (
            "benchmark", "performance", "perf", "speedup", "speed up",
            "optimize", "optimise", "optimization", "optimisation", "faster",
            "accelerate", "profile", "profiling", "flops", "throughput",
            "latency", "measure", "timing",
        ),
    ),
    # Fine-tuning.
    (
        ("ft_",),
        (
            "fine-tune", "finetune", "fine tune", "fine-tuning", "training",
            "train", "lora", "qlora", "sft", "dataset", "epoch", "checkpoint",
            "adapter", "peft",
        ),
    ),
    # Proxy / surrogate model optimization (all seven proxy_* dispatch tools).
    (
        ("proxy_",),
        ("proxy", "surrogate"),
    ),
    # Symbolic mathematics / computer algebra (SymPy). The merged `symbolic` tool is
    # only relevant for algebraic manipulation, calculus, and matrix algebra.
    (
        ("symbolic",),
        (
            "symbolic", "sympy", "algebra", "algebraic", "simplify", "expand",
            "factor", "factorize", "factorise", "derivative", "differentiate",
            "derive", "derivation", "integral", "integrate", "limit", "series",
            "taylor", "maclaurin", "equation", "solve", "matrix", "determinant",
            "polynomial", "prove", "proof", "theorem", "lemma", "complexity",
        ),
    ),
)


# What counts as source: an edit to one of these is recorded as produced work and owes
# a check. Deliberately wider than the set of languages this environment can check —
# whether a checker exists is a separate question, answered per file at write time
# (guardrails.observations._language_checker_missing) so a language nothing here can
# check is *reported* unchecked instead of dropping out of the ledger unnoticed. Every
# spelling of a language belongs here: `.f03` was missing while the checker table
# already knew it, so a Fortran 2003 file was never even recorded as modified.
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
)


# ---------------------------------------------------------------------------
# Query-signal matching. Intent detection matches a *word*, not a raw substring — a
# naive ``token in text`` fires on "create" inside "creative", "add" inside "address".
# Everything goes through ``query_matches_any``, which anchors tokens at ``\b``.
# Multi-word phrases ("speed up") work; ``\w`` is Unicode-aware, so accented French
# tokens ("améliore") get correct boundaries too.
# ---------------------------------------------------------------------------

# Words that negate a following intent token within a short window. Kept minimal and
# bilingual (EN/FR). ``ne``/``n`` cover the French discontinuous "ne … pas"; checking
# the words *before* the match catches the leading half.
_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "without",
    "dont", "don",  # "don't" tokenizes to "don"/"t" under \w+
    "ne", "n", "pas", "sans", "jamais", "aucun", "aucune", "ni",
})

# How many preceding words to scan for a negator before a matched signal token.
_NEGATION_WINDOW = 3


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


def query_has_unnegated_match(query: str, tokens: tuple[str, ...]) -> bool:
    """True if *query* contains at least one *token* not preceded by a negator.

    A match is considered negated when one of ``_NEGATORS`` appears within the
    ``_NEGATION_WINDOW`` words immediately before it — so "ne crée pas de fichier"
    or "without creating" does not count as an active create/edit intent, while
    "crée un fichier" does.
    """
    if not query or not tokens:
        return False
    pattern = _compiled_signal_pattern(tokens)
    if pattern is None:
        return False
    for match in pattern.finditer(query):
        preceding = re.findall(r"\w+", query[: match.start()].lower())
        window = preceding[-_NEGATION_WINDOW:]
        if not any(word in _NEGATORS for word in window):
            return True
    return False


# ── query-intent classifiers ───────────────────────────────────────────────────
# Boolean intent predicates over the raw query. Shared by the guardrail layer and the
# query-engine tool-list construction, hence they live here rather than in either.

def query_is_informational(query: str) -> bool:
    """True when *query* asks for information rather than ordering a change.

    Question-shaped (interrogative opener and/or a trailing "?") **and** free of any
    imperative mutation verb. "what packages do I need?" is informational; "can you
    add a test?" is not, because ``add`` states an action.
    """
    if not query or not query.strip():
        return False
    text = query.strip()
    words = re.findall(r"\w+", text.lower())
    if not words:
        return False
    question_shaped = text.rstrip().endswith("?") or words[0] in _INTERROGATIVE_OPENERS
    if not question_shaped:
        return False
    return not query_has_unnegated_match(text, _ACTION_VERBS)


def _weak_create_match_only(query: str) -> bool:
    """True when the only create signals present are weak ones with no artifact noun.

    A weak signal ("new") is promoted to real create intent when an artifact noun
    follows it within ``_CREATE_ARTIFACT_WINDOW`` words — "a new script" counts,
    "a new machine" does not.
    """
    pattern = _compiled_signal_pattern(QUERY_CREATE_SIGNALS)
    if pattern is None:
        return False
    noun_pattern = _compiled_signal_pattern(_CREATE_ARTIFACT_NOUNS)
    for match in pattern.finditer(query):
        if match.group(0).lower() not in _CREATE_WEAK_SIGNALS:
            return False  # a strong create verb is present
        following = re.findall(r"\w+", query[match.end():].lower())
        window = " ".join(following[:_CREATE_ARTIFACT_WINDOW])
        if noun_pattern is not None and noun_pattern.search(window):
            return False  # weak signal qualifying a writable artifact
    return True


def query_prefers_existing_file_edits(query: str) -> bool:
    if query_is_informational(query):
        return False
    return query_has_unnegated_match(query, QUERY_EDIT_SIGNALS) and not query_has_unnegated_match(
        query, QUERY_CREATE_SIGNALS
    )


def query_requires_repo_discovery(query: str) -> bool:
    return query_matches_any(query, QUERY_DISCOVERY_SIGNALS)


def query_prefers_new_file_creation(query: str) -> bool:
    if query_is_informational(query):
        return False
    if not query_has_unnegated_match(query, QUERY_CREATE_SIGNALS):
        return False
    return not _weak_create_match_only(query)
