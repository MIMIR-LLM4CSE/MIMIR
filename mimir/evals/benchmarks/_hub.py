"""What the adapters of published benchmarks share: their rows from Hugging Face, and
their ``--adapter-arg`` strings read as flags."""
from __future__ import annotations

from typing import Any


def load_rows(dataset: str, split: str) -> list[dict[str, Any]]:
    """The rows of *dataset*'s *split* (``pip install 'mimir-agent[bench]'``)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(f"{dataset} needs the `datasets` package: "
                         "pip install 'mimir-agent[bench]'") from exc
    return [dict(row) for row in load_dataset(dataset, split=split)]


def flag(value: Any) -> bool:
    """An ``--adapter-arg`` value read as a yes or no."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
