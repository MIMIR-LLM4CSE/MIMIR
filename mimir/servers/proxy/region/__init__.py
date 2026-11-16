"""The helpers a harness marks its measured region with (``_lib/region.py`` reads the
marks): ``mimir_region.py``, ``mimir_region.h`` (C, C++), ``mimir_region.f90`` (Fortran).
They are copied next to a harness, which then needs nothing of MIMIR's to run."""
from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
HELPERS = {"python": "mimir_region.py", "c": "mimir_region.h", "cpp": "mimir_region.h",
           "fortran": "mimir_region.f90"}


def install(dest: str, languages: tuple[str, ...] = ("python",)) -> list[str]:
    """Copy the helpers for *languages* into *dest*, leaving one already there alone;
    the paths of the helpers *dest* now holds."""
    os.makedirs(dest, exist_ok=True)
    placed = []
    for name in dict.fromkeys(HELPERS[lang] for lang in languages if lang in HELPERS):
        target = os.path.join(dest, name)
        if not os.path.exists(target):
            shutil.copyfile(os.path.join(HERE, name), target)
        placed.append(target)
    return placed
