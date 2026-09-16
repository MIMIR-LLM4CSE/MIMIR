"""How far along a build says it is, read from the output it prints.

Parsed, never tracked: the only process that knows the answer is the build tool, and
the one thing it shares is its output. So the percentage is the tool's own count or
nothing — a build that prints none gets no bar, since showing 0% for it would invent
a fact.

Read from the output rather than the command line, which is what makes it work inside
a long shell chain: whatever surrounds ``make`` (an ``export``, a ``cd``, a ``&&``
tail), its lines still land in the same log.

Formats recognized, each the tool's default:

    [ 42%] Building CXX object ...        CMake Makefiles, fpm
    [12/345] Building CXX object ...      ninja, meson
    [1,234 / 5,678] Compiling ...         bazel

A plain Makefile, pdflatex or latexmk print no count at all, and get no bar.
"""

from __future__ import annotations

import os
import re

# Anchored at the start of a line: a compiler diagnostic quoting "[3/4]" mid-sentence
# is not a progress report.
_PERCENT = re.compile(r"^\s*\[\s*(\d{1,3})%\]")
_RATIO = re.compile(r"^\s*\[\s*([\d,]+)\s*/\s*([\d,]+)\s*\]")

# How much of a log a progress read looks at. Not a diagnosis size: this runs every
# second or so, and the newest count is always in the last few lines.
TAIL_BYTES = 8 * 1024

# Longest phase text passed on. A CMake line names an object path that can run to
# hundreds of characters; the row it lands in has room for one line.
_PHASE_MAX = 160


def _percent_of(line: str) -> float | None:
    m = _PERCENT.match(line)
    if m:
        return float(min(100, int(m.group(1))))
    m = _RATIO.match(line)
    if m:
        done = int(m.group(1).replace(",", ""))
        total = int(m.group(2).replace(",", ""))
        if total > 0 and done <= total:
            return round(100.0 * done / total, 1)
    return None


def parse(text: str, drop_finished: bool = True) -> tuple[float, str] | None:
    """The newest (percent, line) in *text*, or None when there is none to report.

    A finished build is not a build in progress: in ``make && ./bench`` the last
    count is ``[100%]`` for as long as the benchmark runs. So a 100% that is no
    longer the last line is dropped, rather than leaving a full bar standing over
    work it does not describe. A caller that already knows the build is still the
    current step (the proxy runner's phase says so) passes ``drop_finished=False``.
    A lower count stays shown under the warnings a compiler prints between two
    counts — that build is still going.
    """
    if not text:
        return None
    # splitlines() also breaks on \r, so a redrawn status line counts as its states.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        percent = _percent_of(line)
        if percent is None:
            continue
        if drop_finished and percent >= 100 and i != len(lines) - 1:
            return None
        phase = line.strip()
        if len(phase) > _PHASE_MAX:
            phase = phase[:_PHASE_MAX - 1] + "…"
        return percent, phase
    return None


def read_tail(path: str, max_bytes: int = TAIL_BYTES) -> str:
    """The last *max_bytes* of *path*, or '' if it cannot be read."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size <= max_bytes:
                return fh.read().decode("utf-8", errors="replace")
            fh.seek(size - max_bytes)
            data = fh.read()
    except OSError:
        return ""
    # The seek lands mid-line; a fragment like "5/345] ..." must not read as a count.
    _, _, rest = data.partition(b"\n")
    return rest.decode("utf-8", errors="replace")


def from_file(path: str, drop_finished: bool = True) -> tuple[float, str] | None:
    """:func:`parse` applied to the end of the log at *path*."""
    return parse(read_tail(path), drop_finished)
