import os

from ...shared.responses import err


def require_absolute(path: str, root_abs: str, arg: str = "path") -> "dict | None":
    """None for an absolute *path*; else an error payload naming the absolute path it likely meant.

    Tools that name a file take absolute paths only: resolving a relative one against an
    unseen root fails silently. Call it at the tool boundary.
    """
    raw = "" if path is None else str(path).strip()
    if not raw:
        return err(f"Missing '{arg}'. This tool requires an absolute path.")
    if os.path.isabs(os.path.expanduser(raw)):
        return None
    root = os.path.abspath(root_abs)
    candidate = os.path.abspath(os.path.join(root, os.path.normpath(raw)))
    return err(
        f"Relative {arg} '{raw}' — this tool requires an absolute path.",
        hint=(f"Inside the workspace that is {candidate}. If you meant somewhere else, "
              f"give that absolute path instead. The workspace root is {root}."),
    )


def resolve_path_in_root(
    path: str, root_abs: str, outside_error_label: str,
    extra_roots: "list[str] | tuple[str, ...] | None" = None,
) -> str:
    """Resolve *path* inside the sandbox root, or any of *extra_roots* for absolute paths."""
    root = os.path.abspath(root_abs)
    raw = "." if path is None else str(path).strip()
    if not raw:
        raw = "."

    normalized = os.path.normpath(os.path.expanduser(raw))
    root_base = os.path.basename(root)

    if not os.path.isabs(normalized):
        probe = os.path.abspath(os.path.join(root, normalized))
        if normalized == root_base and not os.path.exists(probe):
            normalized = "."
        else:
            prefix = root_base + os.sep
            if normalized.startswith(prefix):
                stripped = normalized[len(prefix):]
                stripped_probe = os.path.abspath(os.path.join(root, stripped or "."))
                if not os.path.exists(probe) and os.path.exists(stripped_probe):
                    normalized = stripped or "."

    full = os.path.abspath(normalized) if os.path.isabs(normalized) else os.path.abspath(os.path.join(root, normalized))
    full = os.path.realpath(full)
    root = os.path.realpath(root)

    allowed = [root]
    for extra in (extra_roots or ()):
        if extra:
            allowed.append(os.path.realpath(os.path.abspath(os.path.expanduser(extra))))
    for base in allowed:
        try:
            if os.path.commonpath([base, full]) == base:
                return full
        except ValueError:
            continue  # different drive / uncomparable — not this root
    raise ValueError(f"Path '{path}' is outside the allowed {outside_error_label}.")