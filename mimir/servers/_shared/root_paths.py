import os

from responses import err


def require_absolute(path: str, root_abs: str, arg: str = "path") -> "dict | None":
    """None when *path* is absolute; an error payload naming the likely target otherwise.

    Tools that name a file take absolute paths only. A relative path has to be resolved
    against a root the model cannot see, and getting that wrong is silent: asked to
    create a file *outside* a directory, a model writes a bare relative name, the server
    resolves it against that very directory, and the run reports the constraint
    satisfied. No amount of prompt text made that inference reliable — removing the
    inference does.

    It applies to reads for the same reason it applies to writes, and one more: a read
    that quietly accepts a relative path teaches the model a habit the next write will
    refuse. Directory and scan roots keep their tolerant default — "." there means the
    workspace, which is not an inference anyone can get wrong.

    The suggestion is the point. Naming the path the relative form *would* have produced
    makes the rejection a one-step correction rather than an obstacle, and it states the
    root at the moment placement is actually being decided, which no static prompt
    section can do.

    Call it at the tool boundary, not inside the resolver: internal helpers legitimately
    pass relative paths.
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
    """Resolve an input path inside a sandbox root with duplicate-root tolerance.

    ``extra_roots`` widens the allowed set: a resolved path that falls under the
    primary root **or** any extra root is accepted. Relative paths still resolve
    against the primary root only; extra roots admit absolute paths in trusted
    locations (e.g. the proxy cache) without loosening the workspace default.
    """
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