import os


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