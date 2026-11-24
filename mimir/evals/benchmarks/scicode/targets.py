"""The expected values of SciCode's tests, read from its ``test_data.h5``.

Each test of a step is the group ``<step>/test<i>``: one entry is the value itself, several
are a tuple in key order. A group entry holds a list (``list``), a sparse matrix
(``sparse_matrix``) or a dict whose keys that read as numbers become floats. Stand-alone,
numpy, scipy and h5py only: it is copied beside each step's test script.
"""
from __future__ import annotations

from typing import Any

import h5py


def _text_or_value(dataset: Any) -> Any:
    value = dataset[()]
    return value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value


def _sparse(group: Any) -> Any:
    import scipy.sparse

    data, shape = group["data"][()], tuple(group["shape"][()])
    if "row" in group and "col" in group:
        return scipy.sparse.coo_matrix((data, (group["row"][()], group["col"][()])), shape=shape)
    if "blocksize" in group:
        return scipy.sparse.bsr_matrix((data, group["indices"][()], group["indptr"][()]),
                                       shape=shape, blocksize=tuple(group["blocksize"][()]))
    return scipy.sparse.csr_matrix((data, group["indices"][()], group["indptr"][()]), shape=shape)


def _dict(group: Any) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for key, entry in group.items():
        if isinstance(entry, h5py.Group):
            out[key] = _sparse(entry["sparse_matrix"])
        elif isinstance(entry[()], bytes):
            out[key] = _text_or_value(entry)
        else:
            try:
                out[float(key)] = entry[()]
            except ValueError:
                out[key] = entry[()]
    return out


def _group(group: Any) -> Any:
    """A group entry, its kind named by its first key."""
    first = next(iter(group.keys()), None)
    if first == "list":
        return [group["list"][key][()] for key in group["list"].keys()]
    if first == "sparse_matrix":
        return _sparse(group["sparse_matrix"])
    return _dict(group)


def _entry(node: Any) -> Any:
    return _text_or_value(node) if isinstance(node, h5py.Dataset) else _group(node)


def targets(path: str, step: str, count: int) -> list[Any]:
    """The expected values of the *count* tests of *step* (``"13.1"``)."""
    values: list[Any] = []
    with h5py.File(path, "r") as fh:
        for i in range(1, count + 1):
            name = f"{step}/test{i}"
            if name not in fh:
                raise KeyError(f"{name} is not in {path}")
            test = fh[name]
            entries = list(test.keys())
            if len(entries) == 1:
                values.append(_entry(test[entries[0]]))
            else:
                values.append(tuple(_entry(test[key]) for key in entries))
    return values
