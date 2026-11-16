"""Mark the code a harness measures, so MIMIR times and weighs that alone:

    data = make_inputs()          # not measured
    with region():
        out = solve(data)         # measured: time_s, region_peak_rss_mb, region_extra_mb

Outside MIMIR (no ``MIMIR_REGION``) it does nothing. Copy it next to the harness.
"""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import os

_fds: tuple[int, int] | None = None
_off = False
_gpu: tuple[str, ctypes.CDLL] | None | bool = False      # False: not looked for yet


def _channel() -> tuple[int, int] | None:
    global _fds, _off
    path = os.environ.get("MIMIR_REGION")
    if _fds or _off or not path:
        return _fds
    try:
        # Non-blocking open: with no reader (a rank on another node), give up at once.
        out = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        ack = os.open(path + ".ack", os.O_RDONLY | os.O_NONBLOCK)
        fcntl.fcntl(ack, fcntl.F_SETFL, fcntl.fcntl(ack, fcntl.F_GETFL) & ~os.O_NONBLOCK)
        _fds = (out, ack)
    except OSError:
        _off = True
    return _fds


def _loaded(names: tuple[str, ...]) -> ctypes.CDLL | None:
    """A library the process already loaded (CuPy, PyTorch, Numba, a binding...)."""
    for name in names:
        try:
            return ctypes.CDLL(name, mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        except OSError:
            continue
    return None


def _find_gpu() -> tuple[str, ctypes.CDLL] | None:
    lib = _loaded(("libcuda.so.1", "libcuda.so"))
    if lib is not None:
        return "cuda", lib
    lib = _loaded(("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"))
    return ("hip", lib) if lib is not None else None


def _wait_gpu() -> None:
    """Wait for the GPU's work, whatever launched it: this thread's CUDA context, then
    every active primary context; HIP's current device."""
    global _gpu
    if _gpu is False:
        _gpu = _find_gpu()
    if not _gpu:
        return
    kind, lib = _gpu
    try:
        if kind == "hip":
            lib.hipDeviceSynchronize()
            return
        lib.cuCtxSynchronize()
        count = ctypes.c_int(0)
        if lib.cuDeviceGetCount(ctypes.byref(count)):
            return
        release = getattr(lib, "cuDevicePrimaryCtxRelease_v2", None) or lib.cuDevicePrimaryCtxRelease
        for d in range(count.value):
            dev, flags, active, ctx = ctypes.c_int(), ctypes.c_uint(), ctypes.c_int(), ctypes.c_void_p()
            if lib.cuDeviceGet(ctypes.byref(dev), d) or lib.cuDevicePrimaryCtxGetState(
                    dev, ctypes.byref(flags), ctypes.byref(active)) or not active.value:
                continue
            if lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev):
                continue
            if lib.cuCtxPushCurrent_v2(ctx) == 0:
                lib.cuCtxSynchronize()
                lib.cuCtxPopCurrent_v2(ctypes.byref(ctx))
            release(dev)
    except AttributeError:
        pass


def _profiler(word: str) -> None:
    """Under CUDA, the region is the profiler's range: Nsight records it alone."""
    if _gpu and _gpu[0] == "cuda":
        try:
            (_gpu[1].cuProfilerStart if word == "begin" else _gpu[1].cuProfilerStop)()
        except AttributeError:
            pass


def mark(word: str) -> None:
    """Say ``begin`` or ``end``, the GPU's work done first; return once MIMIR has read
    its clocks."""
    fds = _channel()
    if not fds:
        return
    _wait_gpu()
    if word == "end":
        _profiler("end")
    try:
        os.write(fds[0], word.encode() + b"\n")
        os.read(fds[1], 1)
    except OSError:
        pass
    if word == "begin":
        _profiler("begin")


@contextlib.contextmanager
def region():
    """The code in the ``with`` block is what MIMIR measures."""
    mark("begin")
    try:
        yield
    finally:
        mark("end")
