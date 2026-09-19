"""Subprocess execution with captured output, timeout, and best-effort peak
memory. Never uses a shell: commands are argv lists, so Windows paths with
spaces are passed through intact.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class ProcessResult:
    returncode: Optional[int]  # None when the process was killed on timeout
    stdout: str
    stderr: str
    wall_ms: float
    timed_out: bool
    peak_rss_bytes: Optional[int]
    peak_rss_note: str
    cancelled: bool = False  # killed because `should_cancel` returned True


_CANCEL_POLL_S = 0.1


def format_command(command: Sequence[str]) -> str:
    """Human-readable, copy-pasteable rendering (display only, never executed)."""
    return subprocess.list2cmdline([str(c) for c in command])


def _peak_rss_windows(handle: int) -> Optional[int]:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not fn(wintypes.HANDLE(handle), ctypes.byref(pmc), pmc.cb):
        return None
    return int(pmc.PeakWorkingSetSize)


def _peak_rss_linux(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _sample_peak(proc: "subprocess.Popen") -> Optional[int]:
    try:
        if sys.platform == "win32":
            handle = getattr(proc, "_handle", None)
            return None if handle is None else _peak_rss_windows(int(handle))
        if sys.platform.startswith("linux"):
            return _peak_rss_linux(proc.pid)
    except Exception:
        return None
    return None


def run_captured(
    command: Sequence[str],
    timeout: Optional[float] = None,
    cwd: Optional[str] = None,
    poll_interval_s: float = 0.05,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> ProcessResult:
    """Run `command` (argv list, no shell), capture stdout/stderr as UTF-8
    (undecodable bytes replaced), and sample the process's peak working set.

    Peak RSS is the OS-reported peak of the launched process only, so it
    includes memory-mapped model pages that were touched. It is `None` with
    an explanatory note on platforms/handles where it cannot be read.
    """
    argv = [str(c) for c in command]
    start = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        cwd=cwd,
        shell=False,
    )

    peak: list[Optional[int]] = [None]
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            v = _sample_peak(proc)
            if v is not None and (peak[0] is None or v > peak[0]):
                peak[0] = v
            stop.wait(poll_interval_s)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    timed_out = False
    cancelled = False
    try:
        if should_cancel is None:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                out, err = proc.communicate()
        else:
            # Poll so a cancellation request can kill the process promptly.
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    out, err = proc.communicate(timeout=_CANCEL_POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    if should_cancel():
                        cancelled = True
                    elif deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                    else:
                        continue
                    proc.kill()
                    out, err = proc.communicate()
                    break
    except BaseException:
        # Never leave an orphaned model process behind on an unexpected error.
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
    final = _sample_peak(proc)  # PeakWorkingSetSize is monotonic; one last read
    if final is not None and (peak[0] is None or final > peak[0]):
        peak[0] = final
    wall_ms = (time.perf_counter() - start) * 1000

    def decode(b: object) -> str:
        return b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else (b or "")

    if peak[0] is None:
        note = "not_available: peak memory could not be read for this process/platform"
    else:
        note = "peak working set of the launched process (includes touched memory-mapped model pages)"
    return ProcessResult(
        returncode=None if (timed_out or cancelled) else proc.returncode,
        stdout=decode(out),
        stderr=decode(err),
        wall_ms=wall_ms,
        timed_out=timed_out,
        peak_rss_bytes=peak[0],
        peak_rss_note=note,
        cancelled=cancelled,
    )
