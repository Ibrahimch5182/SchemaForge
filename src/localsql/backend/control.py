"""Cooperative cancellation and bounded inference concurrency.

Deliberately tiny and thread-based (the API layer runs the sync pipeline in a
worker thread): no async machinery, nothing that needs a broker.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from localsql.backend.errors import ModelBusyError, RequestCancelledError


class CancelToken:
    """Set once by whoever wants the request to stop; polled by long stages."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise RequestCancelledError("The request was cancelled.")


class InferenceGate:
    """At most `max_concurrent` model processes run at once; at most
    `max_waiting` further requests may queue, each for at most
    `wait_timeout_s`. Anything beyond that is rejected immediately with
    `model_busy` instead of piling up multi-GB processes."""

    def __init__(self, max_concurrent: int = 1, max_waiting: int = 2, wait_timeout_s: float = 20.0):
        if max_concurrent < 1 or max_waiting < 0 or wait_timeout_s < 0:
            raise ValueError("invalid gate limits")
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.wait_timeout_s = wait_timeout_s
        self._cond = threading.Condition()
        self._running = 0
        self._waiting = 0

    def snapshot(self) -> dict:
        with self._cond:
            running, waiting = self._running, self._waiting
        state = "idle" if running == 0 else ("saturated" if waiting >= self.max_waiting else "busy")
        return {
            "state": state,
            "running": running,
            "waiting": waiting,
            "max_concurrent": self.max_concurrent,
            "max_waiting": self.max_waiting,
        }

    @contextmanager
    def slot(self, cancel: Optional[CancelToken] = None) -> Iterator[None]:
        with self._cond:
            if cancel is not None:
                cancel.raise_if_cancelled()
            if self._running >= self.max_concurrent or self._waiting > 0:
                if self._waiting >= self.max_waiting:
                    raise ModelBusyError("The model is busy. Try again shortly.")
                self._waiting += 1
                deadline = time.monotonic() + self.wait_timeout_s
                try:
                    while self._running >= self.max_concurrent:
                        if cancel is not None and cancel.cancelled:
                            raise RequestCancelledError("The request was cancelled.")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ModelBusyError("The model is busy. Try again shortly.")
                        self._cond.wait(min(remaining, 0.1))
                finally:
                    self._waiting -= 1
            self._running += 1
        try:
            yield
        finally:
            with self._cond:
                self._running -= 1
                self._cond.notify_all()
