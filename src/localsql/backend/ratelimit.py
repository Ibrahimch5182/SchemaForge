"""Tiny in-process sliding-window rate limiter for the public demo.

Per-client (IP) limit on POST /query only. State is process-local and bounded
(oldest keys are evicted), which is enough for a single-host demo; it is not a
distributed limiter and not an authentication mechanism.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0, *, max_keys: int = 10_000, clock: Callable[[], float] = time.monotonic):
        if max_requests < 1 or window_seconds <= 0 or max_keys < 1:
            raise ValueError("invalid limiter settings")
        self.max_requests = max_requests
        self.window = window_seconds
        self._max_keys = max_keys
        self._clock = clock
        self._hits: "OrderedDict[str, deque[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            q = self._hits.pop(key, None) or deque()
            while q and now - q[0] >= self.window:
                q.popleft()
            allowed = len(q) < self.max_requests
            retry = 0
            if allowed:
                q.append(now)
            else:
                retry = max(1, math.ceil(self.window - (now - q[0])))
            self._hits[key] = q  # most-recently-used at the end
            while len(self._hits) > self._max_keys:
                self._hits.popitem(last=False)
            return allowed, retry


def client_key(peer_host: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """The client identity used for limiting. With `trusted_hops` reverse proxies
    in front, the address they appended is `X-Forwarded-For[-hops]`; anything
    further left is client-supplied and never trusted. With 0 hops the header is
    ignored entirely (it would be spoofable)."""
    if trusted_hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= trusted_hops:
            return parts[-trusted_hops]
    return peer_host or "unknown"
