"""Circuit breaker (step 4 of the flow, see __init__.py).

Think of the fuse box in a house: if something keeps failing, the fuse "trips" so you stop
wasting effort. Here: if the AI service fails several times IN A ROW (default 3), we stop
calling it and send the remaining tickets straight to people. A single success resets the count.
"""
from __future__ import annotations


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures and stays open for the rest of the
    run (no half-open probing: a batch is short-lived; restart to retry). None disables it."""

    def __init__(self, failure_threshold: int | None = 3) -> None:
        self._threshold = failure_threshold
        self._consecutive_failures = 0

    @property
    def is_open(self) -> bool:
        return self._threshold is not None and self._consecutive_failures >= self._threshold

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
