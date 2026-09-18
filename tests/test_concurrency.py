"""Tests for the threading helpers shared by the source clients."""

import threading
import time
from itertools import count

import pytest

from streetscapes.sources.common import RateLimiter, concurrent_map


def test_concurrent_map_applies_the_function_to_every_item():
    results = concurrent_map(lambda i: i * 2, range(50), workers=8)

    # The results come back in whatever order they finish in.
    assert sorted(results) == [i * 2 for i in range(50)]


def test_concurrent_map_without_workers_stays_on_the_calling_thread():
    """A single worker means no threads at all, which keeps debugging simple."""
    threads = list(concurrent_map(lambda _: threading.get_ident(), range(5), workers=1))

    assert threads == [threading.get_ident()] * 5


def test_concurrent_map_runs_in_parallel():
    """Ten items that each wait a moment should not take ten moments."""
    start = time.monotonic()
    list(concurrent_map(lambda _: time.sleep(0.1), range(10), workers=10))

    assert time.monotonic() - start < 0.5


def test_concurrent_map_consumes_its_items_lazily():
    """An endless stream of items is fine: only a few are ever in flight."""
    consumed = count()
    results = concurrent_map(lambda i: i, (next(consumed) for _ in count()), workers=4)

    taken = [result for result, _ in zip(results, range(5), strict=False)]

    assert len(taken) == 5
    # The window is a couple of items per worker, nowhere near the whole stream.
    assert next(consumed) < 50


def test_concurrent_map_propagates_exceptions():
    def fail_on_the_last_one(i: int) -> int:
        if i == 9:
            raise ValueError("no")
        return i

    with pytest.raises(ValueError, match="no"):
        list(concurrent_map(fail_on_the_last_one, range(10), workers=4))


def test_rate_limiter_spaces_out_requests():
    limiter = RateLimiter(per_minute=6000)  # one every 10 ms
    assert limiter.interval == pytest.approx(0.01)

    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    elapsed = time.monotonic() - start

    # The first one goes through immediately, the other four have to wait.
    assert elapsed >= 4 * limiter.interval


def test_rate_limiter_spaces_out_threads_too():
    """The quota is shared, so threads queue up behind one another."""
    limiter = RateLimiter(per_minute=1200)  # one every 50 ms

    start = time.monotonic()
    list(concurrent_map(lambda _: limiter.acquire(), range(4), workers=4))
    elapsed = time.monotonic() - start

    assert elapsed >= 3 * limiter.interval


def test_rate_limiter_without_a_limit_never_waits():
    limiter = RateLimiter(per_minute=0)

    start = time.monotonic()
    for _ in range(1000):
        limiter.acquire()

    assert time.monotonic() - start < 0.1
