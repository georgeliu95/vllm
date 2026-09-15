# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small tenacity-compatible retry surface used by pinned lm-eval 0.4.10."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
import inspect
import time
from types import SimpleNamespace
from typing import Any, TypeVar

_T = TypeVar("_T")


class RetryError(RuntimeError):
    """Raised after the configured attempts are exhausted."""


@dataclass(frozen=True)
class _StopAfterAttempt:
    attempts: int


@dataclass(frozen=True)
class _WaitExponential:
    multiplier: float
    minimum: float
    maximum: float

    def delay(self, attempt: int) -> float:
        return min(self.maximum, max(self.minimum, self.multiplier * 2 ** (attempt - 1)))


def stop_after_attempt(attempts: int) -> _StopAfterAttempt:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    return _StopAfterAttempt(attempts)


def wait_exponential(
    *, multiplier: float = 1.0, min: float = 0.0, max: float = float("inf")
) -> _WaitExponential:
    if multiplier < 0 or min < 0 or max < min:
        raise ValueError("invalid exponential wait configuration")
    return _WaitExponential(multiplier, min, max)


def retry(
    *,
    stop: _StopAfterAttempt,
    wait: _WaitExponential,
    reraise: bool = False,
    before_sleep: Callable[[Any], object] | None = None,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Retry a synchronous or asynchronous callable on ``Exception``."""

    def decorator(function: Callable[..., _T]) -> Callable[..., _T]:
        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args: object, **kwargs: object) -> Any:
                for attempt in range(1, stop.attempts + 1):
                    try:
                        return await function(*args, **kwargs)
                    except Exception as error:
                        if attempt == stop.attempts:
                            if reraise:
                                raise
                            raise RetryError(str(error)) from error
                        if before_sleep is not None:
                            before_sleep(SimpleNamespace(attempt_number=attempt))
                        await asyncio.sleep(wait.delay(attempt))
                raise AssertionError("unreachable")

            return async_wrapper

        @wraps(function)
        def sync_wrapper(*args: object, **kwargs: object) -> Any:
            for attempt in range(1, stop.attempts + 1):
                try:
                    return function(*args, **kwargs)
                except Exception as error:
                    if attempt == stop.attempts:
                        if reraise:
                            raise
                        raise RetryError(str(error)) from error
                    if before_sleep is not None:
                        before_sleep(SimpleNamespace(attempt_number=attempt))
                    time.sleep(wait.delay(attempt))
            raise AssertionError("unreachable")

        return sync_wrapper

    return decorator


__all__ = ["RetryError", "retry", "stop_after_attempt", "wait_exponential"]
