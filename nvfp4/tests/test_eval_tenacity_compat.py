# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
from pathlib import Path
import sys


COMPAT_ROOT = Path(__file__).resolve().parents[1] / "evaluation"
SPEC = importlib.util.spec_from_file_location(
    "_eval_tenacity_compat", COMPAT_ROOT / "tenacity.py"
)
assert SPEC is not None and SPEC.loader is not None
TENACITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TENACITY
SPEC.loader.exec_module(TENACITY)
retry = TENACITY.retry
stop_after_attempt = TENACITY.stop_after_attempt
wait_exponential = TENACITY.wait_exponential


def test_retry_supports_sync_callables() -> None:
    calls = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0, min=0, max=0),
        reraise=True,
    )
    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("retry")
        return "ok"

    assert flaky() == "ok"
    assert calls == 3


def test_retry_supports_async_callables() -> None:
    calls = 0

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0, min=0, max=0),
        reraise=True,
    )
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise RuntimeError("retry")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert calls == 2
