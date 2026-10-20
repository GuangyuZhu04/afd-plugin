# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise the production setup handshake without a CANN runtime or an NPU."""

from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from afd_plugin.compat.npu.cam_async_a5 import (
    SETUP_NAMESPACE,
    initialize_async_cam_a5_context,
)

WORLD_SIZE = 4
STORE_TIMEOUT_SECONDS = 5


class MemoryStore:
    def __init__(self):
        self.values: dict[str, bytes] = {}
        self.condition = threading.Condition()

    def set(self, key: str, value: str | bytes) -> None:
        with self.condition:
            self.values[key] = value.encode() if isinstance(value, str) else value
            self.condition.notify_all()

    def wait(self, keys: list[str]) -> None:
        with self.condition:
            if not self.condition.wait_for(
                lambda: all(key in self.values for key in keys),
                timeout=STORE_TIMEOUT_SECONDS,
            ):
                raise TimeoutError("setup handshake did not make progress")

    def get(self, key: str) -> bytes:
        self.wait([key])
        with self.condition:
            return self.values[key]


class Ledger:
    def __init__(self):
        self.lock = threading.Lock()
        self.prepared: set[int] = set()
        self.zeroed: set[int] = set()
        self.registered: set[int] = set()


class Context:
    def __init__(self, rank: int, ledger: Ledger):
        self.rank = rank
        self.ledger = ledger

    def prepare(self) -> list[int]:
        with self.ledger.lock:
            self.ledger.prepared.add(self.rank)
        # Negative signed words deliberately preserve uint64 token/hash bits
        # across the Python JSON exchange, without float conversions.
        return [1024 + self.rank, 64 * 1024 * 1024, -42, 53, -91, self.rank]

    def configure(self, resources: list[list[int]]) -> None:
        with self.ledger.lock:
            assert self.ledger.prepared == set(range(WORLD_SIZE))
            assert [row[-1] for row in resources] == list(range(WORLD_SIZE))
            assert all(row[2] == -42 and row[4] == -91 for row in resources)
            self.ledger.zeroed.add(self.rank)

    def initialize(self) -> None:
        with self.ledger.lock:
            assert self.ledger.zeroed == set(range(WORLD_SIZE))
            self.ledger.registered.add(self.rank)


def install_store(monkeypatch: pytest.MonkeyPatch, store: MemoryStore) -> None:
    # Only the Store lookup dependency is replaced; the three production
    # handshake phases and their ordering execute unchanged.
    module = ModuleType("torch.distributed.distributed_c10d")
    module.__dict__["_get_process_group_store"] = lambda group: store
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_all_ranks_finish_each_setup_phase(monkeypatch: pytest.MonkeyPatch):
    store = MemoryStore()
    ledger = Ledger()
    install_store(monkeypatch, store)

    def run(rank: int) -> None:
        initialize_async_cam_a5_context(
            Context(rank, ledger), None, rank=rank, world_size=WORLD_SIZE
        )
        with ledger.lock:
            assert ledger.registered == set(range(WORLD_SIZE))

    with ThreadPoolExecutor(max_workers=WORLD_SIZE) as executor:
        list(executor.map(run, reversed(range(WORLD_SIZE))))
    assert len(store.values) == WORLD_SIZE * 3


def test_failed_configuration_does_not_publish_initialized(
    monkeypatch: pytest.MonkeyPatch,
):
    store = MemoryStore()
    install_store(monkeypatch, store)

    class FailingContext:
        def prepare(self) -> list[int]:
            return [1024, 4096, 3, 53, 91, 0]

        def configure(self, resources: list[list[int]]) -> None:
            raise RuntimeError("invalid registered span")

        def initialize(self) -> None:
            raise AssertionError("registration after failed frame initialization")

    with pytest.raises(RuntimeError, match="invalid registered span"):
        initialize_async_cam_a5_context(FailingContext(), None, rank=0, world_size=1)
    assert list(store.values) == [f"{SETUP_NAMESPACE}/resource/0"]


def test_context_module_import_does_not_load_device_libraries():
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import afd_plugin.compat.npu.cam_async_a5; "
            "assert 'torch' not in sys.modules; "
            "assert 'torch_npu' not in sys.modules; "
            "assert 'vllm' not in sys.modules",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
