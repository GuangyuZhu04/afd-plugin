# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute DS's shared sparse rank tracker against an independent raw-ID oracle.

The probe checks payload uniqueness/order, all assignment address bytes, offsets,
bounded visited-list work and shape admission. Wire packets are opaque fixtures;
no quantization math, DMA completion, NPU timing or new protocol is simulated.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sparse_rank_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip(
            "A GNU-compatible C++17 compiler is required for the sparse rank probe"
        )
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path_factory.mktemp("cam_async_sparse_ranks") / (
        "probe.exe" if os.name == "nt" else "probe"
    )
    compilation = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_kernel"),
            str(Path(__file__).with_name("cam_async_sparse_ranks_probe.cpp")),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    return executable


@pytest.mark.parametrize(
    "scenario",
    (
        "roundtrip",
        "quantized_wire",
        "owned_slice",
        "empty_slice",
        "rank_one",
        "rank_62",
        "k_more_ranks",
        "full_rank_set",
        "stale_tail",
        "mask_rank_63",
        "invalid_config",
        "invalid_rank",
        "uint16_boundary",
        "invalid_slice",
        "early_advance_mutation",
        "missing_reset_mutation",
    ),
)
def test_shared_sparse_payload_ranks(sparse_rank_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(sparse_rank_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
