# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute production UB routing helpers against an independent token-first oracle.

The CPU backend models integer UB atomic additions and writes one contiguous
prefix staging allocation. This verifies algorithm and buffer ownership; it
does not qualify CANN AtomicAdd, VF completion, cache behavior or NPU performance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def simt_routing_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip(
            "A GNU-compatible C++17 compiler is required for SIMT routing tests"
        )
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_simt_routing")
    executable = project / ("probe.exe" if os.name == "nt" else "probe")
    compilation = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_kernel"),
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_host"),
            str(Path(__file__).with_name("cam_async_simt_routing_probe.cpp")),
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
        "dedup",
        "repeated_assignment",
        "rank_63",
        "empty_aivs",
        "tile_reuse",
        "single_aiv",
        "prefix_padding",
        "uint16_boundary",
        "uint16_no_truncation",
        "ids_page_tiles",
        "owner_stride_385",
        "multi_column_300",
        "budget",
    ),
)
def test_production_simt_routing(simt_routing_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(simt_routing_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
