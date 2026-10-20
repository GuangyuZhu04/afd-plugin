# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run production A5 control decisions over explicit CPU-backed windows.

The memory adapter executes the selector/record/credit code, with abort and
completion inputs supplied by the test. It does not implement CANN events,
remote cache coherence, DMA ordering, or physical task retirement.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def control_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("A C++17 compiler is required for A5 control tests")
    assert compiler is not None
    package_tests = Path(__file__).parent
    root = package_tests.parents[2]
    directory = tmp_path_factory.mktemp("a5_control")
    executable = directory / ("probe.exe" if os.name == "nt" else "probe")
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(package_tests / "a5_control_memory_backend"),
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_kernel"),
            str(package_tests / "cam_async_a5_control_probe.cpp"),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "scenario",
    (
        "cycle",
        "partial_chunk",
        "transport_marker",
        "tp_epoch",
        "credit_exhaustion",
        "abort_selection",
    ),
)
def test_production_a5_control(control_probe: Path, scenario: str):
    result = subprocess.run(
        [str(control_probe), scenario],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"{scenario} passed"
