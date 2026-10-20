# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute the runtime frame contract without claiming NPU completion evidence."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def full_profile_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the A5 frame contract")
    assert compiler is not None
    root = Path(__file__).resolve().parents[3]
    directory = tmp_path_factory.mktemp("a5_full_profile")
    executable = directory / ("probe.exe" if os.name == "nt" else "probe")
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_kernel"),
            str(Path(__file__).with_name("cam_async_a5_full_probe.cpp")),
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
        "geometry",
        "bounds",
        "schema",
        "publication",
        "credit",
        "descriptors",
        "tags",
        "multiply",
    ),
)
def test_full_profile_contract(full_profile_probe: Path, scenario: str):
    result = subprocess.run(
        [str(full_profile_probe), scenario],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"{scenario} passed"
