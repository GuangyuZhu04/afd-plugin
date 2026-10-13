# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute the checked CCU descriptor planner; no CCU runtime is simulated.

The oracle rebuilds expert/TP/token return rows from raw routing and compares
all copied bytes and untouched window guards. It does not establish memory
registration, CCU remote completion, device submission or stream semantics.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ccu_segment_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip(
            "A GNU-compatible C++17 compiler is required for the CCU planner probe"
        )
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_ccu")
    executable = project / ("probe.exe" if os.name == "nt" else "probe")
    compilation = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "csrc/npu/experimental/cam_async_ccu"),
            str(Path(__file__).with_name("cam_async_ccu_probe.cpp")),
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
        "partial_chunk",
        "last_chunk",
        "split_aivs",
        "empty_final",
        "empty_nonfinal",
        "rejections",
        "wide_offsets",
    ),
)
def test_production_ccu_segment_plan(ccu_segment_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(ccu_segment_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
