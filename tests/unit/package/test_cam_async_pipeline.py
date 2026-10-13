# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute the shared P3 prefix/descriptor loops against raw-route byte oracles.

This tests deterministic integer planning, slot ownership and bounds. It does
not emulate SIMT scheduling, device memory ordering, CCU or a completion bridge.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pipeline_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("A GNU-compatible C++17 compiler is required for the P3 probe")
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_pipeline")
    executable = project / ("probe.exe" if os.name == "nt" else "probe")
    compilation = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "csrc/npu/experimental/cam_async_pipeline"),
            str(Path(__file__).with_name("cam_async_pipeline_probe.cpp")),
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
        "multi_tile",
        "empty_slots",
        "sparse_peer",
        "bounded_slots",
        "wide_prefixes",
        "uneven_lanes",
        "rank_rejection",
        "budget_boundaries",
    ),
)
def test_shared_pipeline_plan(pipeline_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(pipeline_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
