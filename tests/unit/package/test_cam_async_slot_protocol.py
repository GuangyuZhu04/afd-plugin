# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute the serialized slot reference against observable ownership invariants.

The CPU probe uses separate source, wire and owned backing allocations. It
verifies bounded credit, exact lease identity, quiescent recovery and completed
logical-order FP32 reduction. Proof values are explicit test inputs; no SDK
completion, remote visibility, GM atomic publication or NPU execution is modeled.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def slot_protocol_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip(
            "A GNU-compatible C++17 compiler is required for the slot protocol probe"
        )
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_slot_protocol")
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
            str(Path(__file__).with_name("cam_async_slot_protocol_probe.cpp")),
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
        "config_schema",
        "proof_boundaries",
        "source_ownership",
        "delayed_credit",
        "retirement_barrier",
        "lease_isolation",
        "empty_chunk",
        "ordered_capture",
        "reduction_completion",
        "combine_tickets",
        "abort_recovery",
        "gate_abort",
        "epoch_exhaustion",
    ),
)
def test_production_slot_protocol(slot_protocol_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(slot_protocol_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
