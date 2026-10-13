# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute P4 geometry, immutable-record codec, schema and exact-lease checks.

These tests check field positions and disjoint layout spans. They do not test
raw routing, payload transfer, atomic snapshots, visibility or credit transport.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def slot_wire_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip(
            "A GNU-compatible C++17 compiler is required for the slot wire probe"
        )
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_slot_wire")
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
            str(Path(__file__).with_name("cam_async_slot_wire_probe.cpp")),
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


@pytest.mark.parametrize("scenario", ("geometry", "rejections", "codec", "bad_records"))
def test_slot_wire_codec(slot_wire_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(slot_wire_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
