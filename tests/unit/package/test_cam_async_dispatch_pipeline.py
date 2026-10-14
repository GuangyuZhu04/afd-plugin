# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute production DS buffer/event helpers with raw-ID and opaque-wire oracles.

FIFO delays test ownership and capture boundaries; this does not simulate original
quantization arithmetic, CANN transport visibility or measured NPU overlap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

KERNEL_STUB = """
#pragma once
#include <cstdint>
#define __aicore__
namespace AscendC {
enum class HardEvent { MTE2_V, MTE2_MTE3, MTE3_MTE2, MTE2_S, V_MTE3, S_MTE3 };
using TEventID = int;
class TPipe {
public:
    template <HardEvent E> TEventID AllocEventID();
    template <HardEvent E> void ReleaseEventID(TEventID);
    template <HardEvent E> TEventID FetchEventID();
};
template <HardEvent E> void SetFlag(int);
template <HardEvent E> void WaitFlag(int);
}
"""


@pytest.fixture(scope="module")
def dispatch_pipeline_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the DS pipeline probe")
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_dispatch_pipeline")
    (project / "kernel_operator.h").write_text(KERNEL_STUB, encoding="utf-8")
    executable = project / ("probe.exe" if os.name == "nt" else "probe")
    compilation = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(project),
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_kernel"),
            str(Path(__file__).with_name("cam_async_dispatch_pipeline_probe.cpp")),
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
        "planning",
        "raw_multi_peer",
        "quantized_multi_peer",
        "owned_slice",
        "early_credit",
        "capture",
        "metadata_fence",
        "scratch",
        "tail_drain",
        "borrowed_ready",
    ),
)
def test_dispatch_token_pipeline(dispatch_pipeline_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(dispatch_pipeline_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
