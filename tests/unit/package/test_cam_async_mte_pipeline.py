# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute the production slot lifecycle with asynchronous CPU pipeline queues.

The SDK declarations below only let an ordinary C++ compiler parse the CANN
adapter. The tests inject a stateful backend into the production PipelineEvents
class; independent FIFO queues execute its events in varying topological orders.
Actual CANN API support still requires a build against the installed toolkit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SDK_DECLARATIONS = """\
#pragma once
#include <cstdint>
#define __aicore__
namespace AscendC {
using TEventID = int32_t;
enum class HardEvent { MTE2_MTE3, MTE3_MTE2, MTE2_S };
class TPipe {
public:
    template <HardEvent event> TEventID AllocEventID() { return 0; }
    template <HardEvent event> void ReleaseEventID(TEventID) {}
    template <HardEvent event> TEventID FetchEventID() { return 0; }
};
template <HardEvent event> void SetFlag(int32_t) {}
template <HardEvent event> void WaitFlag(int32_t) {}
}
"""


@pytest.fixture(scope="module")
def mte_pipeline_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip(
            "A GNU-compatible C++17 compiler is required for MTE lifecycle tests"
        )
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_mte_pipeline")
    (project / "kernel_operator.h").write_text(_SDK_DECLARATIONS, encoding="utf-8")
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
            str(Path(__file__).with_name("cam_async_mte_pipeline_probe.cpp")),
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
        "zero",
        "one",
        "cell_boundaries",
        "odd_tile_tail",
        "quantized",
        "reuse_rounds",
        "missing_reuse_wait",
        "early_quant_credit",
        "missing_scalar_fence",
    ),
)
def test_production_mte_lifecycle(mte_pipeline_probe: Path, scenario: str):
    execution = subprocess.run(
        [str(mte_pipeline_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
