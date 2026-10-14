# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validate production tile layout, route capture, cursor and leased events.

Independent raw routing establishes occurrence rows and FP32/RNE goldens. Three
asynchronous CPU queues expose late arguments and unsafe descriptor/slot reuse.
SDK declarations only parse native adapters; no device execution is simulated.
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
enum class HardEvent {
    MTE2_V, V_MTE2, V_MTE3, MTE3_V, MTE2_MTE3, MTE3_MTE2, MTE2_S, V_S
};
class TPipe {
public:
    template <HardEvent event> TEventID AllocEventID();
    template <HardEvent event> void ReleaseEventID(TEventID);
    template <HardEvent event> TEventID FetchEventID();
};
template <HardEvent event> void SetFlag(int32_t);
template <HardEvent event> void WaitFlag(int32_t);
}
"""


@pytest.fixture(scope="module")
def combine_tile_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("A GNU-compatible C++17 compiler is required for tile tests")
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_combine_tile_plan")
    (project / "kernel_operator.h").write_text(_SDK_DECLARATIONS, encoding="utf-8")
    executable = project / ("probe.exe" if os.name == "nt" else "probe")
    compilation = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            "-I",
            str(project),
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_kernel"),
            str(Path(__file__).with_name("cam_async_combine_tile_plan_probe.cpp")),
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
        "fp16",
        "bf16",
        "prefix",
        "odd_tail",
        "wide_hidden",
        "topk_one",
        "topk_eight",
        "late_dma",
        "late_scale",
        "missing_token_retirement",
        "early_output_credit",
        "missing_tail_drain",
        "reverse_k",
        "fused",
    ),
)
def test_production_combine_tile_plan(combine_tile_probe: Path, scenario: str) -> None:
    execution = subprocess.run(
        [str(combine_tile_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
