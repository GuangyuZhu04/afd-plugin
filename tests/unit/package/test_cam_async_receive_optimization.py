# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise production receive helpers with independent asynchronous queues.

The temporary SDK header only declares names needed to parse the native adapter.
It provides no implementation and is never evidence of CANN API support. Actual
MTE2/V/MTE3 work executes through the injected CPU backend. Separate vector jobs
compare the former V_S waits with issue-ahead execution; native SDK compilation
and hardware qualification remain separate checks.
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
def receive_optimization_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("A GNU-compatible C++17 compiler is required for async model tests")
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_receive_optimization")
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
            str(Path(__file__).with_name("cam_async_receive_optimization_probe.cpp")),
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
        "prefetch_fp16",
        "prefetch_bf16",
        "prefetch_partial_hidden",
        "prefetch_large_hidden",
        "prefetch_topk_one",
        "prefetch_topk_one_split",
        "prefetch_topk_eight",
        "prefetch_zero",
        "prefetch_missing_ready",
        "prefetch_unsafe_credit",
        "prefetch_wrong_output_slot",
        "prefetch_missing_tail_drain",
        "prefetch_fused_arithmetic",
        "prefetch_async_fp16",
        "prefetch_async_bf16",
        "prefetch_async_partial_hidden",
        "prefetch_async_topk_eight",
        "prefetch_async_live_scale",
        "prefetch_async_live_address",
        "prefetch_async_scalar_scratch",
        "cache_aa",
        "cache_aba",
        "cache_lru",
        "cache_source_keys",
        "cache_quantized",
        "cache_reset",
        "cache_pending_hit_without_credit",
        "cache_early_scale_credit",
        "cache_missing_reset",
        "cache_missing_tail_drain",
    ),
)
def test_receive_helpers_with_independent_queues(
    receive_optimization_probe: Path, scenario: str
) -> None:
    execution = subprocess.run(
        [str(receive_optimization_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
