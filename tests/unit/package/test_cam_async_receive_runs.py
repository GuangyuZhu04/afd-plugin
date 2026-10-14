# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run production receive planners/cursors/events against a raw byte oracle.

The temporary SDK header contains declarations only. Its sole purpose is to
parse the unused native adapter; the injected three-FIFO backend executes the
real PipelineEvents template. SDK compilation and NPU execution are separate.
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
    template <HardEvent event> TEventID AllocEventID();
    template <HardEvent event> void ReleaseEventID(TEventID);
    template <HardEvent event> TEventID FetchEventID();
};
template <HardEvent event> void SetFlag(int32_t);
template <HardEvent event> void WaitFlag(int32_t);
}
"""


@pytest.fixture(scope="module")
def receive_runs_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("A GNU-compatible C++17 compiler is required for receive-run tests")
    root = Path(__file__).resolve().parents[3]
    project = tmp_path_factory.mktemp("cam_async_receive_runs")
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
            str(Path(__file__).with_name("cam_async_receive_runs_probe.cpp")),
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
        "q0_runs",
        "q1_runs",
        "partial_runs",
        "empty_experts",
        "empty_aiv",
        "aiv_boundaries",
        "random_fallback",
        "spare_fallback",
        "peer_isolation",
        "wire_wrap",
        "unsafe_reuse",
        "early_scale_credit",
        "wrong_padding_stride",
        "wrong_scale_stride",
        "missing_tail_drain",
    ),
)
def test_shared_receive_run_contract(receive_runs_probe: Path, scenario: str) -> None:
    execution = subprocess.run(
        [str(receive_runs_probe), scenario],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stdout.strip() == f"{scenario} passed"
