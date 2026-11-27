# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Optional runtime setup shared by the compiled Ascend dispatcher tests.
# Create: 2026-11-05
# Note: Compact CAM belongs to 910C. Ascend 950 has separate UMDK Meta tests.
# History: 2026-11-05 create
"""Optional runtime setup shared by the compiled Ascend dispatcher tests."""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import pytest

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded, uses_umdk_async


def _load_compiled_ops() -> ModuleType:
    for module in ("torch", "torch_npu", "afd_plugin._C_ascend"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for compiled Ascend dispatcher tests")
    # Import only when a dispatcher test requests this fixture; host-side
    # harness tests and CPU collection must not load the optional runtime.
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    ensure_afd_ascend_ops_loaded()
    return torch


@pytest.fixture(scope="module")
def umdk_meta_runtime() -> None:
    # A 950 Meta run needs the packaged dispatcher, but no DeepEP owner or NPU.
    # Fail a requested 950 build if the installed wheel supplies another backend.
    importlib.import_module("torch_npu")
    ensure_afd_ascend_ops_loaded()
    if not uses_umdk_async():
        pytest.fail("SOC_VERSION=950 requires the plugin's registered-buffer extension")


@pytest.fixture(scope="module")
def ascend_ops_runtime() -> ModuleType:
    soc = os.environ.get("SOC_VERSION", "").lower()
    if soc != "910c" and not soc.startswith("ascend910_93"):
        pytest.skip("Set SOC_VERSION=910c to test the compiled 910C operators")
    return _load_compiled_ops()


@pytest.fixture(scope="module")
def afd_async_ops_runtime() -> ModuleType:
    soc = os.environ.get("SOC_VERSION", "").lower()
    if soc != "910c" and not soc.startswith("ascend910_93"):
        pytest.skip("Compact CAM is built for 910C; 950 uses the UMDK Meta suites")
    return _load_compiled_ops()
