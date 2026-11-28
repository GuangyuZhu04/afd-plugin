# SPDX-License-Identifier: MIT
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Host tests for registered-buffer receive/compute/return and MXFP4 layer
#              selection.
# Create: 2026-11-23
# Note: The native expert compute boundary is stubbed for host execution.
# History: 2026-11-23 add to the Ascend 950 port
"""Host checks for eager MXFP4 execution on UMDK transaction slices."""

from types import SimpleNamespace

import pytest

from afd_plugin.connectors.npu.async_cam_runtime.umdk_layout import LayerRows
from tests.unit.fakes.async_ffn import (
    eager_mxfp_runner,
    load_async_modules,
    mxfp_runtime,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def runtime(monkeypatch):
    _, _, runner = load_async_modules(monkeypatch)
    executor, layers = mxfp_runtime(monkeypatch, torch)
    return runner, executor, layers


def test_mixed_layer_slice_and_empty_slice_return_registered_rows(monkeypatch, runtime):
    runner, _, layers = runtime
    hidden = (2.0 ** (torch.arange(16) % 4)).bfloat16().view(8, 2)
    original = hidden.clone()
    output = torch.full_like(hidden, -1)
    groups = (LayerRows(3, 0, (2, 2)), LayerRows(7, 4, (0, 4)))

    def packet(index, rows, work):
        state = runner.UmdkSlice(transaction=19, index=index, groups=work)
        context = SimpleNamespace(
            metadata=SimpleNamespace(total_tokens=len(rows)), states=state
        )
        return SimpleNamespace(hidden_states=rows, context=context)

    packets = [packet(0, hidden, groups), packet(1, hidden[:0], ())]
    returns = []
    model = eager_mxfp_runner(
        monkeypatch,
        torch,
        runner,
        layers,
        packets,
        output,
        lambda rows, ctx: returns.append((rows, ctx)),
    )
    assert model.execute_connector_driven_step()
    assert model.execute_connector_driven_step()
    assert model.execute_connector_driven_step()  # idle poll
    assert not model.execute_connector_driven_step(shutdown_requested=True)
    assert [ctx for _, ctx in returns] == [p.context for p in packets]
    assert all(
        rows.is_set_to(output[: len(p.hidden_states)])
        for (rows, _), p in zip(returns, packets, strict=True)
    )
    torch.testing.assert_close(hidden, original)
    factors = torch.tensor([3, 3, 4, 4, 8, 8, 8, 8], dtype=torch.bfloat16)
    torch.testing.assert_close(output, original * factors[:, None])


@pytest.mark.parametrize(
    "groups,error",
    [
        ((LayerRows(1, 0, (2, 1)),), "unknown FFN layer"),
        ((LayerRows(3, 0, (2, 4)),), "physical slice"),
        ((LayerRows(7, 0, (2,)),), "physical slice"),
        ((LayerRows(7, 0, (2, 0)),), "cover the physical slice"),
    ],
)
def test_950_rejects_inconsistent_slice(runtime, groups, error):
    _, executor, layers = runtime
    hidden = torch.ones((4, 2), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=error):
        executor(layers, 2)(hidden, groups, torch.empty_like(hidden))


@pytest.mark.parametrize("bad", ["quant", "shared", "mega", "group_size", "activation"])
def test_950_rejects_unsupported_expert_weights_at_startup(runtime, bad):
    _, executor, layers = runtime
    experts = layers[7].experts
    if bad == "quant":
        experts.quant_type = "w4a8"
    elif bad == "shared":
        experts._shared_experts = object()
    elif bad == "group_size":
        experts.routed_experts.quant_method.quant_method.group_size = 64
    elif bad == "activation":
        experts.routed_experts.activation = "gelu"
    else:
        experts.routed_experts.w13_weight = None
    with pytest.raises(ValueError, match="layer 7 requires"):
        executor(layers, 2)
