# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend 950 compute handoff through the actual torch-npu Meta operators."""

import importlib
import os

import pytest

if not os.environ.get("SOC_VERSION", "").lower().startswith(("950", "ascend950")):
    pytest.skip("UMDK FFN is built for Ascend 950", allow_module_level=True)

torch = importlib.import_module("torch")
importlib.import_module("torch_npu")


@pytest.mark.parametrize("quantization", ["W8A8", "W4A8"])
def test_mixed_layer_result_keeps_the_caller_output(quantization):
    from afd_plugin.connectors.npu.async_cam_runtime.umdk_experts import (
        UmdkExpertExecutor,
    )
    from afd_plugin.connectors.npu.async_cam_runtime.umdk_layout import LayerRows
    from afd_plugin.model_executor.ffn.npu.expert_weights import QuantizedExpertWeights

    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    experts, width = 2, 256
    packed = quantization == "W4A8"
    div = 8 if packed else 1
    weights = []
    for layer in (2, 5):
        weights.append(
            QuantizedExpertWeights(
                layer_idx=layer,
                w13=tensor(
                    (experts, width, 2 * width // div),
                    torch.int32 if packed else torch.int8,
                ),
                w2=tensor(
                    (experts, width, width // div),
                    torch.int32 if packed else torch.int8,
                ),
                w13_scale=tensor(
                    (experts, 2 * width), torch.int64 if packed else torch.float32
                ),
                w2_scale=tensor(
                    (experts, 1, width) if packed else (experts, width),
                    torch.int64 if packed else torch.bfloat16,
                ),
                w13_bias=tensor((experts, 2 * width), torch.float32)
                if packed
                else None,
                w2_bias=tensor((experts, width), torch.float32) if packed else None,
                per_channel=True,
                swiglu_limit=10.0,
                routed_scaling_factor=0.5,
                quantization=quantization,
            )
        )
    executor = UmdkExpertExecutor(weights)
    hidden = tensor((64, width), torch.bfloat16)
    out = torch.empty_like(hidden)
    result = executor(
        hidden, (LayerRows(2, 0, (0, 32)), LayerRows(5, 32, (32, 0))), out
    )
    assert result is out
