# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run loaded quantized expert weights on the physical rows of a UMDK slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from afd_plugin.model_executor.ffn.npu.expert_weights import (
    QuantizedExpertWeights,
    validate_expert_weights,
)

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.async_cam_runtime.umdk_layout import LayerRows


class UmdkExpertExecutor:
    def __init__(self, layers: list[QuantizedExpertWeights]):
        self.layers = {
            layer.layer_idx: layer for layer in validate_expert_weights(layers)
        }

    def __call__(
        self, hidden: torch.Tensor, groups: tuple[LayerRows, ...], output: torch.Tensor
    ) -> torch.Tensor:
        import torch_npu

        for group in groups:
            layer = self.layers[group.layer_id]
            counts = torch.tensor(group.counts, dtype=torch.int64, device=hidden.device)
            rows = hidden[group.begin : group.end]
            x, scale = torch_npu.npu_dynamic_quant(rows)
            if layer.quantization == "W8A8":
                activation, act_scale, *_ = (
                    torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                        x,
                        [layer.w13],
                        [layer.w13_scale],
                        scale,
                        counts,
                        dequant_mode=0,
                        dequant_dtype=torch.float32,
                        quant_mode=0,
                        group_list_type=1,
                    )
                )
            else:
                gate_up = self._matmul(
                    x, scale, counts, layer.w13, layer.w13_scale, layer.w13_bias
                )
                if layer.swiglu_limit > 0:
                    gate, up = gate_up.chunk(2, dim=-1)
                    gate_up = torch.cat(
                        (
                            gate.clamp(max=layer.swiglu_limit),
                            up.clamp(-layer.swiglu_limit, layer.swiglu_limit),
                        ),
                        dim=-1,
                    )
                gated = torch_npu.npu_swiglu(gate_up, dim=-1)
                activation, act_scale = torch_npu.npu_dynamic_quant(gated)
            value = self._matmul(
                activation, act_scale, counts, layer.w2, layer.w2_scale, layer.w2_bias
            )
            if layer.routed_scaling_factor != 1.0:
                value.mul_(layer.routed_scaling_factor)
            output[group.begin : group.end].copy_(value)
        return output

    @staticmethod
    def _matmul(x, scale, counts, weight, weight_scale, bias):
        import torch_npu

        # CANN expects an explicit K-group axis for packed INT4 per-channel scales.
        if weight.dtype == torch.int32 and weight_scale.ndim == 2:
            weight_scale = weight_scale.unsqueeze(1)
        return torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[weight],
            bias=[bias] if bias is not None else None,
            scale=[weight_scale],
            per_token_scale=[scale],
            group_list=counts,
            split_item=3,
            group_type=0,
            group_list_type=1,
            output_dtype=torch.bfloat16,
        )[0]
