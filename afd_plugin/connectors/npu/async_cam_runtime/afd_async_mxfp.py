# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: DeepSeek-V4-Flash MXFP4 experts on Ascend 950 UMDK slices.
# Create: 2026-11-17
# Note: The native MXFP4 expert method derives its block scales from the received BF16
#       rows.
# History: 2026-11-17 create
# Adapted from the A5 eager executor in afd-plugin commit bd1e4d28271a.
"""Compute the (layer, local expert) groups selected by a UMDK transaction."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch_npu

if TYPE_CHECKING:
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4MoE

    from afd_plugin.connectors.npu.async_cam_runtime.umdk_layout import LayerRows


def mxfp4_grouped_matmul(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Multiply native FP8 activations by post-load packed FP4 expert weights."""
    # torch-npu infers E4M3FN from x. Its x_dtype override is for packed FP4
    # and HIFLOAT8. Supplying E4M3FN as an override fails on CANN 9.2.
    return torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        antiquant_scale=[weight_scale],
        per_token_scale=[x_scale],
        group_list=counts,
        split_item=2,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.bfloat16,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
    )[0]


class AfdAsyncMXFPExecutor:
    """Select the received layer and run its native, routed-only W4A8MXFP MLP."""

    def __init__(self, layers: dict[int, DeepseekV4MoE], local_experts: int) -> None:
        from vllm_ascend.quantization.quant_type import QuantType

        if not layers:
            raise ValueError("Ascend 950 Async CAM requires loaded MXFP4 expert layers")
        self.layers = layers
        self.local_experts = local_experts
        signatures = []
        for index, moe in sorted(layers.items()):
            experts = moe.experts
            owner = experts.routed_experts
            activation = (
                owner.activation.value
                if isinstance(owner.activation, Enum)
                else owner.activation
            )
            if (
                experts.quant_type != QuantType.W4A8MXFP
                or owner.dynamic_eplb
                or experts._shared_experts is not None
                or activation != "silu"
                or owner.quant_method.quant_method.group_size != 32
            ):
                raise ValueError(
                    f"Ascend 950 Async CAM layer {index} requires static routed-only "
                    "W4A8MXFP SiLU experts with group size 32"
                )
            # Native post-load may retain only Mega's per-expert lists. Eager GMM
            # consumes the finalized non-Mega tensors and their block scales.
            geometry = []
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                weight = owner._parameters.get(name)
                if (
                    weight is None
                    or not weight.numel()
                    or weight.shape[0] != local_experts
                ):
                    raise ValueError(
                        f"Ascend 950 Async CAM layer {index} requires non-Mega "
                        f"{name} for {local_experts} local experts"
                    )
                geometry.append((tuple(weight.shape), str(weight.dtype)))
            signatures.append(
                (
                    index,
                    geometry,
                    str(owner.activation),
                    experts.moe_config.swiglu_limit,
                )
            )
        self.signature = "umdk_mxfp4:" + repr(signatures)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        groups: tuple[LayerRows, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2
            or output.shape != hidden_states.shape
            or output.device != hidden_states.device
            or output.dtype != torch.bfloat16
        ):
            raise ValueError(
                "UMDK MXFP4 requires matching BF16 receive and return slices"
            )
        cursor = 0
        for group in groups:
            end = group.end
            if group.layer_id not in self.layers:
                raise ValueError(f"UMDK selected an unknown FFN layer {group.layer_id}")
            if (
                len(group.counts) != self.local_experts
                or any(n < 0 for n in group.counts)
                or group.begin != cursor
                or end > hidden_states.shape[0]
            ):
                raise ValueError("UMDK expert counts do not match the physical slice")
            if end == cursor:
                continue
            rows = hidden_states[cursor:end]
            counts = torch.tensor(group.counts, dtype=torch.int64, device=rows.device)
            value = self._run_layer(self.layers[group.layer_id], rows, counts)
            target = output[cursor:end]
            if (
                value.shape != target.shape
                or value.dtype != target.dtype
                or value.device != target.device
            ):
                raise ValueError(
                    "MXFP4 output does not match the registered return rows"
                )
            target.copy_(value)
            cursor = end
        if cursor != output.size(0):
            raise ValueError("UMDK expert groups do not cover the physical slice")
        return output

    @staticmethod
    def _run_layer(
        moe: DeepseekV4MoE, rows: torch.Tensor, group_list: torch.Tensor
    ) -> torch.Tensor:
        from vllm_ascend.ops.fused_moe.moe_utils import (
            maybe_normalize_mxfp_scale_layout,
        )

        # Matches 7dcdb690's W4A8MXFP SiLU path. UMDK supplies BF16, and
        # group_list_type=1 already gives the counts required by SwiGLU.
        owner = moe.experts.routed_experts
        x, scale = torch_npu.npu_dynamic_mx_quant(rows, dst_type=torch.float8_e4m3fn)
        gate_up = mxfp4_grouped_matmul(
            x,
            maybe_normalize_mxfp_scale_layout(scale),
            owner.w13_weight,
            owner.w13_weight_scale,
            group_list,
        )
        activation, scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
            gate_up,
            topk_weight=None,
            group_index=group_list,
            dst_type=torch.float8_e4m3fn,
            quant_mode=2,
            clamp_value=moe.experts.moe_config.swiglu_limit or 0.0,
        )
        return mxfp4_grouped_matmul(
            activation,
            maybe_normalize_mxfp_scale_layout(scale),
            owner.w2_weight,
            owner.w2_weight_scale,
            group_list,
        )
