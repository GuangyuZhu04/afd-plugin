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

from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.quantization.quant_type import QuantType

if TYPE_CHECKING:
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4MoE

    from afd_plugin.connectors.npu.async_cam_runtime.umdk_layout import LayerRows


class AfdAsyncMXFPExecutor:
    """Select the received layer and run its native, routed-only W4A8MXFP MLP."""

    def __init__(self, layers: dict[int, DeepseekV4MoE], local_experts: int) -> None:
        if not layers:
            raise ValueError("Ascend 950 Async CAM requires loaded MXFP4 expert layers")
        self.layers = layers
        self.local_experts = local_experts
        signatures = []
        for index, moe in sorted(layers.items()):
            experts = moe.experts
            owner = experts.routed_experts
            if (
                experts.quant_type != QuantType.W4A8MXFP
                or owner.dynamic_eplb
                or experts._shared_experts is not None
            ):
                raise ValueError(
                    f"Ascend 950 Async CAM layer {index} requires static routed-only "
                    "W4A8MXFP experts"
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
            # The native method can dispose its input via Tensor.set_(). Give
            # it a separate view; the owner retains the registered allocation.
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
        experts = moe.experts
        owner, config = experts.routed_experts, experts.moe_config
        # See vllm-ascend bd42eea0's w4a8_mxfp4.py::apply_gmm1_act_quant.
        # With dynamic_scale=None the native method derives MXFP8 block scales
        # from the BF16 UMDK rows. Counts include each expert's alignment padding.
        output, _ = apply_moe_mlp(
            mlp_compute_input=MoEMlpComputeInput(
                hidden_states=rows,
                group_list=group_list,
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                weights=MoEWeights(
                    w1=owner.w13_weight,
                    w2=owner.w2_weight,
                    w1_scale=owner.w13_weight_scale,
                    w2_scale=owner.w2_weight_scale,
                ),
                layer=owner,
                quant=MoEQuantParams(
                    quant_type=QuantType.W4A8MXFP,
                    mxfp=MoEMxfpParams(
                        act_quant_type=torch.float8_e4m3fn,
                        weight_quant_type=torch_npu.float4_e2m1fn_x2,
                        scale_dtype=torch_npu.float8_e8m0fnu,
                        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                        use_bf16=True,
                    ),
                ),
                activation=owner.activation,
                swiglu_limit=config.swiglu_limit or 0.0,
                swiglu_alpha=config.swiglu_alpha or 1.0,
                swiglu_beta=config.swiglu_beta or 0.0,
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
                fusion=False,
            ),
            quant_method=owner.quant_method.quant_method,
        )
        # Attention's top-k weights carry the routed scaling factor. CR applies it.
        return output
