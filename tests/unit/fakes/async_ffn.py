# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Load Async FFN host orchestration without device/runtime dependencies.
# Create: 2026-10-22
# Note: Host fixtures isolate upstream runtime imports and native compute.
# History: 2026-10-22 create
"""Load Async FFN host orchestration without device/runtime dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def load_host_module(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[3] / (name.replace(".", "/") + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        # A cached package attribute otherwise keeps the previous module alive.
        monkeypatch.setitem(vars(parent), child_name, module)
    return module


def load_async_modules(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        module.__dict__["__getattr__"] = lambda attr: MagicMock(name=attr)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    npu = MagicMock()
    npu.get_stream_limit.return_value = {"cube_core_num": 24, "vector_core_num": 48}
    stub("torch", inference_mode=lambda: lambda function: function, npu=npu)
    for name in (
        "torch.distributed",
        "torch.distributed.distributed_c10d",
        "torch.distributed.rendezvous",
        "vllm.logger",
        "vllm.distributed",
        "vllm.utils.torch_utils",
        "vllm_ascend.utils",
        "vllm.v1.worker.gpu.shutdown",
        "afd_plugin.compat.npu.forward_context",
        "afd_plugin.compat.npu.profiler",
        "afd_plugin.connectors.npu.async_cam_runtime.async_cam_layered",
        "afd_plugin.connectors.npu.async_cam_runtime.async_cam_mega",
        "afd_plugin.model_executor.ffn.npu.expert_weights",
        "afd_plugin.v1.worker.ffn_metadata",
    ):
        stub(name)
    stub("afd_plugin.compat.npu.ops", uses_umdk_async=lambda: False)
    stub("vllm.forward_context")
    load_host_module(monkeypatch, "afd_plugin.connectors.metadata")
    load_host_module(monkeypatch, "afd_plugin.connectors.base")
    # These modules capture torch at import time. Scope their module entries
    # and parent attributes to the same fixture as the substituted runtime.
    load_host_module(monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.io_log")
    load_host_module(
        monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.umdk_session"
    )
    load_host_module(
        monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.umdk_experts"
    )
    load_host_module(monkeypatch, "tests.unit.fakes.connector")
    group = load_host_module(monkeypatch, "afd_plugin.distributed.afd_process_group")
    stub("afd_plugin.distributed", init_afd_process_group=group.init_afd_process_group)
    connector = load_host_module(monkeypatch, "afd_plugin.connectors.npu.async_cam")
    runtime = load_host_module(
        monkeypatch, "afd_plugin.v1.worker.npu.async_ffn_model_runner"
    )
    return group, connector, runtime


def mxfp_runtime(monkeypatch, torch):
    """Isolate the native MX kernels while running layer/row selection on CPU."""

    def quantize(rows, *, dst_type):
        return rows.to(dst_type), rows.new_ones((len(rows), 1))

    def grouped_matmul(
        *,
        x,
        weight,
        antiquant_scale,
        per_token_scale,
        group_list,
        split_item,
        group_type,
        group_list_type,
        output_dtype,
        weight_dtype,
        per_token_scale_dtype,
    ):
        assert group_list_type == 1 and x[0].dtype == torch.float8_e4m3fn
        gains = torch.repeat_interleave(weight[0], group_list)
        return [(x[0].to(output_dtype) * gains[:, None]).to(output_dtype)]

    def swiglu(rows, *, topk_weight, group_index, dst_type, quant_mode, clamp_value):
        assert topk_weight is None and quant_mode == 2
        quant, scales = quantize(rows, dst_type=dst_type)
        return quant, scales, None

    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_swiglu_group_quant", swiglu, raising=False
    )
    modules: dict[str, dict[str, object]] = {
        "torch_npu": dict(
            float4_e2m1fn_x2="fp4",
            float8_e8m0fnu="e8m0",
            npu_dynamic_mx_quant=quantize,
            npu_grouped_matmul=grouped_matmul,
        ),
        "vllm_ascend.ops.fused_moe.moe_utils": dict(
            maybe_normalize_mxfp_scale_layout=lambda s: s
        ),
        "vllm_ascend.quantization.quant_type": dict(
            QuantType=SimpleNamespace(W4A8MXFP="fp4")
        ),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    executor = load_host_module(
        monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.afd_async_mxfp"
    ).AfdAsyncMXFPExecutor
    config = SimpleNamespace(
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    layers = {}
    for index in (3, 7):
        owner = torch.nn.Module()
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            weight = torch.tensor([index, index + 1], dtype=torch.bfloat16)
            if name == "w2_weight":
                weight = torch.ones_like(weight)
            owner.register_parameter(
                name, torch.nn.Parameter(weight, requires_grad=False)
            )
        owner.dynamic_eplb = False
        owner.activation = "silu"
        owner.quant_method = SimpleNamespace(
            quant_method=SimpleNamespace(group_size=32)
        )
        layers[index] = SimpleNamespace(
            experts=SimpleNamespace(
                quant_type="fp4",
                routed_experts=owner,
                _shared_experts=None,
                moe_config=config,
            )
        )
    return executor, layers


def eager_mxfp_runner(monkeypatch, torch, runtime, layers, payloads, registered, send):
    """Start the eager runner with a host UMDK queue and registered return buffer."""
    from afd_plugin.compat.npu import umdk_mxfp_stack
    from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
        AsyncCAMGraphStartup,
        resolve_ffn_mode,
    )
    from tests.unit.fakes.vllm_config import npu_vllm_config

    class Model:
        def get_mxfp_expert_layers(self):
            return layers

        def get_experts_layer_indices(self):
            return tuple(layers)

    module = ModuleType("afd_plugin.model_executor.models.npu.deepseek_v4")
    module.__dict__["AFDDeepseekV4ForCausalLM"] = Model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # The fixture provides the native SwiGLU operation on CPU.
    monkeypatch.setattr(
        umdk_mxfp_stack, "load_umdk_ffn_extension", lambda: "cpu-fixture"
    )
    graph_startup = AsyncCAMGraphStartup(1, 1, 2)
    transport = runtime.UmdkSession(
        graph_startup, SimpleNamespace(attn_size=1), 2, 1, 32, 1, 32
    )
    transport.owner = SimpleNamespace(combine_x=registered)
    queued = iter(payloads)

    def receive(*, wait):
        payload = next(queued, None)
        transport.current_slice = None if payload is None else payload.context
        return payload

    monkeypatch.setattr(transport, "receive", receive)
    connector = SimpleNamespace(
        dynamic_quant=0,
        world_rank=1,
        expert_per_rank=2,
        is_initialized=False,
        graph_startup=graph_startup,
        ffn_session=lambda: transport,
        init_afd_connector=lambda: resolve_ffn_mode([graph_startup.plan]),
        send_ffn_output=send,
    )
    cfg = npu_vllm_config(
        role="ffn",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
    )
    session = runtime.AsyncNPUFFNModelRunner(cfg, torch.device("cpu"), connector)
    session.attach_model(Model(), parameter_bytes=0)
    session.initialize_afd_connector()
    return session
