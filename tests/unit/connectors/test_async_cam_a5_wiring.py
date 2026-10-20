# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute the connector's real routing methods without importing vLLM.

Only runtime dependencies and tensor storage are substituted. The method ASTs
come directly from the production connector, so ticket loss, legacy fallback
and teardown-order regressions affect these tests.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class Tensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "bf16"):
        self.shape = shape
        self.dtype = dtype
        self.device = "npu:0"

    def to(self, *, dtype: str):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self

    def new_empty(self, shape: tuple[int, ...]):
        return Tensor(shape, self.dtype)


class State(SimpleNamespace):
    transport_ticket: Tensor | None = None


class NativeContext:
    def __init__(self, events: list[str]):
        self.events = events
        self.calls: list[tuple] = []
        self.sent_ticket = Tensor((128,), "fp16")
        self.received_ticket = Tensor((128,), "fp16")
        self.metadata = Tensor((11,), "int64")

    def dispatch_send(self, hidden: Tensor, ids: Tensor, layer: int):
        self.calls.append(("dispatch", hidden, ids, layer))
        return self.sent_ticket

    def combine_recv(self, ids: Tensor, weights: Tensor, ticket: Tensor):
        self.calls.append(("combine_recv", ids, weights, ticket))
        return Tensor((ids.shape[0], 64))

    def dispatch_recv(self):
        self.calls.append(("receive",))
        return [
            Tensor((16, 64)),
            Tensor((1,), "fp32"),
            self.metadata,
            Tensor((2,), "int64"),
            self.received_ticket,
        ]

    def combine_send(self, hidden: Tensor, metadata: Tensor, ticket: Tensor):
        self.calls.append(("combine_send", hidden, metadata, ticket))

    def quiesce(self):
        self.events.append("quiesce")

    def release_after_group_destroyed(self):
        self.events.append("release")


@pytest.fixture
def connector_type():
    root = Path(__file__).resolve().parents[3]
    source = root / "afd_plugin/connectors/npu/async_cam.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    connector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CAMAsyncAFDConnector"
    )
    methods = {
        "send_attn_output",
        "recv_ffn_output",
        "recv_attn_output",
        "send_ffn_output",
        "_require_initialized",
        "close",
    }
    connector.bases = []
    connector.body = [
        node
        for node in connector.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_require_async_transfer_state", "_validate_topk_payload"}
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            connector,
            *helpers,
        ],
        type_ignores=[],
    )
    namespace = {
        "AFDAsyncTransferState": State,
        "AFDTransferContext": SimpleNamespace,
        "AFDA2FTransferPayload": SimpleNamespace,
        "AFDTransferMetadata": SimpleNamespace(
            create_ffn_metadata=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
        "torch": SimpleNamespace(int32="int32", float32="fp32"),
        "_log_cam_op_values": lambda *args, **kwargs: None,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["CAMAsyncAFDConnector"]


@pytest.fixture
def connector(connector_type):
    instance = connector_type()
    instance._initialized = True
    instance.hidden_size = 64
    instance.topk = 2
    instance.max_num_batched_tokens = 16
    instance.tp_size = 2
    instance.ffn_size = instance.attn_size = instance.expert_per_rank = 2
    instance.world_rank = 0
    instance.topology = SimpleNamespace(world_size=4)
    instance.comm_args = Tensor((1,), "fp16")
    instance.comm_id = 0
    instance.dynamic_quant = 0
    instance.group_name = "test_a5_domain"
    instance._placeholder = Tensor((1,))
    instance.layered_gmm_enabled = False
    instance._pending_attention_payloads = {}
    instance.cam_pg = "owned_group"
    instance._a5_context = NativeContext([])
    return instance


def test_attention_routes_original_ticket_to_native_combine(connector):
    metadata = SimpleNamespace(
        total_tokens=2,
        layer_idx=7,
        stage_idx=0,
        validate_tensor_shape=lambda shape: shape == (2, 64),
    )
    context = SimpleNamespace(metadata=metadata)
    ids, weights = Tensor((2, 2), "int32"), Tensor((2, 2), "fp32")
    native = connector._a5_context
    connector.send_attn_output(
        Tensor((2, 64)), context, topk_ids=ids, topk_weights=weights
    )
    assert context.states.transport_ticket is native.sent_ticket
    output = connector.recv_ffn_output(Tensor((2, 64)), context=context)
    assert output.shape == (2, 64)
    assert native.calls[-1] == ("combine_recv", ids, weights, native.sent_ticket)
    assert connector._pending_attention_payloads == {}


def test_ffn_preserves_complete_metadata_and_ticket(connector):
    native = connector._a5_context
    received = connector.recv_attn_output()
    state = received.context.states
    assert state.token_nums_rankid_layeridx is native.metadata
    assert state.transport_ticket is native.received_ticket
    output = Tensor((4, 64))
    connector.send_ffn_output(output, received.context)
    assert native.calls[-1] == (
        "combine_send",
        output,
        native.metadata,
        native.received_ticket,
    )


def test_missing_ticket_cannot_fall_back_to_legacy_send(connector):
    received = connector.recv_attn_output()
    received.context.states.transport_ticket = None
    with pytest.raises(RuntimeError, match="missing its receive ticket"):
        connector.send_ffn_output(Tensor((4, 64)), received.context)
    assert connector._a5_context.calls == [("receive",)]


def test_registration_owners_outlive_group_teardown(connector, monkeypatch):
    native = connector._a5_context
    torch = ModuleType("torch")
    distributed = ModuleType("torch.distributed")
    distributed.__dict__["destroy_process_group"] = lambda group: native.events.append(
        "destroy"
    )
    torch.__dict__["distributed"] = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    connector.close()
    assert native.events == ["quiesce", "destroy", "release"]
    assert connector._a5_context is None
    assert connector.cam_pg is None


def test_failed_quiescence_keeps_group_and_registration_owners(connector, monkeypatch):
    native = connector._a5_context

    def fail():
        raise RuntimeError("device work is not quiescent")

    monkeypatch.setattr(native, "quiesce", fail)
    with pytest.raises(RuntimeError, match="not quiescent"):
        connector.close()
    assert connector.cam_pg == "owned_group"
    assert connector._a5_context is native
    assert native.events == []
