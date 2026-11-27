# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A rejected single-round send must surface before a blocking receive."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit.fakes.async_ffn import load_host_module


@pytest.mark.parametrize(
    ("sync_error", "message"),
    [(None, "state=3, error=4, round=0"), (RuntimeError("507035"), "507035")],
)
def test_failed_single_round_reports_source_before_waiting_for_return(
    sync_error, message, monkeypatch
):
    torch = pytest.importorskip("torch")
    from tests.unit.fakes.dp_metadata import DPMetadata

    monkeypatch.setitem(
        sys.modules, "vllm.forward_context", SimpleNamespace(DPMetadata=DPMetadata)
    )
    load_host_module(monkeypatch, "afd_plugin.connectors.metadata")
    runtime = load_host_module(
        monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.umdk_session"
    )

    topology = SimpleNamespace(world_rank=3, attn_size=4)
    session = runtime.UmdkSession(None, topology, 4096, 6, 4096, 2, 16384)
    context = SimpleNamespace(metadata=SimpleNamespace(stage_idx=0, layer_idx=7))
    session.pending = SimpleNamespace(
        rounds=1,
        context=context,
        sent=SimpleNamespace(synchronize=Mock(side_effect=sync_error)),
    )
    session.owner = SimpleNamespace(
        send_status=torch.tensor([3, 4, 0], dtype=torch.int32)
    )
    session.cfg = {}
    with pytest.raises(
        RuntimeError, match=rf"DispatchSend.*rank 3, layer 7.*{message}"
    ):
        session.combine(0, context)
