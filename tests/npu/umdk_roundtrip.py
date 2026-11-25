# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend 950 only. Run with torchrun --standalone --nproc-per-node=4."""

import os
from datetime import timedelta
from time import monotonic

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 -- register the NPU and HCCL backends
from deep_ep.utils.envs import get_hccl_group_name

from afd_plugin.compat.npu.ops import ensure_afd_async_ops_available, uses_umdk_async
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam import AFDAsyncTopology
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
    AsyncCAMGraphStartup,
    FFNGraphPlan,
)
from afd_plugin.connectors.npu.async_cam_runtime.umdk_session import (
    UmdkSession,
    UmdkSlice,
)

SOURCES, RECEIVERS, LOCAL_EXPERTS, HIDDEN = 2, 2, 2, 256
LAYERS = (2, 5)
STEPS = 3


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    ensure_afd_async_ops_available()
    if not uses_umdk_async():
        raise RuntimeError(
            "Build the Ascend 950 registered-buffer plugin for this test"
        )
    dist.init_process_group("hccl", timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != SOURCES + RECEIVERS:
        raise ValueError("The reference checks a 2A2F world")
    is_source = rank < SOURCES
    role_rank = rank if is_source else rank - SOURCES
    topology = AFDAsyncTopology(
        "attention" if is_source else "ffn",
        role_rank,
        rank,
        SOURCES,
        RECEIVERS,
        LOCAL_EXPERTS,
    )
    startup = AsyncCAMGraphStartup(rank, SOURCES, world)
    startup.plan = FFNGraphPlan(signature="identity-by-expert", layer_ids=LAYERS)
    store = dist.distributed_c10d._get_default_store()
    startup.initialize(store, "umdk_v2:2a2f:bf16")
    session = UmdkSession(startup, topology, HIDDEN, 2, 65, 1, 64)
    device = f"npu:{local}"
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    session.initialize(device, group_name)
    dist.barrier()
    for step in range(STEPS):
        keys = [
            f"umdk-smoke/{startup.session_epoch}/{step}/{a}" for a in range(SOURCES)
        ]
        if is_source:
            layer = LAYERS[(rank + step) % len(LAYERS)]
            n = 65 if rank == 0 else (0 if step == 1 else 7)
            context = AFDTransferContext(AFDTransferMetadata(layer, step, [n]))
            x = (
                torch.arange(n * HIDDEN, device=device).reshape(n, HIDDEN) % 7
            ).bfloat16()
            first = torch.arange(n, device=device) % LOCAL_EXPERTS
            second = (first + 1) % LOCAL_EXPERTS if step == 2 else first + LOCAL_EXPERTS
            ids = torch.stack((first, second), dim=1)
            weights = (
                torch.tensor([0.25, 0.75], device=device).expand(n, 2).contiguous()
            )
            session.dispatch(context, x, ids, weights, n)
            actual = session.combine(step, context)
            factor = ((2 + layer + ids).float() * weights).sum(dim=1, keepdim=True)
            torch.testing.assert_close(
                actual, (x.float() * factor).bfloat16(), rtol=0, atol=0
            )
            store.set(keys[rank], "done")
        else:
            deadline = monotonic() + 180
            while session.continuation or not store.check(keys):
                if monotonic() >= deadline:
                    raise TimeoutError("2A2F numerical roundtrip timed out")
                payload = session.receive(wait=False)
                if payload is None:
                    continue
                state = payload.context.states
                assert isinstance(state, UmdkSlice)
                out = session.result_buffer(payload.context)
                for group in state.groups:
                    row = group.begin
                    for expert, count in enumerate(group.counts):
                        end = row + count
                        coefficient = (
                            2 + group.layer_id + role_rank * LOCAL_EXPERTS + expert
                        )
                        out[row:end].copy_(payload.hidden_states[row:end] * coefficient)
                        row = end
                session.send_ffn(out, payload.context)
        dist.barrier()
    session.prepare_close(True)
    dist.destroy_process_group()
    if rank == 0:
        print("UMDK Ascend 950 2A2F numerical roundtrip passed", flush=True)


if __name__ == "__main__":
    main()
