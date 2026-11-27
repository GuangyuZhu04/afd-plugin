# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise the plugin's registered-buffer transport with an exact BF16 oracle."""

import argparse
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from time import monotonic

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 -- register the NPU and HCCL backends
from deep_ep.utils.envs import get_hccl_group_name
from torch.distributed.elastic.multiprocessing.errors import record

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


@dataclass(frozen=True)
class Profile:
    sources: int
    receivers: int
    tp: int
    hidden: int
    experts: int
    topk: int
    capacity: int
    backend_rows: int
    layers: tuple[int, ...]


SMALL = Profile(2, 2, 1, 256, 4, 2, 65, 64, (2, 5))
FLASH = Profile(4, 4, 2, 4096, 256, 6, 4096, 262144, tuple(range(43)))


def check_last_dispatch_records(session, ids, weights):
    """Compare the sender's last published BF16 wire records before slot reuse."""
    rows, k = ids.shape
    if not rows:
        return
    begin = ((rows - 1) // session.window) * session.window
    # AFDWorkspace reserves int32 route slots for C_m*K, followed by records.
    # TokenLayout aligns the IDs and the end of each metadata record to 32 B.
    metadata_offset = (4 * session.window * k + 31) // 32 * 32
    weight_offset = (8 * k + 31) // 32 * 32
    token_offset = weight_offset + 4 * k
    stride = (token_offset + 8 + 31) // 32 * 32
    count = rows - begin
    actual = session.owner.workspace.narrow(0, metadata_offset, count * stride).cpu()
    expected = torch.zeros((count, stride), dtype=torch.uint8)
    expected[:, : 8 * k] = ids[begin:].cpu().contiguous().view(torch.uint8)
    expected[:, weight_offset:token_offset] = (
        weights[begin:].cpu().contiguous().view(torch.uint8)
    )
    expected[:, token_offset : token_offset + 4] = (
        torch.arange(count, dtype=torch.int32).view(count, 1).view(torch.uint8)
    )
    mask = (1 << k) - 1 if k < 32 else -1
    expected[:, token_offset + 4 : token_offset + 8] = torch.full(
        (count, 1), mask, dtype=torch.int32
    ).view(torch.uint8)
    actual = actual.view_as(expected)
    # The alignment gaps have no wire value. MTE can fill them with a byte
    # from the source burst (observed for K=6 on 3510).
    for fields in (slice(0, 8 * k), slice(weight_offset, token_offset + 8)):
        torch.testing.assert_close(
            actual[:, fields], expected[:, fields], rtol=0, atol=0
        )


def source_round(session, profile, rank, step, rows, device):
    dp_rank, tp_rank = divmod(rank, profile.tp)
    # The FFN scheduler gathers every TP member of a DP at the same layer,
    # including members that publish zero rows in the first communication round.
    layer = (
        profile.layers[0]
        if step == 0
        else profile.layers[(dp_rank + step) % len(profile.layers)]
    )
    n = rows if step == 0 or rank == 0 else (0 if step == 1 else min(rows, 7))
    context = AFDTransferContext(AFDTransferMetadata(layer, step, [n]))
    token = torch.arange(n, device=device)[:, None]
    slot = torch.arange(profile.topk, device=device)
    # The profile pass uses the same balanced expert IDs as the model's dummy gate.
    domain = profile.experts // profile.receivers if step == 2 else profile.experts
    ids = (token * profile.topk + slot) % domain
    weights = ((slot + 1).float() / 16).expand(n, -1).contiguous()
    if step:
        # Distinguish a later round from the same local rows in round zero.
        weights = weights + (token % 7).float() / 32
    column = torch.arange(profile.hidden, device=device)
    # X in {-1, 0, 1} times an integer gain <= 251 is exactly representable in BF16.
    x = ((token * 7 + column + rank + step) % 17 % 3 - 1).bfloat16()
    print(
        f"rank={rank} step={step} layer={layer} valid_rows={n} dispatch "
        f"dp={dp_rank} tp_member={tp_rank}",
        flush=True,
    )
    session.dispatch(context, x, ids, weights, n)
    actual = session.combine(step, context)
    check_last_dispatch_records(session, ids, weights)
    gains = 1 + (layer + ids) % 251
    expected = (x.float() * (gains * weights).sum(dim=1, keepdim=True)).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def serve_round(session, store, keys, rank, step, local_experts):
    deadline = monotonic() + 180
    while session.continuation or not store.check(keys):
        if monotonic() >= deadline:
            raise TimeoutError(f"FFN rank {rank} timed out in round {step}")
        payload = session.receive(wait=False)
        if payload is None:
            continue
        state = payload.context.states
        assert isinstance(state, UmdkSlice)
        out = session.result_buffer(payload.context)
        print(
            f"ffn={rank} step={step} tx={state.transaction} slice={state.index} "
            f"physical_rows={len(out)}",
            flush=True,
        )
        for group in state.groups:
            row = group.begin
            for expert, count in enumerate(group.counts):
                if count:
                    end = row + count
                    gain = 1 + (group.layer_id + rank * local_experts + expert) % 251
                    out[row:end].copy_(payload.hidden_states[row:end] * gain)
                    row = end
        session.send_ffn(out, payload.context)


@record
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("small", "dsv4-flash"), default="small")
    parser.add_argument(
        "--rows", type=int, help="first-step rows per source, up to profile capacity"
    )
    args = parser.parse_args()
    profile = SMALL if args.profile == "small" else FLASH
    rows = profile.capacity if args.rows is None else args.rows
    if not 1 <= rows <= profile.capacity:
        parser.error(f"--rows must be in 1..{profile.capacity}")
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    ensure_afd_async_ops_available()
    if not uses_umdk_async():
        raise RuntimeError(
            "Build the Ascend 950 registered-buffer plugin for this test"
        )
    dist.init_process_group("hccl", timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != profile.sources + profile.receivers:
        raise ValueError(
            f"{args.profile} requires {profile.sources}A{profile.receivers}F"
        )
    is_source = rank < profile.sources
    role_rank = rank if is_source else rank - profile.sources
    experts = profile.experts // profile.receivers
    topology = AFDAsyncTopology(
        "attention" if is_source else "ffn",
        role_rank,
        rank,
        profile.sources,
        profile.receivers,
        experts,
    )
    startup = AsyncCAMGraphStartup(rank, profile.sources, world)
    startup.plan = FFNGraphPlan(
        signature="identity-by-expert", layer_ids=profile.layers
    )
    store = dist.distributed_c10d._get_default_store()
    startup.initialize(store, f"umdk_v2:{args.profile}:{rows}:bf16")
    session = UmdkSession(
        startup,
        topology,
        profile.hidden,
        profile.topk,
        profile.capacity,
        profile.tp,
        profile.backend_rows,
    )
    device = f"npu:{local}"
    group_name = get_hccl_group_name(dist.distributed_c10d._get_default_group(), rank)
    try:
        session.initialize(device, group_name)
        print(f"rank={rank} profile={args.profile} cfg={session.cfg}", flush=True)
        dist.barrier()
        for step in range(3):
            keys = [
                f"umdk-roundtrip/{startup.session_epoch}/{step}/{a}"
                for a in range(profile.sources)
            ]
            if is_source:
                source_round(session, profile, rank, step, rows, device)
                store.set(keys[rank], "done")
            else:
                serve_round(session, store, keys, role_rank, step, experts)
            dist.barrier()
        session.prepare_close(True)
    except Exception as exc:
        # Peers polling the host store can stop before their device timeouts.
        # A poisoned device may still own in-flight buffers; torchrun owns teardown.
        with suppress(Exception):
            startup.fail(exc)
        raise
    dist.destroy_process_group()
    if rank == 0:
        print(f"UMDK Ascend 950 {args.profile} numerical roundtrip passed", flush=True)


if __name__ == "__main__":
    main()
