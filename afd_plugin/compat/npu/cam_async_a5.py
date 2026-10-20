# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Typed access to the native A5 full-profile session.

The caller owns a dedicated HCCL process group and one submission stream.
Quiesce before destroying that group, then release the native registration
owners. A device ticket is preserved from DS to CR, or from DR to CS; its
routing/epoch fields are never read by Python on the submission path.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributed.distributed_c10d import ProcessGroup

SETUP_NAMESPACE = "afd_a5_full_v1"


class AsyncCamA5Setup(Protocol):
    def prepare(self) -> list[int]: ...

    def configure(self, resources: list[list[int]]) -> None: ...

    def initialize(self) -> None: ...


class AsyncCamA5Context(AsyncCamA5Setup, Protocol):
    def dispatch_send(self, x: Tensor, ids: Tensor, layer: int) -> Tensor: ...

    def dispatch_recv(self) -> list[Tensor]: ...

    def combine_send(self, x: Tensor, metadata: Tensor, ticket: Tensor) -> None: ...

    def combine_recv(self, ids: Tensor, weights: Tensor, ticket: Tensor) -> Tensor: ...

    def quiesce(self) -> None: ...

    def abort(self) -> None: ...

    def release_after_group_destroyed(self) -> None: ...


def initialize_async_cam_a5_context(
    context: AsyncCamA5Setup,
    group: ProcessGroup,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Use the existing rendezvous Store, never the payload windows, for setup.

    Every rank finishes its MC2 resource query before any frame is cleared.
    All frames are cleared before channel registration, and all registrations
    finish before data-plane calls can start. No Store traffic occurs per batch.
    """
    from torch.distributed.distributed_c10d import _get_process_group_store

    store = _get_process_group_store(group)
    resource_keys = [f"{SETUP_NAMESPACE}/resource/{peer}" for peer in range(world_size)]
    store.set(resource_keys[rank], json.dumps(context.prepare()))
    store.wait(resource_keys)
    resources = [json.loads(store.get(key)) for key in resource_keys]
    context.configure(resources)
    initialized_keys = [
        f"{SETUP_NAMESPACE}/initialized/{peer}" for peer in range(world_size)
    ]
    store.set(initialized_keys[rank], b"1")
    store.wait(initialized_keys)
    context.initialize()
    registered_keys = [
        f"{SETUP_NAMESPACE}/registered/{peer}" for peer in range(world_size)
    ]
    store.set(registered_keys[rank], b"1")
    store.wait(registered_keys)


def create_async_cam_a5_context(
    anchor: Tensor,
    group_name: str,
    *,
    rank: int,
    attention_ranks: int,
    moe_ranks: int,
    experts_per_rank: int,
    tp_size: int,
    max_sequence: int,
    hidden_size: int,
    topk: int,
    slot_count: int,
    dynamic_quant: int,
) -> AsyncCamA5Context:
    """Create an uninitialized owner so failed setup can retain its resources."""
    # The native backend is loaded only when an NPU session is requested.
    import torch

    config = [
        rank,
        attention_ranks,
        moe_ranks,
        experts_per_rank,
        tp_size,
        max_sequence,
        hidden_size,
        topk,
        slot_count,
        dynamic_quant,
    ]
    return cast(
        AsyncCamA5Context,
        torch.classes.afd_ascend.AsyncCamA5Context(anchor, group_name, config),
    )
