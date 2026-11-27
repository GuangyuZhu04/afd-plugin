# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Eager Ascend 950 Attention/FFN communication using registered UMDK storage."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from struct import Struct
from time import monotonic
from typing import TYPE_CHECKING

import torch

from afd_plugin.connectors.base import FFNExecutionSession
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu.async_cam_runtime.umdk_layout import (
    COMPLETE,
    IDLE,
    SELECTED,
    SLICE,
    TRANSACTION_MAGIC,
    LayerRows,
    TransactionLayout,
)

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.async_cam import AFDAsyncTopology
    from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
        AsyncCAMGraphStartup,
    )

_TASK = Struct("<16I")
_ABI_VERSION = 2
_LOCAL_ROUNDS = 1
_MAX_LANE_ROWS = 32768
_EXPERT_ALIGNMENT = 32
_AIV_CORES = 32
_TIMEOUT_SECONDS = 120
_TRANSPORT_RESERVE = 256 << 20
_TRANSFER_DONE = [2, 0]


@dataclass(slots=True)
class UmdkSlice(AFDTransferState):
    transaction: int
    index: int
    groups: tuple[LayerRows, ...]


@dataclass(slots=True)
class SourceRound:
    context: AFDTransferContext
    x: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor
    task: torch.Tensor
    output: torch.Tensor
    inputs_ready: torch.npu.Event
    sent: torch.npu.Event
    rows: int
    rounds: int


class UmdkSession(FFNExecutionSession):
    signature_prefix = "umdk_v2:"
    comm_id = 0
    comm_args = None
    allow_row_slicing = False

    def __init__(
        self,
        startup: AsyncCAMGraphStartup,
        topology: AFDAsyncTopology,
        hidden: int,
        topk: int,
        capacity: int,
        tp: int,
        backend_rows: int,
    ):
        self.startup, self.topology = startup, topology
        self.hidden, self.topk, self.capacity, self.tp = hidden, topk, capacity, tp
        if not 0 < capacity <= _MAX_LANE_ROWS:
            raise ValueError("UMDK CombineRecv admits 1..32768 rows per Attention rank")
        self.window = min(
            capacity,
            _MAX_LANE_ROWS,
            (backend_rows // _EXPERT_ALIGNMENT * _EXPERT_ALIGNMENT)
            // topology.attn_size,
        )
        if self.window < 1:
            raise ValueError("UMDK receive capacity cannot fit one row per source")
        self.backend_rows = backend_rows
        self.owner = None
        self.pending: SourceRound | None = None
        self.current_slice: AFDTransferContext | None = None
        self.continuation = False
        self.sequence = 0

    @property
    def has_pending_attention(self) -> bool:
        return self.pending is not None

    def attention_step(self):
        return nullcontext()

    def initialize(self, device: str, group_name: str) -> None:
        import deep_ep

        t = self.topology
        if self.startup.mode != "eager":
            raise ValueError("UMDK transaction selection requires eager execution")
        self.device, self.group = torch.device(device), group_name
        self.ops = torch.ops.umdk_cam_op_lib
        sources = list(range(t.attn_size))
        receivers = list(range(t.attn_size, t.world_size))
        lanes = [[a, 0, a // self.tp, a % self.tp, 0, self.window] for a in sources]
        self.layout = TransactionLayout(
            len(lanes), t.expert_per_rank, self.startup.layer_ids, _EXPERT_ALIGNMENT
        )
        self.owner = deep_ep.AFDBuffer(
            t.world_rank,
            group_name,
            sources,
            receivers,
            lanes,
            list(self.layout.layers),
            self.hidden,
            0,
            self.topk,
            t.expert_per_rank * t.ffn_size,
            self.capacity,
            False,
            _AIV_CORES,
            _TIMEOUT_SECONDS,
            _TIMEOUT_SECONDS,
            torch.npu.mem_get_info(self.device)[0],
            _TRANSPORT_RESERVE,
        )
        o = self.owner
        o.enable_local_rounds()
        if t.world_rank < t.attn_size:
            o.prepare_combine_recv(round_step=False, fair_round_robin=False)
            self.send_stream = torch.npu.Stream(device=self.device)
            self.sf = torch.empty(1, dtype=torch.int16, device=device)
            self.progress = torch.empty(0, dtype=torch.uint8, device=device)
        else:
            q = self.window * t.attn_size
            qa = (q + _EXPERT_ALIGNMENT - 1) // _EXPERT_ALIGNMENT * _EXPERT_ALIGNMENT
            o.prepare_recv(
                q,
                qa,
                _EXPERT_ALIGNMENT,
                ffn_transaction=True,
                backend_max_rows=self.backend_rows,
            )
            o.prepare_ffn(False)
            self.sf = torch.empty((qa, 0), dtype=torch.int16, device=device)
            o.compileFFNPollOnly(False)
            o.compileFFNSchedulerOnly()
            o.compileComputeReadyOnly()
        self.cfg = o.camConfig()

    def dispatch(
        self,
        context: AFDTransferContext,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        valid_rows: int,
    ) -> None:
        if self.pending is not None:
            raise RuntimeError("UMDK source slot still owns a pending return")
        rows = x.size(0)
        if not 0 <= valid_rows <= rows <= self.capacity or x.shape != (
            rows,
            self.hidden,
        ):
            raise ValueError("UMDK source rows exceed the admitted geometry")
        if (
            x.dtype != torch.bfloat16
            or context.metadata.layer_idx not in self.layout.layers
        ):
            raise ValueError("UMDK needs BF16 activations and a negotiated FFN layer")
        o, c = self.owner, self.cfg
        rounds = max(1, (valid_rows + self.window - 1) // self.window)
        header = _TASK.pack(
            _ABI_VERSION,
            self.topology.world_rank,
            0,
            self.sequence,
            context.metadata.layer_idx,
            valid_rows,
            rounds,
            _LOCAL_ROUNDS,
            *([0] * 8),
        )
        task = torch.tensor(list(header), dtype=torch.uint8, device=self.device)
        idx = torch.full(
            (self.capacity, self.topk), -1, dtype=torch.int64, device=self.device
        )
        weight = torch.zeros_like(idx, dtype=torch.float32)
        idx[:valid_rows].copy_(indices[:valid_rows])
        weight[:valid_rows].copy_(weights[:valid_rows])
        mask = torch.arange(self.capacity, device=self.device) < valid_rows
        output = torch.zeros(
            (self.capacity, self.hidden), dtype=torch.bfloat16, device=self.device
        )
        hidden = x.contiguous() if rows else output[:1]
        ready, sent = torch.npu.Event(), torch.npu.Event()
        ready.record(torch.npu.current_stream(self.device))
        self.pending = SourceRound(
            context, hidden, idx, weight, mask, task, output, ready, sent, rows, rounds
        )
        self.sequence = (self.sequence + 1) % (1 << 32)
        with torch.npu.stream(self.send_stream):
            self.send_stream.wait_event(ready)
            n = hidden.size(0)
            self.ops.afd_async_dispatch_send(
                hidden,
                self.sf,
                idx[:n],
                weight[:n],
                mask[:n],
                task,
                o.context,
                o.control_buffer,
                o.round_ledger,
                o.workspace,
                o.jetty_ptrs,
                o.send_status.view(torch.int32),
                self.group,
                c["num_ranks"],
                c["num_experts"],
                self.topk,
                c["num_tokens_per_tile"],
                0,
                0,
                0,
                c["timeout_cycles"],
                8,
                c["num_vec_cores"],
            )
            sent.record(self.send_stream)

    def combine(self, stage: int, context: AFDTransferContext | None) -> torch.Tensor:
        p = self.pending
        if (
            p is None
            or p.context.metadata.stage_idx != stage
            or (context is not None and p.context is not context)
        ):
            raise RuntimeError("UMDK return does not match the outstanding source slot")
        o, c = self.owner, self.cfg
        # A single round publishes its ledger and returns independently. If it
        # rejects the task, there is no ledger for a blocking receive to consume.
        # A multi-round send can need credit from this receive to reuse its slot.
        if p.rounds == 1:
            self._check_dispatch_status(p)
        else:
            torch.npu.current_stream(self.device).wait_event(p.inputs_ready)
        self.ops.afd_async_combine_recv(
            o.payload,
            p.task,
            o.context,
            o.control_buffer,
            o.round_ledger,
            o.combine_recv_workspace,
            p.mask,
            p.output,
            p.indices,
            p.weights,
            o.recv_status.view(torch.int32),
            self.progress,
            self.group,
            c["num_experts"],
            c["num_ranks"],
            self.topk,
            0,
            0,
            self.hidden,
            self.capacity,
            8,
            0,
            c["timeout_cycles"],
            c["num_vec_cores"],
        )
        recv_status = o.recv_status.view(torch.int32)[:2].tolist()
        if p.rounds > 1:
            self._check_dispatch_status(p)
        if recv_status != _TRANSFER_DONE:
            raise RuntimeError(
                f"UMDK CombineRecv failed on rank {self.topology.world_rank}, "
                f"layer {p.context.metadata.layer_idx}: {recv_status}"
            )
        self.pending = None
        return p.output[: p.rows]

    def _check_dispatch_status(self, pending: SourceRound) -> None:
        rank = self.topology.world_rank
        layer = pending.context.metadata.layer_idx
        try:
            pending.sent.synchronize()
            status = self.owner.send_status.view(torch.int32)[:3].tolist()
        except RuntimeError as exc:
            raise RuntimeError(
                f"UMDK DispatchSend synchronization failed on rank {rank}, "
                f"layer {layer}, rounds={pending.rounds}: {exc}"
            ) from exc
        if status[:2] != _TRANSFER_DONE:
            raise RuntimeError(
                f"UMDK DispatchSend failed on rank {rank}, layer {layer}: "
                f"state={status[0]}, error={status[1]}, round={status[2]}"
            )

    def _transaction(self) -> list[int]:
        return (
            self.owner.active.narrow(0, self.layout.transaction_offset, 32)
            .view(torch.int32)
            .tolist()
        )

    def receive(self, *, wait: bool) -> AFDA2FTransferPayload | None:
        o, c = self.owner, self.cfg
        if self.current_slice is not None:
            raise RuntimeError("Previous UMDK slice has not been returned")
        if not self.continuation:
            deadline = monotonic() + _TIMEOUT_SECONDS
            while True:
                self.startup.check_failure()
                o.ffnPoll()
                o.ffnScheduler(
                    o.context,
                    o.control_buffer,
                    o.scheduler_workspace,
                    o.active,
                    o.execution_state,
                    o.completion,
                    o.psum_num_recv_tokens_per_expert,
                    o.num_unaligned_recv_tokens_per_expert,
                )
                state = self._transaction()[0]
                if state == SELECTED:
                    break
                if state != IDLE:
                    raise RuntimeError(
                        f"UMDK scheduler returned transaction state {state}"
                    )
                if not wait:
                    return None
                if monotonic() >= deadline:
                    raise TimeoutError("UMDK timed out waiting for an FFN transaction")
        self.ops.afd_async_dispatch_recv(
            o.payload,
            o.context,
            o.active,
            o.execution_state,
            o.recv_workspace,
            o.control_buffer,
            o.recv_x,
            self.sf,
            o.recv_src_metadata,
            self.group,
            c["num_ranks"],
            c["num_experts"],
            self.topk,
            c["num_source_lanes"],
            c["num_layers"],
            c["recv_q"],
            c["recv_q_alloc"],
            c["recv_alignment"],
            c["recv_tile"],
            c["recv_rank"],
            0,
            0,
            c["payload_bytes"],
            c["control_bytes"],
            c["context_bytes"],
            self.hidden,
            0,
            c["num_vec_cores"],
        )
        tx = self._transaction()
        stage, error = o.execution_state[:8].view(torch.int32).tolist()
        header = o.active[:40].view(torch.int32).tolist()
        if tx[0] != SLICE or (stage, error) != (2, 0) or header[9] != TRANSACTION_MAGIC:
            raise RuntimeError(
                f"UMDK DispatchRecv failed: tx={tx}, execution={(stage, error)}"
            )
        first, end = tx[2:4]
        if not 0 <= first <= end <= len(self.layout.layers) * self.layout.experts:
            raise ValueError("UMDK returned an invalid group interval")
        counts = (
            o.active.narrow(0, self.layout.group_offset + first * 4, (end - first) * 4)
            .view(torch.int32)
            .tolist()
        )
        groups = self.layout.layer_rows(first, end, counts, header[7], header[8])
        if not 0 <= header[8] <= c["recv_q_alloc"]:
            raise ValueError("UMDK physical slice exceeds registered FFN storage")
        metadata = AFDTransferMetadata(
            groups[0].layer_id if len(groups) == 1 else -1, tx[7], [header[8]]
        )
        self.current_slice = AFDTransferContext(
            metadata, UmdkSlice(tx[1], tx[7], groups)
        )
        return AFDA2FTransferPayload(o.recv_x[: header[8]], self.current_slice)

    def result_buffer(self, context: AFDTransferContext) -> torch.Tensor:
        if context is not self.current_slice or not isinstance(
            context.states, UmdkSlice
        ):
            raise RuntimeError("UMDK CombineSend requires the outstanding FFN slice")
        return self.owner.combine_x[: context.metadata.total_tokens]

    def send_ffn(self, result: torch.Tensor, context: AFDTransferContext) -> None:
        target = self.result_buffer(context)
        o, c = self.owner, self.cfg
        if (
            result.shape != target.shape
            or result.dtype != target.dtype
            or result.device != target.device
        ):
            raise ValueError("UMDK FFN output must match the registered BF16 slice")
        if not result.is_set_to(target):
            target.copy_(result)
        o.computeReady()
        self.ops.afd_async_combine_send(
            o.combine_x,
            o.recv_src_metadata,
            o.context,
            o.active,
            o.control_buffer,
            o.execution_state,
            o.combine_send_workspace,
            o.jetty_ptrs,
            o.completion,
            self.group,
            c["num_src_ranks"],
            c["num_ranks"],
            c["num_experts"],
            self.topk,
            c["num_source_lanes"],
            c["num_layers"],
            c["recv_q_alloc"],
            c["combine_tile"],
            c["recv_rank"],
            c["timeout_cycles"],
            c["completion_bytes"],
            c["num_vec_cores"],
        )
        tx = self._transaction()
        if tx[0] not in (SELECTED, COMPLETE) or (tx[1], tx[7]) != (
            context.states.transaction,
            context.states.index + 1,
        ):
            raise RuntimeError(
                f"UMDK CombineSend did not commit the expected slice: {tx}"
            )
        self.continuation = tx[0] == SELECTED
        self.current_slice = None

    def prepare_close(self, initialized: bool) -> None:
        if (
            self.pending is not None
            or self.current_slice is not None
            or self.continuation
        ):
            raise RuntimeError("Outstanding UMDK transaction during shutdown")
        if self.owner is not None:
            self.owner.destroy()
            self.owner = None

    def release(self) -> None:
        self.owner = None
