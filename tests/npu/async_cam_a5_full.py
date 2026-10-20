# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run the A5 session through the existing independent four-phase golden.

Use a fresh torchrun with the matching 950 extension. This precision launcher
checks sequential complete rounds and rotates through the configured slots;
it is not a multi-round overlap or performance benchmark. --dry-run imports
only the standard library. Device execution is required to qualify CCU/MTE
visibility, resource registration, completion markers and slot reuse.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

    from afd_plugin.compat.npu.cam_async_a5 import AsyncCamA5Context

if TYPE_CHECKING or __package__:
    from .async_cam_common import (
        HarnessConfig,
        NativeHarness,
        add_common_arguments,
        config_from_args,
        describe_config,
        window_requirements,
    )
else:
    from async_cam_common import (
        HarnessConfig,
        NativeHarness,
        add_common_arguments,
        config_from_args,
        describe_config,
        window_requirements,
    )

MIN_SLOTS = 2
MAX_SLOTS = 8
FRAME_ALIGNMENT = 128
CONTROL_RECORD_BYTES = 128
CONTROL_RECORDS_PER_SLOT = 2
RING_HEADER_BYTES = 128
MIB_BYTES = 1024 * 1024
DESCRIPTOR_METADATA_BYTES = 64 * 1024
DESCRIPTOR_ARENA_BYTES = DESCRIPTOR_METADATA_BYTES + 8 * 1024 + 32
DS_INPUT = 0
DS_IDS = 1
DS_LAYER = 13
CS_INPUT = 0
CS_METADATA = 2
CR_IDS = 1
CR_WEIGHTS = 2


def full_profile_plan(config: HarnessConfig, slots: int) -> dict:
    if config.soc_version != "950":
        raise ValueError("The full profile requires --soc-version 950")
    if not MIN_SLOTS <= slots <= MAX_SLOTS:
        raise ValueError("slots must be between 2 and 8")
    if (
        config.ub_budget_bytes is None
        or config.ub_budget_bytes < DESCRIPTOR_ARENA_BYTES
    ):
        raise ValueError(
            "A5 descriptor preparation needs at least "
            f"{DESCRIPTOR_ARENA_BYTES} UB bytes"
        )
    metadata_entries = 5 + config.tp_size + config.experts_per_rank * config.tp_size
    if metadata_entries * 8 > DESCRIPTOR_METADATA_BYTES:
        raise ValueError("A5 descriptor metadata exceeds 64 KiB")
    fixed = RING_HEADER_BYTES + slots * CONTROL_RECORDS_PER_SLOT * CONTROL_RECORD_BYTES
    metadata = config.world_size * CONTROL_RECORDS_PER_SLOT * CONTROL_RECORD_BYTES
    frame = (
        ((config.window_mb * MIB_BYTES - fixed) // slots - metadata)
        // FRAME_ALIGNMENT
        * FRAME_ALIGNMENT
    )
    required = window_requirements(config)["minimum_bytes"]
    if frame < required:
        raise ValueError(f"Each A5 slot needs {required} bytes but has {frame}")
    report = describe_config(config)
    report["mte_pipeline_policy"]["candidate_operators"] = ["dispatch_recv"]
    report["combine_send_policy"] = {
        "transport": "custom_ccu",
        "descriptor_candidates": ["scalar", "simt"],
        "publication": "planning success and separate stream retirement marker",
    }
    report.update(
        full_features=True,
        wire_protocol="AFDA5SLT/v1",
        transport="custom_ccu",
        slot_count=slots,
        frame_capacity_bytes=frame,
        precision_in_flight_rounds=1,
    )
    return report


class FullNativeHarness(NativeHarness):
    """Reuse the baseline golden and metadata checks with the real A5 owner."""

    def __init__(self, config: HarnessConfig, slots: int):
        super().__init__(config)
        full_profile_plan(config, slots)
        self.slots = slots
        self.context: AsyncCamA5Context | None = None
        self.attention_tickets: deque[Tensor] = deque()
        self.ffn_tickets: dict[int, Tensor] = {}

    def __enter__(self) -> FullNativeHarness:
        super().__enter__()
        from afd_plugin.compat.npu.cam_async_a5 import (
            create_async_cam_a5_context,
            initialize_async_cam_a5_context,
        )

        config = self.config
        self.context = create_async_cam_a5_context(
            self.anchor,
            self.group_name,
            rank=self.rank,
            attention_ranks=config.attn_ranks,
            moe_ranks=config.moe_ranks,
            experts_per_rank=config.experts_per_rank,
            tp_size=config.tp_size,
            max_sequence=config.max_seq_len,
            hidden_size=config.hidden_size,
            topk=config.top_k,
            slot_count=self.slots,
            dynamic_quant=config.dynamic_quant,
        )
        assert self.payload_group is not None
        initialize_async_cam_a5_context(
            self.context,
            self.payload_group,
            rank=self.rank,
            world_size=config.world_size,
        )
        self.operations = {
            "dispatch_send": self._full_dispatch_send,
            "dispatch_recv": self._full_dispatch_recv,
            "combine_send": self._full_combine_send,
            "combine_recv": self._full_combine_recv,
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        clean = exc_type is None and not self._failed
        if clean and self.context is not None:
            self.context.quiesce()
        super().__exit__(exc_type, exc_value, traceback)
        if clean and self.context is not None:
            self.context.release_after_group_destroyed()
        # On failure torchrun terminates the domain; do not claim a timeout
        # proves quiescence or attempt an in-process reset of its buffers.

    def _full_dispatch_send(self, *arguments: Tensor | int | str) -> Tensor:
        context = cast("AsyncCamA5Context", self.context)
        ticket = context.dispatch_send(
            cast("Tensor", arguments[DS_INPUT]),
            cast("Tensor", arguments[DS_IDS]),
            cast(int, arguments[DS_LAYER]),
        )
        self.attention_tickets.append(ticket)
        return ticket

    def _full_dispatch_recv(self, *arguments: Tensor | int | str) -> list[Tensor]:
        context = cast("AsyncCamA5Context", self.context)
        outputs = context.dispatch_recv()
        self.ffn_tickets[id(outputs[2])] = outputs[4]
        return outputs[:4]

    def _full_combine_send(self, *arguments: Tensor | int | str) -> None:
        context = cast("AsyncCamA5Context", self.context)
        metadata = cast("Tensor", arguments[CS_METADATA])
        ticket = self.ffn_tickets[id(metadata)]
        context.combine_send(cast("Tensor", arguments[CS_INPUT]), metadata, ticket)
        del self.ffn_tickets[id(metadata)]

    def _full_combine_recv(self, *arguments: Tensor | int | str) -> Tensor:
        context = cast("AsyncCamA5Context", self.context)
        return context.combine_recv(
            cast("Tensor", arguments[CR_IDS]),
            cast("Tensor", arguments[CR_WEIGHTS]),
            self.attention_tickets.popleft(),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(soc_version="950")
    parser.add_argument("--slots", type=int, default=MIN_SLOTS)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        report = full_profile_plan(config, args.slots)
        iterations = args.iterations if args.iterations is not None else args.slots + 1
        if iterations <= 0:
            raise ValueError("iterations must be positive")
    except ValueError as error:
        parser.error(str(error))
    report["iterations"] = iterations
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    with FullNativeHarness(config, args.slots) as harness:
        report["environment"] = harness.gather_reports(harness.environment())
        rounds = [
            harness.gather_reports(harness.run_round(layer, validate=True))
            for layer in range(iterations)
        ]
        passed = not any(row["failures"] for round_rows in rounds for row in round_rows)
        report.update(status="passed" if passed else "failed", rounds=rounds)
    report["cleanup_complete"] = True
    if harness.rank == 0:
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
