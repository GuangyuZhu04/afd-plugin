# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Eager view of the UMDK v2 FFN transaction trailer (all offsets in bytes)."""

from dataclasses import dataclass

ACTIVE_BYTES = 64
UMDK_MAX_SOURCE_ROWS = 32768
RECORD_BYTES = 32
TRANSACTION_MAGIC = 0x41465431
IDLE, SELECTED, SLICE, COMPLETE, ABORTED = range(5)


@dataclass(frozen=True, slots=True)
class LayerRows:
    layer_id: int
    begin: int
    counts: tuple[int, ...]  # physical rows per local expert, including padding

    @property
    def end(self) -> int:
        return self.begin + sum(self.counts)


@dataclass(frozen=True, slots=True)
class TransactionLayout:
    lanes: int
    experts: int
    layers: tuple[int, ...]
    alignment: int

    @property
    def transaction_offset(self) -> int:
        return (
            ACTIVE_BYTES
            + 2 * RECORD_BYTES * self.lanes
            + RECORD_BYTES * self.lanes * self.experts
        )

    @property
    def group_offset(self) -> int:
        count_bytes = 4 * self.lanes * self.experts
        return (
            self.transaction_offset
            + RECORD_BYTES
            + 2 * RECORD_BYTES * self.lanes
            + (count_bytes + 31) // 32 * 32
        )

    def layer_rows(
        self, first: int, end: int, counts: list[int], valid: int, physical: int
    ) -> tuple[LayerRows, ...]:
        if (
            not 0 <= first <= end <= self.experts * len(self.layers)
            or len(counts) != end - first
        ):
            raise ValueError("Invalid UMDK slice interval")
        if any(n < 0 for n in counts) or sum(counts) != valid:
            raise ValueError("UMDK group counts disagree with valid rows")
        groups = []
        offset = 0
        for slot in range(
            first // self.experts, (end + self.experts - 1) // self.experts
        ):
            sizes = [0] * self.experts
            for group in range(
                max(first, slot * self.experts), min(end, (slot + 1) * self.experts)
            ):
                n = counts[group - first]
                sizes[group % self.experts] = (
                    (n + self.alignment - 1) // self.alignment * self.alignment
                )
            if any(sizes):
                layer = LayerRows(self.layers[slot], offset, tuple(sizes))
                groups.append(layer)
                offset = layer.end
        if offset != physical:
            raise ValueError("UMDK group counts disagree with physical rows")
        return tuple(groups)
