# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Physical FFN groups passed to the compute backend."""

import pytest

from afd_plugin.connectors.npu.async_cam_runtime.umdk_layout import (
    LayerRows,
    TransactionLayout,
)


def test_slice_preserves_expert_slots_layer_order_and_physical_offsets():
    layout = TransactionLayout(lanes=2, experts=3, layers=(2, 9), alignment=4)
    # The slice starts inside layer 2 and ends inside layer 9.
    assert layout.layer_rows(1, 5, [3, 0, 1, 5], 9, 16) == (
        LayerRows(2, 0, (0, 4, 0)),
        LayerRows(9, 4, (4, 8, 0)),
    )
    assert layout.layer_rows(1, 3, [0, 0], 0, 0) == ()


@pytest.mark.parametrize(
    "end,counts,valid,physical",
    [
        (7, [0] * 7, 0, 0),
        (2, [1], 1, 4),
        (2, [-1, 2], 1, 4),
        (2, [1, 3], 3, 8),
        (2, [1, 3], 4, 4),
    ],
)
def test_inconsistent_transaction_cannot_reach_expert_compute(
    end, counts, valid, physical
):
    layout = TransactionLayout(2, 3, (2, 9), 4)
    with pytest.raises(ValueError):
        layout.layer_rows(0, end, counts, valid, physical)
