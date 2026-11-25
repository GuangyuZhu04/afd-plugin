# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in shape diagnostics for compact and registered-buffer CAM calls."""

import os

from torch import Tensor
from vllm.logger import init_logger

# Worker evidence uses vLLM's configured handler on every local rank.
cam_logger = init_logger("vllm.afd.cam")

_CAM_LOG_SKIPPED_ARGS = frozenset({"comm_args", "comm_id", "group_name"})
_CAM_OP_IO_LOG_ENV = "AFD_CAM_OP_IO_LOG"


def log_cam_op_values(op_name: str, label: str, **kwargs: object) -> None:
    if os.environ.get(_CAM_OP_IO_LOG_ENV, "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    lines: list[str] = []
    for name, value in kwargs.items():
        if name in _CAM_LOG_SKIPPED_ARGS:
            continue
        if isinstance(value, Tensor):
            description = (
                f"Tensor(dtype={value.dtype}, shape={tuple(value.shape)}, "
                f"device={value.device})"
            )

        else:
            description = repr(value)
        lines.append(f"  {name}={description}")
    cam_logger.warning("AFD CAM %s %s:\n%s", op_name, label, "\n".join(lines))
