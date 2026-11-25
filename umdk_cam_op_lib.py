# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""UMDK-compatible entry points shipped by the Ascend 950 AFD plugin wheel."""

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded, uses_umdk_async

ensure_afd_ascend_ops_loaded()
if not uses_umdk_async():
    raise ImportError(
        "The registered-buffer operators require an Ascend 950 plugin build"
    )

from torch import ops  # noqa: E402 -- loading the extension registers the schemas

afd_async_dispatch_send = ops.umdk_cam_op_lib.afd_async_dispatch_send
afd_async_dispatch_recv = ops.umdk_cam_op_lib.afd_async_dispatch_recv
afd_async_combine_send = ops.umdk_cam_op_lib.afd_async_combine_send
afd_async_combine_recv = ops.umdk_cam_op_lib.afd_async_combine_recv
