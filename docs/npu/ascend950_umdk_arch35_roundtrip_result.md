# Ascend 950 arch35 registered-buffer roundtrip on hardware

This record reports one device run of the four 950 registered-buffer operators
through `tests/npu/umdk_roundtrip.py` on framework commit `893f765`. Every
rank registers its buffers and the session completes, and the first
`DispatchSend` of the first round then traps the AIV core with device error
507035. The recorded failure in
[the bringup note](ascend950_umdk_bringup.md#recorded-failure-on-november-26)
is the same fault, at the same relative PC. This run keeps the device dump,
the ACL process logs and the kernel descriptors that the November 26 attempt
lost, so the trap can be attributed below operator granularity.

## Environment

| Item | Value |
| --- | --- |
| Host | 8 x Ascend 950DT (9582), driver 25.1.rc2, CANN 9.2.1 |
| Python | 3.12.16, torch 2.10.0+cpu, torch_npu 2.10.0.post7 |
| Plugin | `codex/ascend950-umdk-dsv4-flash` @ `893f765`, operator baseline `fee97356` |
| vLLM | `0.32.0+empty.g5df63a7` @ `5df63a72` (image checkout) |
| vLLM-Ascend | `0.32.0.dev0+gbd42eea` @ `bd42eea` (built from source) |
| DeepEP-Ascend | `1.0.0+25b5153` @ `25b5153e`, DeepJIT `65f5139` |
| Container | host network, all `/dev/davinci*`, the `/dev/uburma/*` endpoints and the four URMA runtime/provider libraries |

The four operators register as `afd_async_dispatch_send_arch35`,
`afd_async_dispatch_recv_arch35`, `afd_async_combine_send_arch35` and
`afd_async_combine_recv_arch35` with 23/28/22/24 arguments and 1/3/1/5
outputs, matching the schemas `compat/npu/ops.py` asserts.
`check_umdk_mxfp_stack()` returns the pinned pair and `AFDBuffer` is present.

## Method

Three runs cover the operator in isolation, without vLLM or a checkpoint:

| Run | Profile | Ranks | First-round rows |
| --- | --- | --- | --- |
| 1 | `small` (H=256, K=2, E=4) | 4 (2A2F) | 65 |
| 2 | `small` | 4 | 1 |
| 3 | `dsv4-flash` (H=4096, K=6, E=256) | 8 (4A4F) | 64 |
| 4 | `small`, repeated on a fresh container | 4 (2A2F) | 64 |

Run 3 is the model geometry the Flash case uses. Runs 1 and 2 isolate scale:
run 2 keeps the same topology as run 1 with a single source row.

Each rank prints its `session.cfg` before the first round, which confirms the
registered-buffer geometry and the F-local transaction layout were accepted.

## Result

All three runs fail at the same point, before any round completes:

```text
rank=0 step=0 layer=2 valid_rows=65 dispatch
RuntimeError: UMDK DispatchSend synchronization failed on rank 1, layer 2,
              rounds=1: synchronize: ... NPU function error:
              device error type 3, error code is 507035
```

The device reported the trap on AIV core 0:

```text
aivec error, core id is 0, error code = 95
pc start: 0x120040819000, current: 0x12004081ab20
errcode:(95) The MTE instruction accesses an invalid GM address or the
             cross-device memory access timeout. subErrType: 0x4
```

The offset from the kernel entry, `0x1b20`, is the relative PC the bringup note
records for the archived four-rank `rt11` probe. Five independent executions
therefore trap at the same instruction: every run, including the Flash
geometry, the single-row run and the `_AIV_CORES = 1` control, reports
`current: 0x12004081ab20` and no other trap address. The fault is
deterministic, not a race or a scale-dependent overflow.

Failure counts per run: 10 / 14 / 28 occurrences of `507035`, all raised from
the host copy of the operator status through `CopyKernelOpApi.cpp:94`, that
is, from reading back the status the kernel publishes. The host-side tiling
checks emit no diagnostic, so parameter and tiling validation pass; the fault
is in device execution.

## Scale and configuration controls

| Hypothesis | Control | Outcome |
| --- | --- | --- |
| 32 AIV cores too wide for the launch geometry | rebuilt with `_AIV_CORES = 1` | same 507035 |
| first-round row count too large | `--rows 1` (run 2) | same 507035 |
| role layout or rank count | 4-rank 2A2F and 8-rank 4A4F (run 3) | same 507035 |
| missing URMA runtime or endpoints | library set and `/dev/uburma/*` compared against a working tenant container; `ldd` clean | not the cause |
| operator registration or ABI drift | preflight schemas match the asserted arity and outputs | not the cause |
| pinned version gate | `check_umdk_mxfp_stack()` passes | not the cause |

Session setup itself succeeds in every run: all ranks build `AFDBuffer`,
register the 4096-row source capacity and 43 layer IDs for the Flash geometry,
and reach the first dispatch. The fault is therefore in the operator data path,
not in buffer registration or link setup.

## Retained evidence

```text
docs/testing/ascend950_umdk_dsv4_flash_logs/
  dsv4-flash-64rows/roundtrip.log   eight-rank log with the error excerpt
  dsv4-flash-64rows/plog/           ACL process logs for every rank
  dsv4-flash-64rows/data-dump/      exception_info and the DispatchSend object
```

The DispatchSend kernel descriptor records `core_type: AIV` and
`blockDim: -1`, and the exception dump names
`AfdAsyncDispatchSend_21c646510794c58ff569a30a2208ad7f`, so a future fix can be
compared against the same kernel identifier and the same trap PC.

## Conclusion

The arch35 registered-buffer operators integrate with the plugin: they register
with the expected schemas, the pinned stack gate passes, the DeepEP
`AFDBuffer` owner is available, and the UMDK session completes on four and
eight dies. The transport cannot carry a single round on this host, because the
first `DispatchSend` traps the AIV MTE unit on an invalid GM address at
relative PC `0x1b20`.

Because the fault reproduces with no model, no checkpoint and a single source
row, it is confined to the operator kernel. It is the same fault the bringup
note records, now with the device dump and ACL logs attached.
