# Ascend 950 arch35 roundtrip evidence

Complete captures of the five device runs described in
[`docs/npu/ascend950_umdk_arch35_roundtrip_result.md`](../../npu/ascend950_umdk_arch35_roundtrip_result.md),
plus the compact extracts the earlier revision carried.

| Directory | Run | Profile | Ranks | First-round rows | 507035 count |
| --- | --- | --- | --- | --- | ---: |
| `dsv4-flash-64rows-full/` | 3 | `dsv4-flash` (H=4096, K=6, E=256) | 8 (4A4F) | 64 | 28 |
| `small-65rows-full/` | 1 | `small` (H=256, K=2, E=4) | 4 (2A2F) | 65 | 10 |
| `small-1row-full/` | 2 | `small` | 4 | 1 | 14 |
| `aiv1-control-full/` | 1' | `small` with `_AIV_CORES = 1` | 4 | 1 | 14 |
| `small-64rows-repeat-full/` | 4 | `small`, fresh container | 4 (2A2F) | 64 | 10 |
| `deepep-reference-full/` | 5 | DeepEP reference pipeline | 4 | 64 | 3 |
| `dsv4-flash-64rows/` `small-65rows/` `small-1row/` `aiv1-control/` `small-64rows-repeat/` | extracts | — | — | — | — |

Each `*-full/` directory is the complete run directory as the test wrote it:

```text
roundtrip.log                     launcher plus all four or eight rank logs
plog/debug/plog/plog-*.log        ACL process logs, one per rank process
plog/run/device-*/                per-device run logs
plog/security/                    ACL security logs
extra-info/extra-info/data-dump/  exception_info plus the kernel objects
atrace/trace_*/                   Ascend trace: schedule and stackcore events
ranks/<uuid>/                     elastic launcher state
```

The smaller directories hold text extracts of the same runs, for reading
without downloading the binaries:

- `error-excerpt.txt` — per-rank session configuration, the dispatch line and
  the first `DispatchSend` failure block.
- `device-error.txt` — the AIV trap records from the ACL process logs, with
  the kernel entry and trap PCs.
- `dispatchsend-descriptor.json` — the DispatchSend kernel descriptor, which
  pins `core_type: AIV` and the kernel identifier
  `AfdAsyncDispatchSend_21c646510794c58ff569a30a2208ad7f`.
- `combine-recv-descriptor.json` — the CombineRecv descriptor from the same
  dump, for the paired operator.

`deepep-reference-full/` is the independent control: the pinned DeepEP test tree
(`25b5153e`) with `tests/npu/deepep_pipeline_cam.patch` applied, so DeepEP owns
the buffers and scheduler and the plugin only supplies the translated operator
calls. It traps with the same 507035 while reading back `owner.recv_status`,
and at the same relative PC offset as the plugin runs. See the report section
[Independent reference pipeline](../../npu/ascend950_umdk_arch35_roundtrip_result.md#independent-reference-pipeline-traps-at-the-same-relative-pc).

## Why the log files are force-added

The repository ignores `*.log` globally. This directory carries a nested
`.gitignore` that re-includes them, because these captures are the evidence
the report cites. `.gitattributes` marks the device dumps and kernel objects
as binary so Git does not attempt text diffs on them.

## What to look at first

`dsv4-flash-64rows-full/plog/debug/plog/` carries the trap on AIV core 0:

```text
aivec error, core id = 0, error code = 95
pc start: 0x120040819000, current: 0x12004081ab20
errcode:(95) The MTE instruction accesses an invalid GM address or the
             cross-device memory access timeout. subErrType: 0x4
```

`0x12004081ab20 - 0x120040819000 = 0x1b20`, the relative PC the
[bringup note](../../npu/ascend950_umdk_bringup.md#recorded-failure-on-november-26)
already records. Later cores in the same record report `errcode:(0)` with
`timeout or trap error`, which is the secondary propagation after core 0 traps.

## Cross-run invariant

Every run in this directory, whatever its profile, rank count or launch width,
reports the primary fault at `current: 0x12004081ab20` with `error code = 95`.
That address is the only one carrying the MTE error; the other addresses in the
same records carry `error code = 0` and `timeout or trap error`, which is the
secondary propagation after the first core traps.

The distinction matters when reading the logs:

```bash
# Primary fault: MTE error on the trap address
grep -rho 'error code = 95, dump info: pc start: 0x[0-9a-f]*, current: 0x[0-9a-f]*' \
  *-full/plog/debug/plog/ | grep -o 'current: 0x[0-9a-f]*' | sort | uniq -c
```

```bash
# Secondary: timeout propagation, error code 0
grep -rho 'current: 0x[0-9a-f]*' *-full/plog/debug/plog/ | sort | uniq -c
```

All four distinct addresses sit inside the same kernel image, so the secondary
records still identify `AfdAsyncDispatchSend_21c646510794c58ff569a30a2208ad7f`.
Only the `0x1b20` offset is the defect; the rest are fallout.
