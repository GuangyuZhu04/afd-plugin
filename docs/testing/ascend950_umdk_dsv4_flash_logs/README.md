# Ascend 950 arch35 roundtrip evidence

Compact extracts from the device runs described in
[`docs/npu/ascend950_umdk_arch35_roundtrip_result.md`](../../npu/ascend950_umdk_arch35_roundtrip_result.md).

| Directory | Run | Profile | Ranks | First-round rows | 507035 count |
| --- | --- | --- | --- | --- | ---: |
| `dsv4-flash-64rows` | 3 | `dsv4-flash` (H=4096, K=6, E=256) | 8 (4A4F) | 64 | 28 |
| `small-65rows` | 1 | `small` (H=256, K=2, E=4) | 4 (2A2F) | 65 | 10 |
| `small-1row` | 2 | `small` | 4 | 1 | 14 |
| `aiv1-control` | 1' | `small` with `_AIV_CORES = 1` | 4 | 1 | 14 |

Each directory holds:

- `error-excerpt.txt` — per-rank session configuration, the dispatch line and
  the first `DispatchSend` failure block.
- `device-error.txt` — the AIV trap records from the ACL process logs, with
  the kernel entry and trap PCs.
- `dispatchsend-descriptor.json` — the DispatchSend kernel descriptor, which
  pins `core_type: AIV` and the kernel identifier
  `AfdAsyncDispatchSend_21c646510794c58ff569a30a2208ad7f`.
- `combine-recv-descriptor.json` — the CombineRecv descriptor from the same
  dump, for the paired operator.

These are text extracts. The full per-rank ACL plogs and binary device dumps
stay with the run host; the extracts keep the trap PC, the kernel identifier
and the failing operator call, which are the fields a fix has to be compared
against.

## What to look at first

`dsv4-flash-64rows/device-error.txt` carries the trap on AIV core 0:

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
