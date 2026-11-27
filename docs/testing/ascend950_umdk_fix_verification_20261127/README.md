# Ascend 950 UMDK: verification of the arch35 sender alignment fix

Device re-run of the four registered-buffer operators after
`f4b1f20` (align the arch35 sender stack objects before binding GM arguments)
and `243aec6` (address the CombineRecv UB pool by its physical offset) landed
on `codex/ascend950-umdk-afd` and `codex/ascend950-umdk-dsv4-flash`.

- Framework HEAD under test: `7cc7911`
  (`codex/ascend950-umdk-dsv4-flash`, which strictly contains
  `codex/ascend950-umdk-afd` at `243aec6`).
- Baseline that failed: the roundtrip record in
  [the arch35 result](../../npu/ascend950_umdk_arch35_roundtrip_result.md) and the
  bisection in
  [the investigation note](../../npu/ascend950_umdk_507035_root_cause_investigation.md).

## Result

**The 507035 device error is gone.** Every run in this directory reports zero
`507035`, zero `invalid GM`, and zero `error code = 95` / `aivec error`
entries in its ACL process logs. The previous baseline run
`roundtrip-small-1127082506` reported `507035` ten times.

| Run | Geometry | Outcome |
| --- | --- | --- |
| `4card-small-pass-1` | 4 dies, 2A2F, H=256, K=2, E=4 | **passed** |
| `4card-small-pass-2` | same, second container-fresh run | **passed** |
| `4card-reference-pass` | DeepEP reference pipeline, H=7168, K=2, E=8 | **RUN_PASS** |
| `8card-flash-fail-1` | 8 dies, 4A4F, H=4096, K=6, E=256, rows=64 | failed, protocol timeout |
| `8card-flash-fail-2` | same, repeated | failed, identical |
| `8card-flash-rows1-fail` | same, rows=1 | failed, identical |

The kernel really changed: the installed DispatchSend object moved from
`2290fc09…` — the ELF the earlier brings-up failure recorded — to
`d8d7ad20…`.

### What passes

```text
[default0]:UMDK Ascend 950 small numerical roundtrip passed
```

```text
{"status": "RUN_PASS", "scope": "CAM four operators, identity experts, no graph",
 "lengths": [64, 64], "window": 64, "rounds": 1, "q": 128,
 "q_mode": "whole_expert_bound", "precheck": false, "microbatches": 1,
 "round_step": false, "sweeps_on_rank0": 0}
```

The reference pipeline is the sharper of the two: before the fix it trapped at
the same relative PC `0x1b20` as the service path. It now completes, so the
fault the investigation note tracked is fixed in the operator, not merely
avoided by a different geometry.

### What still fails

The Flash geometry fails on `CombineRecv`, and the failure is a status the
kernel returns rather than a device fault. No plog in these runs contains a
trap:

```text
RuntimeError: UMDK CombineRecv failed on rank 0, layer 1: [3, 7]
RuntimeError: UMDK CombineRecv failed on rank 1, layer 2: [3, 7]
RuntimeError: UMDK CombineRecv failed on rank 2, layer 3: [3, 7]
RuntimeError: UMDK CombineRecv failed on rank 3, layer 4: [3, 7]
```

`[3, 7]` is the `(TransferStatus, ErrorCode)` word pair, not `(stage, error)`:

- `umdk_session.py:50` — `_TRANSFER_DONE = [2, 0]`
- `umdk_session.py:330` — `o.recv_status.view(torch.int32)[:2].tolist()`
- `afd_async_rt_common.h:36` — `TransferStatus { Running=1, Complete=2, Failed=3 }`
- `context.h:39` — `Error { OK=0, BAD_ABI=1, BAD_TASK=2, BAD_LAYER=3, BAD_ROUTE=4, CAPACITY=5, STATE_ORDER=6, TIMEOUT=7, TRANSPORT=8 }`
- `combine_recv.h:436` — `words[0] = code != OK ? 3 : ...; words[1] = code;`

So the pair means `kTransferFailed` with `Error::TIMEOUT`. The layer number in
the message is the rank's own `layer_idx`, not an error code.

The wait that times out is in
`afd_async_combine_recv_rt.h` `finish()`:

```cpp
if (!wait_control(stage, controls + addresses[lane_idx * num_ranks + f].consumed_offset,
                  last_input_bit)) {
    AscendC::WriteGmByPassDCache(error, uint32_t(TIMEOUT));
```

CombineRecv waits for the FFN destination ranks to publish their `consumed`
credit and none arrives. Three observations narrow it:

1. **Not data volume.** `rows=1` fails with the same rank/layer/`[3,7]` set as
   `rows=64`.
2. **Deterministic.** The repeat run is byte-identical in
   ranks, layers and error codes.
3. **Progress stalls after step 0.** In the passing small profile all three
   roundtrip steps advance and the FFN side records transactions in each. In the
   Flash geometry the FFN side records transactions for step 0 and then none, so
   the attention side blocks in step 1 waiting for credit that no FFN round will
   publish.

## Why this points at the Flash FFN side, not the transport

The operators themselves carry the small profile through three full steps, and
the reference pipeline completes a full CAM round. Both use the same four
operators the Flash geometry uses. What differs is the FFN-side transaction
schedule: in Flash, no FFN transaction is issued after step 0. A CombineRecv
timeout is the expected consequence of an FFN side that never runs, rather than
evidence of a second transport defect.

That is a hypothesis from the recorded step boundaries, not a measured cause.
Whoever picks this up should confirm from the FFN-side log why no transaction is
emitted for step 1 in the Flash geometry.

## Reproducing

```bash
export DOCKER_HOST=unix:///var/run/docker-network-lab.sock
N=$(cat /home/z00952334/A3_AFD/work/afd-umdk-e2e/.container-name)

# build at HEAD 7cc7911 with SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1, then install
# the wheel; the DispatchSend object sha256 must change from 2290fc09… to d8d7ad20…

docker exec $N bash -lc 'DEVICES=0,1,2,3 NWORKERS=4 PROFILE=small bash /work/scripts/run_roundtrip.sh'
docker exec $N bash -lc 'DEVICES=0,1,2,3,4,5,6,7 NWORKERS=8 PROFILE=dsv4-flash ROWS=64 bash /work/scripts/run_roundtrip.sh'
docker exec $N bash -lc 'DEVICES=0,1,2,3 NWORKERS=4 bash /work/scripts/run_reference.sh'
```

`PROFILE=dsv4-flash` requires eight ranks: the profile is `4A4F` and the harness
rejects a world size that does not equal sources plus receivers.

## Contents

Each directory holds `roundtrip.log` (or `reference.log`), the per-rank
`plog/` ACL process logs, and `extra-info/` device dumps. The nested
`.gitignore` re-includes `*.log` for this tree; `.gitattributes` marks the
dumps binary.
