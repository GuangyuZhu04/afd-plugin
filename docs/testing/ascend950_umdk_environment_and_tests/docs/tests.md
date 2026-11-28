# Tests

Three scripts, cheapest first. The order is deliberate: the roundtrip takes
about a minute and fails on the same transport defects the model case takes
four minutes to report.

| Script | Scope | Dies | Time |
| --- | --- | --- | --- |
| `test/10_roundtrip.sh small` | four operators, exact BF16 oracle, no model | 4 | ~1 min |
| `test/10_roundtrip.sh dsv4-flash` | same, at the model geometry | 8 | ~2 min |
| `test/11_model_e2e.sh` | full attention/FFN services, ten requests | 8 | ~4 min |
| `test/20_diagnose.sh <run>` | reads a finished run's artefacts | — | seconds |

Every script checks for free dies before starting; the host is shared.

Read that check carefully if you bypass it. The summary table that
`npu-smi info` prints at the top can show "No running processes found" while
the die is already busy: the per-die detail is the only reliable source.

```bash
npu-smi info                         # summary: can lag a starting process
npu-smi info -t proc-mem -i 0        # per-die: what the scripts use
```

A process that started seconds ago shows up in `proc-mem` and not yet in the
summary, so a summary-only check will happily start a run into a die another
tenant just took. `afd_free_npus` in `lib/common.sh` counts `proc-mem` lines
for this reason.

## 1. Operator roundtrip

```bash
bash test/10_roundtrip.sh small        # 4 dies, 2A2F
bash test/10_roundtrip.sh dsv4-flash  # 8 dies, 4A4F, the model geometry
bash test/10_roundtrip.sh dsv4-flash 64   # first round uses 64 rows
```

Runs `tests/npu/umdk_roundtrip.py` under `torchrun`. Each rank dispatches its
rows through the registered-buffer operators and compares every returned BF16
element against an independently computed expert-weighted reference, so a pass
means the transport moved the right bytes, not merely that it did not crash.

**Pass:**

```text
UMDK Ascend 950 small numerical roundtrip passed
```

The profile enforces its own rank count: `dsv4-flash` is 4A4F and the harness
rejects a world size that is not sources plus receivers, so it cannot be run on
four dies.

This profile has been observed passing on eight Ascend 950DT dies
(H=4096, topk 6, 256 experts, 4A4F, 4096 rows per source rank): every rank
printed the marker above after matching the BF16 oracle. That makes it a usable
canary — when it fails, the transport is broken; when it passes and the model
case fails, the fault is above the transport.

## 2. Model acceptance

```bash
bash test/11_model_e2e.sh
```

Starts four Attention ranks (DP2/TP2) and four FFN ranks (EP4/TP1) as separate
vLLM services, issues ten overlapping chat requests, and requires strict
cleanup afterwards. Each service loads its own share of the checkpoint, which
takes a few minutes before the first request can be answered.

**Pass:** `acceptance_passed=True`, ten non-empty answers with
`finish_reason=stop`, and `cleanup=passed`.

> This case is a smoke test. Its own README records that A5 can corrupt part of
> a concurrent batch, so a pass means the ten requests were served, not that the
> answers are correct. Read the answer check the case reports rather than
> assuming it.

## 3. Reading a failure

```bash
bash test/20_diagnose.sh /path/to/run
```

The launcher prints a summary; the cause is in the per-role service logs. This
script separates the two failure families that matter:

```text
rundir: model-e2e-r1024

verdict:
  execution        : failed
  cleanup          : passed
  acceptance_passed: False

device faults:
  none (a returned status, not a device trap)

operator errors:
  RuntimeError: call aclnnSparseFlashMla failed, ...
    Invalid_Argument_Tensor_Shape(EZ0009): Parameter metadata of
    SparseFlashMla has incorrect shape (1024).
    Reason: Dimension 0 must be 2048.
```

**Device fault vs returned status** is the first question. A device trap
(`error code = 95`, `aivec error`, `507035`) means the kernel touched memory it
should not have. A returned status means the kernel ran and rejected the work,
which is a contract or protocol problem and is usually visible in the message.

## Status codes you will see

`CombineRecv` returns a `(TransferStatus, ErrorCode)` word pair. It is not a
stage number, which is easy to misread:

| Pair | Meaning |
| --- | --- |
| `[2, 0]` | success |
| `[3, 7]` | `kTransferFailed` + `TIMEOUT`: a peer never published its credit |

`ErrorCode` values: `OK=0`, `BAD_ABI=1`, `BAD_TASK=2`, `BAD_LAYER=3`,
`BAD_ROUTE=4`, `CAPACITY=5`, `STATE_ORDER=6`, `TIMEOUT=7`, `TRANSPORT=8`.

## Evidence layout

Every run writes to `$AFD_WORK/runs/<tag>-<timestamp>/`:

```text
roundtrip.log | e2e.log   launcher and rank output
plog/                     ACL process logs, one per rank process
extra-info/               device dumps on a trap
ranks/                    elastic launcher state
pytest/.../attention.log  per-role service logs (model case)
pytest/.../ffn.log        per-role service logs (model case)
pytest/.../run.json       verdict and phase results
pytest/.../evaluation/    completions.json when the case reached evaluation
```

Keeping the device dump matters: on this project an earlier investigation lost
its dump and could not attribute the trap afterwards.

## Failures seen so far, and what they meant

| Message | Meaning |
| --- | --- |
| `507035` / `error code = 95` | operator trapped on an invalid GM address; transport or memory layout, not the model |
| `The expanded size of the tensor (1024) must match ... (2048)` | a metadata buffer is half the size the operator requires |
| `Parameter metadata has incorrect shape (1024) ... must be 2048` | the same buffer reaching the operator, now rejected by name |
| `AttributeError: npu_hc_pre_v2` | `_C_ascend` is empty: the extension is missing from the installed wheel |
| `aclnnXxx not in libopapi.so` | the toolkit does not ship that kernel; a version mismatch, not a config error |
| `AI CPU kernel execution failed` | an AI CPU kernel failed inside the forward; the preceding log line names it |
| `CombineRecv failed ... [3, 7]` | a peer never published its credit; look at the FFN side, which may not be running that step |
| `Thread export failed ... commEngine[AICPU_TS]` | HCCL could not export its threads to the AI CPU engine; see below |
| `current working operator name is HcclAllreduce` | the exception surfaced at a sync point, not here; the failing collective was an all-reduce started earlier |
| `Engine core initialization failed` | services did not come up; the per-role log has the real exception |
| all ten requests HTTP 500 | services started and rejected the work; run `20_diagnose.sh` |

### The AICPU thread-export failure

One failure family is worth spelling out because its Python traceback names the
wrong function. The sequence, outermost first:

```text
torch.tensor(list(header), dtype=torch.uint8, device=self.device)
  RuntimeError: ... current working operator name is HcclAllreduce
HCCL  hcomm_thread_c_adpt.cc:377  ThreadKernelLaunchForComm failed, ret[15]
HCCL  thread_c_adpt.cc:555        Thread export failed.
                                  Export threadNum[1], commEngine[AICPU_TS]
AICPU RunAicpuIndOpThreadInit (libccl_kernel.so) errorCode=0x2a
```

`ret[15]` is `HCCL_E_RUNTIME` — "call runtime api fail", per
`hccl_types.h` in the toolkit. The `torch.tensor` call is only where an
already-failed asynchronous collective became visible; it is a small host-to-
device copy and is not itself doing an all-reduce.

Two things follow. First, the line that raises is not the line to read; go to
`plog` and match the timestamp. Second, this is an HCCL/AICPU environment
fault, not an operator contract violation: the same four operators pass the
roundtrip above at the same geometry, and the failing group is vLLM's own
communication group, not `group_name_afd_async_cam`.

Set `ASCEND_LAUNCH_BLOCKING=1` when you need the traceback to point at the
real collective rather than the sync that noticed it, at the cost of speed.

## Before blaming the environment

`test/10_roundtrip.sh small` passing tells you the transport, the operator
registration and the device mounts are all fine. If it passes and the model
case fails, the problem is in the model path, not the environment.
