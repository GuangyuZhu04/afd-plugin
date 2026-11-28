# Widen the SparseFlashMla metadata buffer (vllm-ascend patch)

> 2026-11-28　host a5-111 (8x Ascend950DT), CANN 9.2.1
> Framework under test: `afd-plugin` `ce86c6c7`
> Patched package: `vllm-ascend` `0.32.0.dev0+gbd42eea` (external, not in this repo)

## Why this patch exists

With the operator and model stack otherwise working, the DeepSeek-V4-Flash
acceptance case starts both services, registers all eight UMDK buffers, issues
all ten requests, and then every request fails with HTTP 500. The service log
names one operator and one cause, repeated eighteen times with no other
signature:

```text
RuntimeError: call aclnnSparseFlashMla failed
  Invalid_Argument_Tensor_Shape(EZ0009):
  Parameter metadata of SparseFlashMla has incorrect shape (1024).
  Reason: Dimension 0 must be 2048.
```

`dsa_v1.py` declares one size for two different operators:

| Buffer | Producer | Contract |
| --- | --- | --- |
| `sas_metadata_buffer` | `cann_ops_transformer.sparse_flash_mla_metadata` | **2048** |
| `qli_metadata_buffer` | `npu_quant_lightning_indexer_v2_metadata` | 1024 |

Both were sized from `DSA_METADATA_BUFFER_SIZE = 1024`, so the sas buffer is
half the length the operator requires.

---

## Root cause

Upstream writes the sas buffer with a bare slice assignment:

```python
self.sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = sas_metadata
```

A slice assignment requires the source length to equal the slice length, so it
raises as soon as the operator publishes 2048 into a 1024 buffer:

```text
RuntimeError: The expanded size of the tensor (1024) must match the existing
size (2048). Target sizes: [1024]. Tensor sizes: [2048]
```

That error is upstream correctly reporting an undersized buffer. Truncating the
copy to 1024 silences it but hands the operator half its metadata, which is what
produces the `incorrect shape (1024)` failure above.

The same file already truncates in two other sas paths, which is why the
truncating form looks natural and is still wrong here: those paths only need
whatever fits, while `aclnnSparseFlashMla` validates the full length.

---

## The patch

`dsa_sas_metadata_2048.patch` (5 hunks, applies with `git apply -p1` from a
vllm-ascend checkout root):

1. add `SAS_METADATA_BUFFER_SIZE = 2048` next to the existing constant;
2. size `sas_metadata_buffer` from it, leaving `qli_metadata_buffer` at 1024;
3. replace the bare slice assignment with a truncating `copy_` so an operator
   publishing fewer than 2048 words cannot raise either;
4. widen the draft-path copy that feeds the same buffer.

Nothing else changes: the QLI buffer, its producer and every other consumer keep
their current size.

```bash
cd /path/to/vllm-ascend
git apply -p1 dsa_sas_metadata_2048.patch
```

Verified with `git apply --check` against the pinned `bd42eea` tree: applies
cleanly.

---

## Status: measured, not yet verified end to end

| Step | Result |
| --- | --- |
| Operator contract measured on device | `(2048,)` for `num_heads_q` 1/2/4/8 |
| Patch applies to the pinned tree | yes |
| Patch correctness on hardware | **not confirmed** |

The verification run was started and then stopped because another tenant needed
the dies, so this patch has not produced a passing run. It is the direct
consequence of the operator's own error message, but that is an argument, not a
measurement.

An earlier attempt that only widened the **shared** constant, resizing the QLI
buffer too, failed earlier than this one and is recorded separately in
[dsa_metadata_2048](../ascend950_umdk_dsa_metadata_2048_20261128/README.md).

---

## Contents

```text
dsa_sas_metadata_2048.patch     the fix, applies to vllm-ascend bd42eea
evidence/operator-shape-error.txt   the device error that motivates it
scripts/run_model_e2e.sh        drives the acceptance case
scripts/run_roundtrip.sh        operator-level roundtrip, the cheaper probe
scripts/preflight.sh            operator registration and version gate
```

`run_model_e2e.sh` takes the tests from the current git HEAD with
`git archive` and runs from a per-run temporary directory; `preflight.sh` needs
the wheel digest the version gate checks, which is supplied by
`artifacts/stack-env.sh` in the test environment.

---

## Reproducing the failure without the patch

```bash
export DOCKER_HOST=unix:///var/run/docker-network-lab.sock
N=$(cat /home/z00952334/A3_AFD/work/afd-umdk-e2e/.container-name)
docker exec $N bash -lc 'source /work/artifacts/stack-env.sh; bash /work/scripts/run_model_e2e.sh'
```

All eight dies must be free. Services take roughly three minutes to come up;
the ten requests fail immediately after that.
