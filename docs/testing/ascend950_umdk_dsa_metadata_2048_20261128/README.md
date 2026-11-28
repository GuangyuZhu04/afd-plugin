# Widen DSA_METADATA_BUFFER_SIZE to 2048: measured, not adopted

> 2026-11-28　host a5-111 (8x Ascend950DT), CANN 9.2.1
> Parent record: [CANN operator gap](../ascend950_umdk_cann_operator_gap_20261128/README.md)

## Summary

The DSA metadata operator publishes 2048 int32 words. `DSA_METADATA_BUFFER_SIZE`
is 1024, so the upstream slice assignment raises. Widening the constant to 2048
looks like the obvious fix and was tried on hardware. It does not work: the
run regresses and stops before loading a single weight.

**The 2048 constant is therefore not adopted.** The truncating `copy_` form
already committed stays. This note records what was measured so the next
attempt does not repeat it.

---

## 1. The operator contract is 2048

`sparse_flash_mla_metadata` was called directly on device with the adapter's own
kwargs (`layout_kv='PA_BBND'`, `head_dim=512`, one request, four query rows).
The output length does not depend on the geometry:

| `num_heads_q` | output shape |
| ---: | --- |
| 1 | `(2048,)` |
| 2 | `(2048,)` |
| 4 | `(2048,)` |
| 8 | `(2048,)` |

So the mismatch against `DSA_METADATA_BUFFER_SIZE = 1024` is real and is a
property of the operator, not of the launch shape.

| Value | Source |
| --- | --- |
| 1024 | `dsa_v1.py:95`, unchanged upstream |
| 2048 | measured from `cann_ops_transformer.sparse_flash_mla_metadata` |

---

## 2. What widening the constant touches

`DSA_METADATA_BUFFER_SIZE` sizes four buffers, and they are not all fed by the
same operator:

| Line | Buffer | Producer |
| --- | --- | --- |
| 762 | `spec_sas_metadata[]` | SparseFlashMla (2048) |
| 820 | `sas_metadata_buffer` | SparseFlashMla (2048) |
| 823 | `qli_metadata_buffer` | `npu_quant_lightning_indexer_v2_metadata` |
| 1213 | `qli_metadata_buffer` write | same, via bare slice assignment |

The `qli` operator is **not present** in the installed extension:

```
AttributeError: '_OpNamespace' '_C_ascend' object has no attribute
  'npu_quant_lightning_indexer_v2_metadata'
```

Its output length could not be measured, so widening the shared constant may
resize a buffer whose producer still publishes 1024. A bare slice assignment
then raises in the opposite direction. Verified in isolation:

```
buf[:2048] = src_2048   -> OK
buf[:2048] = src_1024   -> RuntimeError
```

---

## 3. Device result

Three runs, the first two already recorded in the parent note:

| Run | `dsa_v1.py` state | attn weight loads | `expanded size` errors | worker log lines (attn / ffn) |
| --- | --- | ---: | ---: | --- |
| `model-e2e-dsafix` | truncate to 1024 (`copy_`) | **4** | 0 | 8411 / 8293 |
| `model-e2e-pyfallback` | truncate to 1024 (`copy_`) | **4** | 0 | - |
| `model-e2e-2048` | constant 2048 + upstream slice | **0** | 0 | **321 / 550** |

The 2048 run removes the size error, which is why it looked promising, but it
never reaches model loading. The worker logs collapse from ~8300 lines to ~500,
and the harness reports workers exiting gracefully rather than crashing:

```
[shutdown] Executor: all workers exited gracefully
RuntimeError: Engine core initialization failed. See root cause above.
Failed core proc(s): {}
```

The failing child's own traceback is swallowed by the vLLM launcher, so the
identity of the triggering exception was not recovered in this pass.

### What this rules out

Widening the constant is **not** a standalone fix. Whatever fails in the 2048
configuration happens earlier than the metadata size matters, and the previous
1024-truncating form reaches further on the same host.

---

## 4. The unresolved question

The operator publishes 2048 and the buffer holds 1024, so truncation discards
half of what the operator wrote. That is the reason the `copy_` form is
suspected of being the source of the later `AI CPU kernel execution failed`
seen in `model-e2e-pyfallback`.

Two facts point in opposite directions:

- truncating to 1024 reaches weight loading and a later aicpu fault;
- widening to 2048 fails even earlier, for a not-yet-identified reason.

Neither is a passing configuration. Closing this needs the 2048 run's child
traceback, which requires capturing worker stderr rather than the launcher
summary.

---

## 5. Reproducing

```bash
export DOCKER_HOST=unix:///var/run/docker-network-lab.sock
N=$(cat /home/z00952334/A3_AFD/work/afd-umdk-e2e/.container-name)

# measure the operator contract
docker exec $N bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
  ASCEND_RT_VISIBLE_DEVICES=0 python3 /tmp/sizes2.py'

# run the acceptance case
docker exec $N bash -lc 'source /work/artifacts/stack-env.sh; \
  STARTUP_TIMEOUT=2400 bash /work/scripts/run_model_e2e.sh'
```

All eight dies must be free. Everything else in the stack is already in place.

---

## 6. State

| Item | State |
| --- | --- |
| Committed `dsa_v1.py` form | truncating `copy_` (unchanged by this note) |
| Test host `dsa_v1.py` | constant 2048, upstream slice form |
| Plugin source tree | clean at `ce86c6c7` |

The device-level experiment is left in the test container only; no source-tree
change accompanies this note because the result did not support one.
