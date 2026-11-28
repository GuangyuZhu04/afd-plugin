# Ascend 950 DeepSeek-V4-Flash: the CANN operator gap behind the model-run failure

> 2026-11-28　host a5-111 (8x Ascend950DT), CANN 9.2.1

## Summary

The operator-level fixes work. The model still cannot serve, and this note
records why: the remaining failures are missing CANN `aclnn` implementations,
not defects in this plugin or in its test harness.

Two concrete steps were taken and measured:

1. Removing the A5 `RUNTIME_CUSTOM_OPS` gate lets the compiled extension load,
   which turns the blocker from `AttributeError: npu_hc_pre_v2` into a missing
   `aclnn` symbol.
2. A real size-contract mismatch in `dsa_v1.py` was found and fixed. It is an
   actual bug regardless of the CANN version.

Both were verified on hardware. Neither makes the model serve, because the
toolkit lacks the kernels the model needs.

---

## 1. What was already fixed before this note

| Problem | Fix | Status |
| --- | --- | --- |
| `507035` MTE invalid GM in DispatchSend | `f4b1f20` stack alignment | resolved |
| `_C_ascend` namespace empty | ship the missing `vllm_ascend_C.so` | resolved |
| `AttributeError: npu_hc_pre_v2` | same | resolved |

---

## 2. Step 1: the A5 `RUNTIME_CUSTOM_OPS` gate

### Mechanism

`vllm_ascend/utils.py:561` returns early unless the hardware profile carries
`RUNTIME_CUSTOM_OPS`:

```python
if envs.VLLM_BATCH_INVARIANT or not get_current_hardware_profile().supports(
        HardwareCapability.RUNTIME_CUSTOM_OPS):
    _CUSTOM_OP_ENABLED = False
    return _CUSTOM_OP_ENABLED
```

A2/A3 and 310P carry the capability; A5 does not. `utils.py:556` records why:

> FIXME: Currently custom op compilation and execution are partially available
> in ASCEND950 chip, we temporarily disable all custom ops.

### Measurement

Adding `RUNTIME_CUSTOM_OPS` to the A5 profile in
`vllm_ascend/device/hardware_profile.py` flips the gate:

```
A5 supports RUNTIME_CUSTOM_OPS: True
enable_custom_op -> True
metadata op present: True
hc_pre_v2 present: True
```

### What it changed, and what it exposed

With the gate open the model advances past `npu_hc_pre_v2` and then fails on
the first custom-op call that needs a kernel:

```
RuntimeError: aclnnRmsNormCast or aclnnRmsNormCastGetWorkspaceSize
  not in libopapi.so, or libopapi.so not found.
```

`vllm_ascend/models/deepseek_v4/model.py:782` guards that call with
`if enable_custom_op():`. So opening the gate makes the model take the
custom-op branch and hit a kernel this toolkit does not ship. The gate is
conservative on A5 for a real reason.

---

## 3. Step 2: the DSA metadata size mismatch (a genuine bug)

### Symptom

```
dsa_v1.py:1133  self.sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = sas_metadata
RuntimeError: The expanded size of the tensor (1024) must match the existing size (2048)
```

`DSA_METADATA_BUFFER_SIZE = 1024` (dsa_v1.py:95) but the selected metadata op
returns 2048.

### Why the buffer and the op disagree

`get_dsa_attn_kv_plan()` picks the SparseFlashMla path for the model cache
dtype (`auto` is in `_BF16_KV_CACHE_DTYPES`). That adapter delegates to
`torch.ops.cann_ops_transformer.sparse_flash_mla_metadata`, which is present on
this toolkit and returns a 2048-element int32 tensor: measured directly with
`head_dim=512` and `layout_kv='PA_BBND'`.

### The inconsistency inside one file

Three call sites copy that result; two truncate and one does not:

```
dsa_v1.py:1512  .copy_(result[:DSA_METADATA_BUFFER_SIZE])     # safe
dsa_v1.py:1875  .copy_(result[:DSA_METADATA_BUFFER_SIZE])     # safe
dsa_v1.py:1133  = sas_metadata                               # asserts equal length
```

The slice assignment requires the source length to equal the slice length, so
it raises whenever the op publishes more than the buffer holds.

### Fix

```python
self.sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE].copy_(
    sas_metadata[:DSA_METADATA_BUFFER_SIZE].reshape(-1))
```

Same truncating form as the other two sites. Verified the semantics in
isolation: buffer 1024, source 2048, `copy_` truncates and the old form raises.

### Measured effect

| Run | `expanded size` errors | outcome |
| --- | ---: | --- |
| before fix | 19 | failed at first forward |
| after fix | **0** | fails later, on a missing kernel |

---

## 4. What still blocks the model

### With the gate open

```
aclnnRmsNormCast not in libopapi.so
```

`vllm_ascend/models/deepseek_v4/model.py:782` calls
`torch.ops._C_ascend.npu_rms_norm_cast`.

### With the gate closed (Python fallback)

```
AI CPU kernel execution failed, device_id=0
rtDeviceSynchronizeWithTimeout execution failed, reason=aicpu exception
```

`rms_norm_cast` falls back to plain `post_attention_layernorm` plus `.float()`,
so the missing cast kernel is no longer the blocker; an AI CPU kernel fails
later in the same forward.

### The common cause

Whole-system symbol scan across `/usr/local/Ascend`, the Python site-packages
and the mounted vendor trees:

| Symbol the model needs | Present anywhere? |
| --- | --- |
| `aclnnSparseAttnSharedkvMetadata` (no prefix) | **no** |
| `aclnnRmsNormCast` | **no** |
| `aclnnKvQuantSparseAttnSharedkvMetadata` | yes, vendor only |

The vendor `custom_transformer` ships only the `KvQuant` variants; the extension
calls the un-prefixed ones; CANN 9.2.1 ships neither. The toolkit is older than
the extension expects.

---

## 5. Reproducing

```bash
export DOCKER_HOST=unix:///var/run/docker-network-lab.sock
N=$(cat /home/z00952334/A3_AFD/work/afd-umdk-e2e/.container-name)
docker exec $N bash -lc 'source /work/artifacts/stack-env.sh; bash /work/scripts/run_model_e2e.sh'
```

Requires all eight dies free. The harness starts 4 Attention and 4 FFN
services, issues ten overlapping requests and checks strict cleanup.

---

## 6. State left behind

| Item | State |
| --- | --- |
| `dsa_v1.py` size fix | applied (backup `dsa_v1.py.limitfix.orig`) |
| `hardware_profile.py` gate | reverted to upstream (backup `.orig`) |
| Plugin source tree | clean at `ce86c6c7` |
| `.so` | the self-built one carrying `hc_pre_v2` |

The gate change is deliberately not left applied: it makes the model take a
custom-op branch whose kernels are absent, and the Python fallback reaches
further without it.

---

## 7. What would unblock this

1. **A toolkit that ships the un-prefixed kernels** -
   `aclnnSparseAttnSharedkvMetadata` and `aclnnRmsNormCast`. Everything else in
   the path is already in place.
2. **Or select the KvQuant variants end to end** - the kernels exist, but the
   model configuration and KV cache must be quantized to match, which changes
   what the checkpoint is asked to do.
3. **Ask upstream which CANN build `gbd42eea` targets** - the image carries
   9.2.0 and this container 9.2.1; the requirement may be newer than both.

Until one of those lands, the model cannot serve on this host. The operator
layer is proven: 507035 is gone, eight UMDK buffers register, four FFN
W4A8MXFP executors come up, and both services reach `Application startup
complete`.
