# Ascend 950 registered-buffer transport on hardware

Build a 950 plugin wheel and install it with the DeepEP `AFDBuffer` owner at
`25b5153e` and its DeepJIT submodule at `65f51395`. On openEuler the owner
build needs C++20 `<format>`, the matching libstdc++ development package,
elfutils/libelf development headers and the Torch C++ ABI of the installed
wheel. Source the CANN environment in the container before running Python.

## Device and library access

Expose the assigned `/dev/davinciN` devices plus `/dev/davinci_manager`,
`/dev/hisi_hdc`, `/dev/ummu`, `/dev/ubcore` and the required
`/dev/uburma/*` endpoints. Expose the matching Ascend driver libraries and
`/etc/hixlep`. Install or mount the URMA runtime and providers at the paths
and SONAMEs resolved by the container loader. The 9582 host used
`liburma.so.0.0.3`, `liburma_common.so.0.0.1`,
`urma/liburma-udma.so` and `urma/liburma_ubagg.so.0.0.3`.
Use the host's paths and device allocation when constructing mounts.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_OP_EXPANSION_MODE=AIV
export SOC_VERSION=950
# A directory on a persistent host volume, outside the source checkout.
run=$(mktemp -d "${AFD_RESULTS:?set AFD_RESULTS to the mounted results directory}/umdk.XXXXXX")
export ASCEND_PROCESS_LOG_PATH="$run/plog"
export ASCEND_WORK_PATH="$run"
export ASCEND_DUMP_PATH="$run/extra-info"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_DUMP_PATH"
# Run the tests from the same revision as the wheel. Keep sources off PYTHONPATH.
cp -R tests "$run/"
cd "$run"
python -m pytest -q tests/npu/test_umdk_meta_dispatch.py tests/npu/test_umdk_meta_combine.py
```

The four 950 PyTorch calls are `afd_async_dispatch_send_arch35`,
`afd_async_dispatch_recv_arch35`, `afd_async_combine_send_arch35` and
`afd_async_combine_recv_arch35`, under `torch.ops.umdk_cam_op_lib`.
The Meta fixtures explicitly load `afd_plugin._C_ascend` through the plugin
loader and require the 950 registered-buffer provider. They exercise the
schemas without a device or a separate UMDK Python package.

## Communication before the first model forward

Set the visible devices for each standalone job to its allocation. The small
case runs two Attention and two FFN ranks and includes a multi-round source.
The eight-rank case uses DP2/TP2, EP4, 43 layer IDs, BF16 H=4096, K=6,
E=256 and 32 AIVs. The registered window has 4096 rows per source and the
FFN slice allocation has 16384 rows. The first step puts every source on
the same layer with the dummy gate's balanced IDs. Later steps exercise
mixed layers, empty sources/destinations and slot reuse. Each source compares
every returned BF16 element with an independent expert-weighted reference.
It also checks the last published round's metadata byte for byte, including
INT64 expert IDs, FP32 weights, relative token index, mask and padding.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
for rows in 1 65; do
  torchrun --standalone --nproc-per-node=4 --log-dir "$run/small-$rows" --tee 3 \
    tests/npu/umdk_roundtrip.py --profile small --rows "$rows" || exit 1
done
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for rows in 64 4096; do
  torchrun --standalone --nproc-per-node=8 --log-dir "$run/flash-$rows" --tee 3 \
    tests/npu/umdk_roundtrip.py --profile dsv4-flash --rows "$rows" || exit 1
done
```

The first source-side synchronization for a single round checks DispatchSend
publication. A rejected send has no ledger for CombineRecv to consume.
Multi-round sends can require return credit while they are still running.
Keep the elastic rank logs, CANN process logs, exception dump and exact wheel
together. Record the wheel SHA-256 and the loaded kernel descriptors for any
comparison of fault PCs.

### DeepEP reference pipeline

To run the pinned independent CAM pipeline against the plugin, apply the
test-only patch to a temporary copy. It selects the plugin extension and
translates its four call names. DeepEP still owns the buffers and scheduler.

```bash
ref=$(mktemp -d "$run/reference.XXXXXX")
git -C "$DEEPEP_SOURCE" archive 25b5153eb2549107c13c8d462e325ac28cea8dea tests/afd | tar -x -C "$ref"
(cd "$ref" && git apply "$run/tests/npu/deepep_pipeline_cam.patch")
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  --log-dir "$run/reference-ranks" --tee 3 "$ref/tests/afd/run_pipeline_cam.py" \
  --ffn-transaction --local-comm-rounds --hidden 7168 --topk 2 --experts 8 \
  --batch 64 --window 64 --backend-max-rows 4096 --routes balanced --weights nonunit
```

The reference script pins H=7168 for local rounds. Use `umdk_roundtrip.py`
for the model's H=4096 geometry.

## Recorded failure on November 26

On framework commit `cbd8e2a` all eight ranks registered. In the archived
4A4F service run, Attention rank 3 surfaced 507035 at the host copy of
`recv_status` during the dummy forward. AIV core 0 reported MTE error 95
(invalid GM address or cross-device access timeout). The archived four-rank
`rt11` probe reports the same relative PC, `0x1b20`. Its synchronization
covers device work queued by that probe. Neither observation identifies the
offending GM pointer. The FFN store disconnected later. Execution failed,
the evaluation phase never started, and process cleanup passed.

The November 26 device dump and ACL plog were lost. The November 20
CombineRecv object has a different kernel identifier. The earlier compact
CAM model run, the DeepEP JIT run and issue 1110's H=7168 CombineSend fault
each concern a different execution. The targeted 950 operators are present
on upstream `cam_feature` at `fee97356`.

[Archived run, probes and provenance](https://github.com/GuangyuZhu04/afd-plugin/tree/fb7b1bc371bba1a2f450fbf6d859a79446bb1c00/docs/testing/ascend950_umdk_dsv4_flash_logs)
· [UMDK cam_feature](https://gitcode.com/openeuler/umdk/tree/fee97356e28f9945b44ceb3372bac89f9332a515)
· [UMDK issue 1110](https://gitcode.com/openeuler/umdk/issues/1110)

## November 27 sender fix

The five captures on framework commit `893f765` and the independent DeepEP
CAM pipeline all used the same CANN 9.2.1 DispatchSend ELF (SHA-256
`2290fc090ffcd40f47205766b6287f67e1ca6bf63c82f5fd29541aa4c303dae0`).
The raw plogs report a scalar PC of `0x1b20` and an MTE-corrected PC of
`0x1714`, the first GM-to-UB read of the expert IDs. The 72-byte tiling dump
starts with the correctly populated 64-byte operator struct, followed by eight
bytes outside that struct. The compiled entry copies the 64-byte struct.
[Device captures](https://github.com/GuangyuZhu04/afd-plugin/tree/50f8cb9ddf09b5f2948820af10bd27d666a0dd13/docs/testing/ascend950_umdk_dsv4_flash_logs)
· [Independent pipeline and bisection](https://github.com/GuangyuZhu04/afd-plugin/blob/88194220788a6d6fe524c9b0464b12fbacd194f2/docs/npu/ascend950_umdk_507035_root_cause_investigation.md)

In the captured binary, the entry aligns its frame pointer to 32 bytes and
places the worker at `frame_pointer - 1768`, an address ending in 24 modulo 32.
The instructions at `0x1e8..0x200` use bitwise OR with 8, 16 and 24 to
address the first three pointer fields. For that base all three stores go to
the same location. CombineSend has the same collision for its input, source
metadata and context pointers. Both senders now align the actual stack address
before constructing the worker. This preserves their original DMA operations,
queue protocol and record layouts.

With the captured one-row inputs, the old DispatchSend produced IDs
`[0, 0]` in the 950 simulator. The repaired sender published `[0, 1]` and the
entire 64-byte record matched the expected bytes. In a separate two-row
CombineSend test, the old kernel returned execution state `ABORTED/BAD_ABI`.
The repaired kernel returned `COMMITTED/OK`, and its two 512-byte WRITE
descriptors contained the expected BF16 source and destination addresses.
These tests used real ACLNN kernels on simulated device memory, with software
queue completions. Use the four- and eight-device cases above to verify the
physical transport, then run the model acceptance on the framework branch.
