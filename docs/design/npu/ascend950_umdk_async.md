# Ascend 950 registered-buffer AFD transport

The Ascend 950 plugin selects the four operators in
`torch.ops.umdk_cam_op_lib.afd_async_*`. The plugin builds and ships the four
operator directories, ACLNN headers and implementations from UMDK
`cam_feature@dcc0350e`, alongside synchronous A2E/E2A. Install this same
plugin build on all A/F ranks. A separate UMDK wheel would register the same
operator definitions twice. Ascend 910C keeps the
compact `torch.ops.afd_ascend` operators, HCCL windows and existing connector
configuration. The build target selects the 950 backend before communicator
initialization.

| Phase | Attention or FFN responsibilities |
| --- | --- |
| Initialize | All ranks agree on the A/F domain, DP and TP membership, layer IDs, lane capacities and protocol. Create a paired DeepEP `AFDBuffer` per rank, retain the registered GM storage for the session, and prepare the source return or FFN transaction buffers. |
| Dispatch | Attention retains its task, mask, INT64 expert IDs and FP32 weights until combine finishes. DispatchSend publishes into the registered remote payload and control windows. |
| Select and receive | FFN polls arrivals and schedules a frozen transaction. DispatchRecv materializes a slice in the owner's BF16 receive tensor with four routing words per physical row. A transaction can span multiple slices and model layers. |
| Compute and return | Consume complete (layer, local expert) groups. Their aligned physical row intervals include padding. Quantized models need fresh activation quantization and per-token scales for the FFN. Preserve the physical row order in the registered BF16 `combine_x`, signal compute-ready and call CombineSend. |
| Complete | Attention's CombineRecv uses the original route weights and ledger. Release a slot after its return completes. Drain outstanding device work and destroy registered resources before tearing down the HCCL group. |

The implementation baseline for the owner and the FFN poll/scheduler/compute-ready
JIT is [DeepEP-Ascend at 25b5153e](https://github.com/puxiaokun3721/DeepEP-Ascend/tree/25b5153eb2549107c13c8d462e325ac28cea8dea).
Use its matching DeepJIT submodule and CANN environment. The independent
[UMDK caller](https://gitcode.com/openeuler/umdk/blob/dcc0350e149955fcb2d1975bab189749aa9aaba9/test/cam/afd_async_benchmark_runtime.py#L277)
shows the four native calls and registered-buffer lifetimes. The legacy FFN
batch descriptor carries one layer ID. UMDK groups carry an explicit layer
slot and independent valid and physical row counts. Reusing the legacy third
receive output as routing metadata would return expert results to incorrect
tokens.

## Build and run

Build the A5 package in a CANN 9.2 development environment with matching
PyTorch and torch-npu. The UMDK code is already in the source distribution.

```bash
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --no-deps dist/vllm_afd_plugin-*.whl
SOC_VERSION=950 python -m pytest -q tests/npu/test_umdk_meta_dispatch.py tests/npu/test_umdk_meta_combine.py tests/npu/test_umdk_experts.py
```

The 950 wheel exposes both `import umdk_cam_op_lib` and the upstream
`torch.ops.umdk_cam_op_lib` schemas. Its private `libafd_umdk_opapi.so`
contains the UMDK workspace-query and execution functions. The compact 910C
build retains its original namespace and ACLNN prototypes. Load only one
provider of the four UMDK schemas per process.

For serving, install the pinned DeepEP owner and its matching DeepJIT build
on each rank. Select `CAMAsyncAFDConnector`, eager execution and BF16 routed
transport. The W4A8/W8A8 FFN quantizes the received BF16 rows locally.
The connector negotiates the model's layer IDs, divides Attention ranks
into TP members of each DP, and enables independent communication rounds.
The first integration reuses a single physical slot per Attention rank.
It serializes slot reuse through CombineRecv, uses a separate send stream to
allow return credit to advance during multi-round sends, and reads FFN
transaction headers on the host in eager mode. The owner admits capacity for
a whole expert group and its aligned rows. It enforces its own rank, shape,
memory and AIV limits.

With four 950 devices and the serving dependencies installed, run the
deterministic two-Attention/two-FFN check from the plugin checkout.

```bash
torchrun --standalone --nproc-per-node=4 tests/npu/umdk_roundtrip.py
```

The fixture uses nonuniform top-k weights, two layer IDs, different source
lengths, a zero-row source, an empty FFN destination and repeated use of the
same slot. It checks a weighted expert transform against an independent
reference. Real model weights and kernel timings are separate device checks.

## Verification

1. Build the 950 vendor and PyTorch extension. Inspect the package for
   synchronous A2E/E2A and the eight registered-buffer ACLNN exports. Check
   all four Meta schemas and output storage aliases. Check the original
   compact 910C build selection and loader independently.
2. On four Ascend 950 devices start with eager 2A2F, BF16, nonuniform route
   weights and an identity FFN. Compare against an independent weighted sum.
   Exercise zero-route participants, repeated slot reuse, alignment padding,
   mixed layers and multiple slices.
3. Validate the target FFN quantization and real weights, TP row restoration,
   then serving latency and resource lifetime. Graph replay and overlapping
   microbatches require separate device evidence. Host compilation and
   tensor-only tests do not qualify a device run.
