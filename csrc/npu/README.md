# Ascend NPU Custom Ops Build

AFD CAMP2P depends on plugin-owned Ascend custom operators for
Attention-to-Expert and Expert-to-Attention transfers:

- `torch.ops.afd_ascend.a2e`
- `torch.ops.afd_ascend.e2a`

The 910C and 950 builds also contain four experimental routed-only CAM operators:
`afd_async_dispatch_send`, `afd_async_dispatch_recv`, `afd_async_combine_send`,
and `afd_async_combine_recv`, under the same `torch.ops.afd_ascend` namespace.
Their compact protocol and future connector migration are documented in
[the routed-only operator guide](../../docs/npu/CAM_ASYNC_ROUTED_OPS.md).

The NPU native sources live in this directory:

```text
csrc/npu/ascend_kernels   # ACLNN operator run package (npu_op_* build system)
csrc/npu/pybind           # PyTorch extension (_C_ascend)
csrc/npu/scripts          # build drivers
```

`ascend_kernels/` is organised by operator, with shared headers deduplicated
into `utils/`:

```text
ascend_kernels/
  CMakeLists.txt           # npu_op_package(... TYPE RUN) driver
  AddCustom.json           # msopgen input template
  operator_registry.json   # SOC generation -> operators, plus per-op metadata
  cmake_files/             # cmake/, op_host/, op_kernel/ build fragments
  a2e/{op_api,op_host,op_kernel}
  e2a/{op_api,op_host,op_kernel}
  afd_async_*/{op_api,op_host,op_kernel}  # four routed-only ops
  grouped_matmul_swiglu_quant_v2_layered/{op_api,op_host,op_kernel}
  grouped_matmul_layered/{op_api,op_host,op_kernel}
  utils/op_host/           # CAM host logging/check/tiling-base headers
  utils/op_kernel/         # comm_args.h, data_copy.h, moe_distribute_base.h, layered/
```

The supported SOC generations and the operators compiled for each are declared
in `ascend_kernels/operator_registry.json`; `scripts/select_ops.py` resolves the
operator list from that registry for a given SOC.

## Default Behavior

Ascend custom ops build by default. Set `AFD_BUILD_ASCEND_OPS=0` only when
intentionally skipping the NPU extension, for example on local CPU or macOS
development machines. The Python package remains import-safe when the extension
is not built.

## Build In An Ascend Environment

Install the package with Ascend custom ops enabled. Use
`--no-build-isolation` so pip builds against the current CANN/torch-npu
environment; pip's isolated build environment may not include CANN-provided
Python modules or packages such as `numpy`, `cmake`, and `tbe`.

```bash
SOC_VERSION=910c \
pip install -e . -v --no-build-isolation
```

Common environment variables:

- `ASCEND_HOME_PATH`: CANN toolkit path. Defaults to
  `/usr/local/Ascend/ascend-toolkit/latest`.
- `TORCH_NPU_PATH`: optional path to the `torch_npu` package.
- `SOC_VERSION`: `910c`, `ascend910_93*`, and `ascend910_9392` build
  A2E/E2A plus the four routed-only CAM operators for Ascend 910C.
  `950`, `ascend950*`, and `Ascend950*` build A2E/E2A and the four routed-only
  CAM operators for Atlas A5 (`ascend950`). This is a source/build baseline;
  A5 communication correctness and performance still require device qualification.
- `MAX_JOBS`: number of parallel CMake build jobs for the PyTorch extension.
- `AFD_BUILD_JOBS`: number of parallel CMake build jobs for the ACLNN operator
  project. Defaults to `8`.
- `CMAKE_BUILD_TYPE`: ACLNN operator build type, `Release` (default) or `Debug`.
- `AFD_SKIP_ACLNN_BUILD=1`: skip rebuilding the ACLNN operator package and
  build the PyTorch extension against an existing custom-op installation.

The setup flow calls:

```text
csrc/npu/build_aclnn.sh                 # entry point, selects the SOC argument
csrc/npu/scripts/compile_ascend_proj.sh # msopgen + cmake for one SOC generation
```

`compile_ascend_proj.sh` resolves the operator list through
`scripts/select_ops.py`, generates the project with `msopgen`, patches the
generated `CMakePresets.json` through `scripts/set_conf.py`, then emits a
deterministically named run package at `csrc/npu/output/AFD_<soc>.run`.

Staging copies each selected operator's `op_host/`, `op_kernel/`, and `op_api/`
trees recursively, so operators that keep sources in nested directories (for
example `grouped_matmul_swiglu_quant_v2_layered`, whose tiling implementation
lives in `op_host/op_tiling/`) keep that layout in the generated project.
`cmake_files/op_host/CMakeLists.txt` therefore collects host sources recursively
and hands them to both `npu_op_code_gen` and `cust_optiling`; a top-level-only
collection would drop a nested operator's tiling registration from the run
package without failing the registry or Meta tests. After an Ascend build, that
registration can be re-checked in the packaged library, for example with
`strings .../vendors/afd-plugin/.../libcust_optiling.so | grep -i <OpName>`.

The generated artifacts are installed into the Python package:

```text
afd_plugin/_C_ascend*.so
afd_plugin/_cann_ops_custom/
```

`csrc/npu/output/` and `csrc/npu/build/` are build outputs and are not tracked.

## vLLM-Ascend Coexistence

AFD custom ops must coexist with vLLM-Ascend in the same process:

- The Python extension is owned by this plugin package, for example
  `afd_plugin._C_ascend`.
- A2E/E2A are registered under `torch.ops.afd_ascend`, not under
  vLLM-Ascend's `torch.ops._C_ascend` namespace.
- The CANN custom-op package is installed under the AFD vendor path,
  `afd_plugin/_cann_ops_custom/vendors/afd-plugin/...`.
- The loader uses the package-local `libcust_opapi.so` path through
  `AFD_CUST_OPAPI_LIB_PATH`; it must not rely on a bare
  `dlopen("libcust_opapi.so")`.

Load the extension at runtime with:

```python
from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
```

## Current Scope

A2E/E2A continue to serve the existing CAMP2P connector. The four A3/A5
routed-only CAM operators serve the source-built `CAMAsyncAFDConnector` path.
The A5 baseline preserves the compact-v2 protocol and eager caller lifecycle;
see [the A5 baseline record](../../docs/npu/CAM_ASYNC_A5_BASELINE.md) for build
evidence and the remaining device qualification gates. This branch also contains
[P0 MTE pipelines](../../docs/npu/CAM_ASYNC_A5_MTE_PIPELINE.md) and
[P1 SIMT routing](../../docs/npu/CAM_ASYNC_A5_SIMT_ROUTING.md), with scalar
fallback when the additional SIMT resources do not fit.

The [receive optimization](../../docs/npu/CAM_ASYNC_A5_RECEIVE_OPTIMIZATION.md)
adds ordered CR input prefetch, a local DR two-row cache, and matching DS/CR
source core selection for small batches. These A5 paths preserve compact-v2,
operator keys, and baseline minimum resource checks; hardware performance is
still unmeasured.

The [CR synchronization refinement](../../docs/npu/CAM_ASYNC_A5_COMBINE_SYNC.md)
removes explicit per-tile Scalar waits from the A5 prefetch arithmetic while
retaining vector dependencies, output completion, and the final drain.

The [DS sparse rank refinement](../../docs/npu/CAM_ASYNC_A5_SPARSE_RANKS.md)
uses a token bitmap and a visited list in the existing flag UB arena. It
preserves payload deduplication, all assignment addresses, and token-end offsets.

The [payload pipeline and tile work](../../docs/npu/CAM_ASYNC_A5_PAYLOAD_TILING.md)
adds A5 DS input/wire slots with exclusive quantization-ready events. It retains
metadata staging fences and the original quantization/resource contracts.
It also batches contiguous DR wire rows within expert/source/AIV boundaries,
with exact payload padding removal and scale copies under each tile lease.
CR can select tile-sized FP32 scratch and frozen per-token source descriptors
when this increases its prefetch tile; every hidden element retains logical
top-k order and the original output rounding.

`grouped_matmul_swiglu_quant_v2_layered` is a fused
grouped-matmul + SiLU + dynamic-quantization CAM operator (A8W8 / A4W4 /
A8W4-MSD variants) built for `ascend910_93`. It is registered as
`torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered` (inference + Meta; no
autograd grad) and is excluded from `ascend950` builds. Nothing selects it at
runtime yet; switching a connector/model path to it is a separate follow-up.

A8W8 accepts an NZ weight as either an explicit five-dimensional view or the
three-dimensional logical view `npu_format_cast` produces over NZ storage; the
dimension and shape checks accept both, as the A8W4/A4W4 path already did.
`layer_index` is validated by element count at the ACLNN entry
(`GetViewShape().GetShapeSize() == 1`), not by the storage shape the tiling
context exposes - a rank-1 one-element `int64` tensor can arrive there as rank 2
`[1, 1]`.

Note both attributes reach the tiling only as data, while the tiling cache is
keyed on shape and tiling key: calls that share a shape share the first call's
tiling and its `group_list_type` / `dequant_mode` values. Vary the shape when a
process needs both values of either attribute.

For CANN 9.0.1, the layered ACLNN entry explicitly registers
`grouped_matmul_swiglu_quant_v2_layered.json` as a kernel configuration
candidate. The default runtime name conversion of `V2Layered` does not match
opbuild's filename (issue #372). No extra JSON aliases are installed.

No new NPU correctness, performance, or ACL graph support is claimed without
device validation. The new operators are inference-only and do not transfer
shared-expert payloads.

## Layered GMM device group list

`torch.ops.afd_ascend.grouped_matmul_layered` requires a device `group_list`
tensor. It must be a nonempty 1D `int64` tensor on the same
NPU as `x[0]`, with one entry per expert in each layer's weight tensor.
`group_list_type=1` means counts and passes the tensor directly to ACLNN.
`group_list_type=0` means cumulative offsets, which are converted to counts on
the NPU. It accepts one merged 2D activation in `x`, `split_item=3`, and a
one-element device `int64` `layer_index`. Output rows come from `x[0].shape[0]`;
the caller can supply a capacity-sized activation without a group-count read.
`per_token_scale` is optional at the binding level; A8W4 MSD requires a
capacity-sized device `float32` tensor. An absent `group_list` and other split
modes are not supported by this binding.

The [P2 CCU experiment](../../docs/npu/CAM_ASYNC_A5_CCU.md) builds separately
under `experimental/cam_async_ccu`. It implements device GM descriptors and a
custom CCU segment kernel; it is not selected by the four routed-only operators.

The [P3/P4 experiment](../../docs/npu/CAM_ASYNC_A5_PIPELINE_SLOTS.md) adds a
SIMT descriptor producer and a separately versioned epoch/credit reference
protocol under `experimental/cam_async_pipeline`. It builds independently and
does not select a new transport or wire protocol for the existing operators.
