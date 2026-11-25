# 在 8 张 Ascend 950 上启动 DeepSeek-V4-Flash

这个用例运行两个 eager ModelRunnerV2 服务。前四张设备组成 Attention DP2/TP2，
负责 Hash/普通门控、共享专家和最终加权合并。后四张组成 FFN EP4/TP1，每个 rank
保存连续的一段路由专家。Attention 与 FFN 使用插件内置的四个
`torch.ops.umdk_cam_op_lib.afd_async_*` 算子，在线上传输 BF16。

## 版本与安装

| 组件 | 本用例的版本或配置 |
| --- | --- |
| 算子基线 | afd-plugin `codex/ascend950-umdk-afd@b1eb133`，算子来自 UMDK `cam_feature@dcc0350e` |
| 框架适配 | `codex/ascend950-umdk-dsv4-flash`，重放 [PR #628](https://github.com/vllm-project/afd-plugin/pull/628) 的模型侧改动 |
| 推理栈 | vLLM `0.32.0@5df63a72`、vLLM-Ascend `bd42eea0`、匹配的 torch-npu 和 CANN 950 环境 |
| 注册缓冲区与 FFN 调度 | [支持 AFDBuffer 的 DeepEP-Ascend](https://github.com/puxiaokun3721/DeepEP-Ascend/tree/25b5153eb2549107c13c8d462e325ac28cea8dea) `25b5153e` 与该版本的 DeepJIT 子模块 `65f51395` |
| 权重 | 本地 DeepSeek-V4-Flash W4A8MXFP/FP4 完整 checkpoint，包括 tokenizer 和专家块尺度 |

先合入算子分支，再合入框架分支。如果第一条分支在上游采用 squash merge，第二条分支的模型改动范围是 `b1eb133..HEAD`，可以将这个范围重放到新的上游合入点。

两侧 worker 使用同一套环境。从本分支源码执行测试时，在已经初始化 CANN 的环境中安装插件和匹配的 owner，保证 Python 可以从当前 checkout 加载原生扩展。

```bash
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 python -m pip install --no-deps --no-build-isolation -e .
python - <<'PY'
import torch
import torch_npu
import deep_ep
from afd_plugin.compat.npu.ops import ensure_afd_async_ops_available, get_afd_cust_opapi_path, uses_umdk_async

ensure_afd_async_ops_available()
assert uses_umdk_async() and hasattr(deep_ep, "AFDBuffer")
print(get_afd_cust_opapi_path())
print(torch.ops.umdk_cam_op_lib.afd_async_dispatch_recv.default._schema)
PY
```

预编译 wheel 已包含同版 950 OPP。使用 wheel 时，在八卡节点安装本框架分支的 wheel；
从独立目录执行同一提交的 `tests/`，使 Python 加载已安装的插件。预检应显示
`libafd_umdk_opapi.so` 和带 `active`、`recv_src_metadata` 的 DispatchRecv 签名。

## 启动配置

`dynamicQuant=0` 选择 BF16 UMDK 传输和 MXFP4 FFN。收到的切片可以跨模型层。
FFN 根据每层的本地专家物理行数和对齐偏移调用原生 W4A8MXFP MLP，由 BF16
激活生成组大小为 32 的 MX 块尺度，并写入已注册的 `combine_x`。零行切片也执行
ComputeReady 和 CombineSend。Attention 将路由权重应用一次。

本用例设置 2048 MiB 的 HCCL 域、每个 Attention rank 4096 行的源容量、
FP8 indexer K cache 和线性专家分区。FFN 的 `enable_fused_mc2=0` 保留普通
packed 权重及对应尺度供本地专家计算。原始 FP4 checkpoint 可声明 HF `fp8`，
启动器按其元数据选择加载器；ModelSlim checkpoint 使用 `ascend`。本用例覆盖
eager、静态 EP 和单个源物理槽。

把以下环境变量设置为分配到的设备、实际权重目录和可路由的 HCCL 网卡。两组设备按列表前四和后四分配。

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3,4,5,6,7
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V4-Flash
export HCCL_IF_IP='<本机 HCCL 接口地址>'
export HCCL_SOCKET_IFNAME='<通信网卡名>'
export AFD_NPU_E2E_VLLM_BIN="$(command -v vllm)"
scenario=afd-dsv4-flash-async-cam-dp2tp2-ep4
run_dir="$(mktemp -d)"

python -m tests.e2e.models.deepseek_v4_flash.cases \
  --scenario "$scenario" --device-backend npu --devices "$AFD_E2E_DEVICES" \
  --model "$AFD_NPU_E2E_MODEL" --vllm-bin "$AFD_NPU_E2E_VLLM_BIN" \
  --afd-host "$HCCL_IF_IP" --output-path "$run_dir/plan" --describe > "$run_dir/plan.json"

python -m pytest -q -s --basetemp "$run_dir/pytest" \
  "tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[$scenario]"
```

`plan.json` 给出两组实际启动命令。设备测试会启动 Attention 和 FFN，发送 10 个重叠请求，逐条核对 19 到 28 的精确答案及 `finish_reason=stop`，最后检查严格清理。保留
`run.json`、`evaluation/completions.json`、`attention.log` 和 `ffn.log`。
用例从服务日志核对 Attention 0–3 和 FFN 4–7 的 UMDK 注册，以及四个 FFN rank
的 W4A8MXFP executor，将结果写入 `run.json` 的 `phases[0].result.umdk`。
定位收发问题时可设置 `AFD_CAM_OP_IO_LOG=1`，查看四个真实注册缓冲区调用
及接收切片的有效行数和物理行数。
