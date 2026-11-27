# 在 8 张 Ascend 950 上启动 DeepSeek-V4-Flash

这个用例运行两个 eager ModelRunnerV2 服务。前四张设备组成 Attention DP2/TP2，
负责 Hash/普通门控、共享专家和最终加权合并。后四张组成 FFN EP4/TP1，每个 rank
保存连续的一段路由专家。Attention 与 FFN 使用插件内置的四个
`torch.ops.umdk_cam_op_lib.afd_async_*` 算子，在线上传输 BF16。

## 版本与安装

| 组件 | 本用例的版本或配置 |
| --- | --- |
| 算子基线 | afd-plugin `codex/ascend950-umdk-afd`，算子来自 UMDK `cam_feature@dcc0350e` |
| 框架适配 | `codex/ascend950-umdk-dsv4-flash`，重放 [PR #628](https://github.com/vllm-project/afd-plugin/pull/628) 的模型侧改动 |
| vLLM | `0.32.0+empty.g5df63a7`，源码 [5df63a7203cd](https://github.com/vllm-project/vllm/commit/5df63a7203cdb9328d824a27b682e77c7b9928be) |
| vLLM-Ascend | `0.32.0.dev0+gbd42eea`，源码 [bd42eea01d83](https://github.com/vllm-project/vllm-ascend/commit/bd42eea01d8312c4df469fabaed45f77a12df4d3) |
| 注册缓冲区与 FFN 调度 | [支持 AFDBuffer 的 DeepEP-Ascend](https://github.com/puxiaokun3721/DeepEP-Ascend/tree/25b5153eb2549107c13c8d462e325ac28cea8dea) `25b5153e` 与该版本的 DeepJIT 子模块 `65f51395` |
| 权重 | 本地 DeepSeek-V4-Flash W4A8MXFP/FP4 完整 checkpoint，包括 tokenizer 和专家块尺度 |

先合入算子分支，再合入框架分支。查看当前两个分支之间的模型侧差异可用
`git diff codex/ascend950-umdk-afd...codex/ascend950-umdk-dsv4-flash`。

两个角色都检查 vLLM 和 vLLM-Ascend 的完整版本与来源。Git 安装可用 pip 写入的
PEP 610 提交号。Editable 安装需要保留实际导入位置的 Git checkout，HEAD 对应表中
提交，已跟踪文件没有本地修改。离线 wheel 安装使用构建系统为上述提交发布的可信
SHA-256。在两侧服务环境中保留这两个值，并通过 wheel 路径安装，让 pip 记录制品摘要。

```bash
export AFD_UMDK_VLLM_WHEEL_SHA256='<vLLM 固定提交的构建记录中的 SHA-256>'
export AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256='<vLLM-Ascend 固定提交的构建记录中的 SHA-256>'
python -m pip install --no-index --no-deps --force-reinstall "$VLLM_WHEEL" "$VLLM_ASCEND_WHEEL"
```

从本框架分支构建插件，安装生成的 wheel。以下命令在已初始化 CANN 并安装匹配
DeepEP owner 的环境执行。结果目录放在容器外持久卷；工作目录只复制同一提交的测试。

```bash
export SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1
run_dir=$(mktemp -d "${AFD_RESULTS:?set AFD_RESULTS to a persistent absolute path}/flash.XXXXXX")
python -m pip wheel --no-deps --no-build-isolation . -w "$run_dir/wheels"
python -m pip install --no-index --no-deps --force-reinstall "$run_dir"/wheels/vllm_afd_plugin-*.whl
git archive HEAD tests | tar -x -C "$run_dir"
cd "$run_dir"
python - <<'PY'
import torch
import torch_npu
import deep_ep
from afd_plugin.compat.npu.ops import ensure_afd_async_ops_available, get_afd_cust_opapi_path, uses_umdk_async
from afd_plugin.compat.npu.umdk_mxfp_stack import check_umdk_mxfp_stack, umdk_worker_environment

print(check_umdk_mxfp_stack())
print(umdk_worker_environment())
ensure_afd_async_ops_available()
assert uses_umdk_async() and hasattr(deep_ep, "AFDBuffer")
print(get_afd_cust_opapi_path())
print(torch.ops.umdk_cam_op_lib.afd_async_dispatch_recv.default._schema)
PY
```

预检应显示插件私有的 `libafd_umdk_opapi.so`、原生 `custom_transformer` 路径，
以及带 `active`、`recv_src_metadata` 的 DispatchRecv 签名。在上板前按
[950 容器与单算子联调用例](../../../../docs/npu/ascend950_umdk_bringup.md)
配置 URMA，并先运行四卡控制组、八卡模型形状的独立通信测试。

## 启动配置

`dynamicQuant=0` 选择 BF16 UMDK 传输和 MXFP4 FFN。收到的切片可以跨模型层。
FFN 根据每层的本地专家物理行数和对齐偏移调用原生 W4A8MXFP MLP，由 BF16
激活生成组大小为 32 的 MX 块尺度，并写入已注册的 `combine_x`。零行切片也执行
ComputeReady 和 CombineSend。Attention 将路由权重应用一次。

两次 grouped matmul 从激活张量推断原生 E4M3FN 类型，显式指定 packed FP4 权重和
E8M0 尺度。中间的原生 `npu_swiglu_group_quant` 接收当前层的专家行数与模型的
SwiGLU clamp。安装后可用 `SOC_VERSION=950 python -m pytest -q tests/npu/test_umdk_experts.py`
在无卡主机核对投影的 Meta 合同。

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

启动器在创建两组服务前设置 `HCCL_OP_EXPANSION_MODE=AIV`、两个 vendor 的
`ASCEND_CUSTOM_OPP_PATH`、对应的库路径和插件私有 ACLNN 路径。用例目录中的
`umdk-runtime.json` 记录此轮包版本、固定提交、卡号和日志目录；
`packaged-kernels/` 保存这次安装的四个 ELF 及描述文件。默认 plog 与
`extra-info/` 写在本次运行目录，也可通过相应 CANN 环境变量指定持久路径。
判定模型验收时同时查看 `execution`、`phases`、`shutdown` 与 `acceptance_passed`。

## 11 月 26 日故障的证据范围

报告分支 `fb7b1bc` 补传的服务日志来自框架提交 `cbd8e2a`。存档的 Attention
日志在 13:52:06 首次显示 rank 3 从 `recv_status` 拷回状态时收到 507035。
AIV 0 的 MTE 错误码 95 同时涵盖非法 GM 地址和跨设备访问超时。FFN 在
13:52:27 才报告对端断链。`run.json` 中 `phases=[]`、
`acceptance_passed=false`、`cleanup.status=passed`，同时记录了 FFN
管理进程的强制退出。该次启动命令仍带 `enable_dsa_cp=false`，且已到达首轮
前向；省略这个默认键处理的是另一处启动兼容问题。

本次服务日志和独立四卡探针 `rt11` 都出现了同一组相对 PC
`0x1b20`、`0x410c`。在无卡机重建的 CANN 9.2.1 DispatchSend 中，
`0x410c` 对应跨核等待；该版本 CombineRecv 的整个 `.text` 只有
`0x2428` 字节。这个对照给出优先排查的发送阶段。本次板上 plog 与
异常 ELF/dump 已丢失，无法确认实际加载的 ELF 哈希、故障地址或触发指令。
11 月 20 日的 dump、JIT 探针和采用 compact 算子的 Issue #5 各有自己的
代码与用例。模型是否通过仍以本分支下一次真实的八卡运行结果为准。

[11 月 26 日的运行记录与探针](https://github.com/GuangyuZhu04/afd-plugin/tree/fb7b1bc371bba1a2f450fbf6d859a79446bb1c00/docs/testing/ascend950_umdk_dsv4_flash_logs)
