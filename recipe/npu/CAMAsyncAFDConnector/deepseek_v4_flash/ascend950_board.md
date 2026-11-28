# 在另一台 Ascend 950 上复测 Flash

本分支直接基于官方 afd-plugin `main@e4ddab17`，将 UMDK `cam_feature@fee97356` 的四个 `_arch35` 算子和 Flash 框架接入移至 vLLM 0.34。目标配置为一台主机上的 4A4F、Attention DP2/TP2、FFN EP4、BF16 KV、MXFP4 权重和 eager 执行。完整验收会检查 10 个并发请求的答案和停止原因。

## 准备主机与容器

主机需要分配到的 Ascend 950 设备、匹配的驱动和可用的 HCCL/URMA 网络。准备 **aarch64、Python 3.12 的 A5 开发及推理镜像**，含 CANN toolkit/OPP、与其配套的 `cann_ops_transformer`、torch/torch-npu、vLLM 运行依赖、支持 C++20 `<format>` 的编译器和 libelf 开发头文件。本分支面向 CANN 9.3.0、torch `2.10.0+cpu`、torch-npu `2.10.0.post7`。固定的 Ascend Dockerfile 使用 A5 openEuler 24.03 / Python 3.12 工具链镜像 `swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann@sha256:76308e2d7988559b75f88a5c807b0c6b06acb8ce6e8f2dfdf504277e68c41954`。这个开发镜像还需安装匹配的推理依赖。以下示例使用完整的 A5 openEuler arm64 镜像，已从其内部核对两个 Git 提交、干净的 editable checkout、Python 3.12.13 和原生扩展。镜像中的 vLLM 为 `0.34.0+empty`，vLLM-Ascend 的发行标签为 `0.19.1rc2.dev3955+g7dcdb690c`。[镜像构建记录](https://github.com/vllm-project/vllm-ascend/actions/runs/45558867839)

```bash
git clone --branch codex/ascend950-umdk-flash-v034 --single-branch \
  https://github.com/GuangyuZhu04/afd-plugin.git
cd afd-plugin
export recipe=recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash
export AFD_IMAGE=swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend@sha256:d62178d964fdbbfb3543b9c0a76f1fe16d28d7f0a14405e4351df76a6aa8dd03
export AFD_WORK=/data/afd-board
export AFD_NPU_E2E_MODEL=/data/models/DeepSeek-V4-Flash
export AFD_E2E_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP='<本机 HCCL 接口 IP>'
export HCCL_SOCKET_IFNAME='<对应网卡名>'
bash "$recipe/container_ascend950.sh"
```

容器入口按分配的设备列表挂载 davinci 设备、管理设备、UB 端点、驱动和 checkpoint。镜像需要能解析主机相配的 `liburma` 和 provider。已有容器可直接进入下一节。额外的只读 wheel 卷、URMA 或驱动配置卷可作为 `docker run` 参数加在 `container_ascend950.sh` 后。容器中的仓库在 `/workspace/afd-plugin`，持久工作目录在 `/work/afd`，权重在 `/models/DeepSeek-V4-Flash`。裸机使用同样的脚本，自行设置 `AFD_WORK`、`AFD_RESULTS` 和权重的绝对路径。

## 安装测试栈

```bash
cd /workspace/afd-plugin
recipe=recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash
AFD_INSTALL_BUILD_DEPS=1 bash "$recipe/setup_ascend950.sh" installed
source "$AFD_WORK/stack-env.sh"
```

示例在新开的 openEuler 容器中为 DeepEP 安装 GCC Toolset 14 与 libelf 开发包。自备编译器或已有 DeepEP 时可省略 `AFD_INSTALL_BUILD_DEPS`。已有本地 DeepJIT 镜像可用 `DEEPJIT_SOURCE` 指定其绝对路径。

每次安装在持久卷建立独立 venv，从当前分支构建并安装自带私有 OPP 的插件 wheel，生成 `setup.log`、`wheels/`、`plugin-wheel.sha256` 和激活脚本。上一个成功环境在新安装完成前仍可使用。三种模式如下。

| 模式 | 输入和用途 |
| --- | --- |
| `source` | 从固定提交构建 vLLM `0.34.0 / 62d4bc3c7e33`、vLLM-Ascend `7dcdb690caf8` 和 DeepEP `25b5153e` / DeepJIT `65f51395`。Ascend 扩展按 `COMPILE_CUSTOM_KERNELS=1` 构建。默认芯片型号为 `ascend950dt_9582`，可用 `AFD_ASCEND_SOC` 指定实际 950 型号。可用 `VLLM_SOURCE`、`VLLM_ASCEND_SOURCE`、`DEEPEP_SOURCE` 指定已有 Git 源或镜像。 |
| `wheels` | 提供 `VLLM_WHEEL`、`VLLM_ASCEND_WHEEL`、`DEEPEP_WHEEL` 的绝对路径及各自构建记录中的 `AFD_UMDK_VLLM_WHEEL_SHA256`、`AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256`、`DEEPEP_WHEEL_SHA256`。Ascend wheel 需包含与目标 toolkit 配套的 `vllm_ascend_C.so` 和 `custom_transformer`，DeepEP wheel 需提供 `AFDBuffer`。 |
| `installed` | 复用基础镜像中的固定 vLLM 配对。缺少 `AFDBuffer` 时会从固定提交构建 DeepEP，或可用 `DEEPEP_WHEEL` 与 `DEEPEP_WHEEL_SHA256` 提供配套包。通过 wheel 安装的 vLLM 与 vLLM-Ascend 仍需提供来自固定构建的可信 SHA-256；干净的固定提交 editable checkout 或带提交记录的 Git 安装也可识别。 |

脚本从基础镜像继承 torch 和机器相关运行依赖，以免安装 vLLM-Ascend 时把匹配的 torch-npu 降级。可用 `AFD_RUNTIME_REQUIREMENTS` 指定基础镜像缺少的运行依赖锁文件，`AFD_WHEELHOUSE` 指定离线 pip 目录。源码模式需要能够取得 Git 子模块和原生构建依赖。完整离线部署可用 wheels 模式。使用本分支预编译插件时，额外传入 `AFD_PLUGIN_WHEEL` 和 `AFD_PLUGIN_WHEEL_SHA256`。脚本校验 wheel 版本中的提交号，0.32 框架 wheel 适用于历史分支。切换 CANN 或 torch ABI 后应在目标环境从源码构建。
[本分支插件 wheel、独立 OPP 和 SHA256SUMS](https://github.com/GuangyuZhu04/afd-plugin/releases/tag/ascend950-umdk-flash-v034-20261128)

激活文件分别记录测试代码和 wheel 的提交。复用旧 wheel 时，安装脚本会核对两个提交的插件、原生算子和打包输入；运行结果的 `manifest.txt` 记录两者身份。

## 上板执行

```bash
source "$AFD_WORK/stack-env.sh"
export AFD_RESULTS="${AFD_RESULTS:-$AFD_WORK/results}"
bash "$recipe/run_ascend950.sh" all
```

| 可单独执行的阶段 | 检查与证据 |
| --- | --- |
| `preflight` | 执行四个 `_arch35` 入口及专家计算的 Meta 测试，记录加载的扩展、固定包版本、全局 custom-op 开关和各 op_api 库中执行函数/WorkspaceSize 符号对。 |
| `attention` | 在第一张分配的卡上执行 HcPre/HcPost 零输入回环，再以 1、2、4、8 个 Q 头复测 SAS 发布长度。按 0.34 原生 builder 的 2048 字缓冲区拷贝完整输出，并执行真实 SparseFlashMla，比较每个输出元素与 causal attention 的数值参考。最后执行 QLI metadata 并校验独立的 1024 字缓冲区。 |
| `hccl` | 先用前两张设备，再用所选的整个通信组，执行 INT32 和 BF16 HcclAllreduce 并校验求和结果。该阶段可以单独定位报告中出现的 AICPU/HCCL 初始化错误。 |
| `roundtrip` | 四卡 1、65 行。分配八卡时继续执行 1、64、4096 行 Flash 回环，逐步核对 BF16 结果、路由记录、零行成员和跨 DP 不同层。 |
| `model` | 使用现有八卡模型验收入口，加载真实 checkpoint，检查两个服务、八个通信 rank、四个 W4A8MXFP executor、十路请求及进程清理。可用 `AFD_NPU_E2E_STARTUP_TIMEOUT` 调整模型启动时间，默认 2400 秒。 |

`all` 在第一个失败阶段退出，返回原命令的非零退出码。每次运行都有独立目录，含 `manifest.txt`、`packages.txt`、`exit-status.tsv`、阶段日志、torchrun rank 日志、`plog/` 和 `extra-info/`。模型阶段的 `workers/<角色>-<PID>.log` 保存子进程 Python 日志及 traceback，`.stderr` 保存 vLLM 日志初始化后的原始标准错误。更早的错误在启动日志中。模型证据在 `model/` 下的 `run.json`、`attention.log`、`ffn.log` 和 `evaluation/completions.json`，以 `acceptance_passed=true` 及清理结果判定完整通过。

## 11 月 28 日结果的适用范围

测试在 `ce86c6c` 上记录了八个 UMDK 缓冲区和四个 FFN executor。按报告对 SAS 做定长拷贝后，原有 19 次尺寸错误降为零。强制打开 A5 的 `RUNTIME_CUSTOM_OPS` 会触发缺失的 `aclnnRmsNormCast`；保留默认开关时，保留的首错为 `libccl_kernel.so / RunAicpuIndOpThreadInit / HcclAllreduce`。将共享常量改为 2048 的那次运行在加载权重前退出，存档未包含首个子进程 traceback。上述运行均未通过模型验收。
[算子缺口报告](../../../../docs/testing/ascend950_umdk_cann_operator_gap_20261128/README.md) · [2048 对照实验](../../../../docs/testing/ascend950_umdk_dsa_metadata_2048_20261128/README.md)

## 0.34 的模型接口

vLLM 固定到 `62d4bc3c7e33`，用原生 `get_compress_ratios()` 解析旧字段或新版 `layer_types`。vLLM-Ascend 以 `7dcdb690caf8` 源码提交证明身份，其包版本随构建标签而异。Attention 显式加载 HC、压缩比 4 层的 QLI 绑定，FFN 单独加载 SwiGLU 量化绑定。原生 decoder 使用 `post_attention_layernorm`。FFN 沿用主仓的 `reset_mm_cache` 以及 MRV2 / microbatch 调用约定。

`7dcdb690` 的 DSA builder 原生为 BF16 SparseFlashMla 和 MixedQuant SAS 选择 2048 个 INT32，为旧 SAS 与 QLI 选择 1024 个 INT32。0.34 分支使用原生 builder 和每个生产者对应的缓冲区。四个通信算子的调用 schema 和内存协议仍对齐 UMDK。0.32 实板报告只用于追溯历史故障。当前 0.34 的服务就绪、应答正确性和退出清理以本分支在 Ascend 950 的新运行记录为准。
[官方安装基线](https://github.com/vllm-project/afd-plugin/blob/e4ddab1762a4ba5fee5da492fc13b470d1ff872a/docs/installation.md#ascend-npu-installation) · [原生 SAS/QLI 容量](https://github.com/vllm-project/vllm-ascend/blob/7dcdb690caf879a17ab07382880d5773f5202e20/vllm_ascend/attention/dsa_v1.py#L814-L823) · [0.32 独立复测分支](https://github.com/GuangyuZhu04/afd-plugin/tree/codex/ascend950-umdk-flash-board-bringup)
