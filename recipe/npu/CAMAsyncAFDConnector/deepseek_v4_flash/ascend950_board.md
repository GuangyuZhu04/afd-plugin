# 在另一台 Ascend 950 上复测 Flash

本分支从 11 月 28 日的两份实板报告出发，包含 `ce86c6c` 的框架、修正 TP 层号的回环以及本次 Attention 适配。目标配置为一台主机上的 4A4F、Attention DP2/TP2、FFN EP4、BF16 KV、MXFP4 权重和 eager 执行。完整验收会检查 10 个并发请求的答案和停止原因。

## 准备主机与容器

主机需要分配到的 Ascend 950 设备、匹配的驱动和可用的 HCCL/URMA 网络。准备 **aarch64、Python 3.12 的 A5 开发及推理镜像**，含 CANN toolkit/OPP、与其配套的 `cann_ops_transformer`、torch/torch-npu、vLLM 运行依赖、支持 C++20 `<format>` 的编译器和 libelf 开发头文件。11 月 28 日的故障发生在 950DT / CANN 9.2.1。编译新 wheel 使用的组合是 CANN 9.2.1、torch `2.10.0+cpu`、torch-npu `2.10.0.post7`。其它镜像需用下文的原生算子和 HCCL 检查确认可用性。

```bash
git clone --branch codex/ascend950-umdk-flash-board-bringup --single-branch \
  https://github.com/GuangyuZhu04/afd-plugin.git
cd afd-plugin
export recipe=recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash
export AFD_IMAGE='<这台机器上与驱动匹配的 A5 镜像>'
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
bash "$recipe/setup_ascend950.sh" source
source "$AFD_WORK/stack-env.sh"
```

每次安装在持久卷建立独立 venv，从当前分支构建并安装自带私有 OPP 的插件 wheel，生成 `setup.log`、`wheels/`、`plugin-wheel.sha256` 和激活脚本。上一个成功环境在新安装完成前仍可使用。三种模式如下。

| 模式 | 输入和用途 |
| --- | --- |
| `source` | 从固定提交构建 vLLM `5df63a7203cd`、vLLM-Ascend `bd42eea01d83` 和 DeepEP `25b5153e` / DeepJIT `65f51395`。Ascend 扩展按 `COMPILE_CUSTOM_KERNELS=1` 构建。默认芯片型号为 `ascend950dt_9582`，可用 `AFD_ASCEND_SOC` 指定实际 950 型号。可用 `VLLM_SOURCE`、`VLLM_ASCEND_SOURCE`、`DEEPEP_SOURCE` 指定已有 Git 源或镜像。 |
| `wheels` | 提供 `VLLM_WHEEL`、`VLLM_ASCEND_WHEEL`、`DEEPEP_WHEEL` 的绝对路径及各自构建记录中的 `AFD_UMDK_VLLM_WHEEL_SHA256`、`AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256`、`DEEPEP_WHEEL_SHA256`。Ascend wheel 需包含与目标 toolkit 配套的 `vllm_ascend_C.so` 和 `custom_transformer`，DeepEP wheel 需提供 `AFDBuffer`。 |
| `installed` | 复用基础镜像中已安装的上述固定版本。通过 wheel 安装的 vLLM 与 vLLM-Ascend 仍需提供来自固定构建的可信 SHA-256；干净的固定提交 editable checkout 或带提交记录的 Git 安装也可识别。 |

脚本从基础镜像继承 torch 和机器相关运行依赖，以免安装 vLLM-Ascend 时把匹配的 torch-npu 降级。可用 `AFD_RUNTIME_REQUIREMENTS` 指定基础镜像缺少的运行依赖锁文件，`AFD_WHEELHOUSE` 指定离线 pip 目录。源码模式需要能够取得 Git 子模块和原生构建依赖。完整离线部署可用 wheels 模式。使用本分支预编译插件时，额外传入 `AFD_PLUGIN_WHEEL` 和 `AFD_PLUGIN_WHEEL_SHA256`。脚本校验 wheel 版本中的提交号，`7cc7911` 的旧框架 wheel 只适用于历史回环。切换 CANN 或 torch ABI 后应在目标环境从源码构建。
[本分支插件 wheel、独立 OPP 和 SHA256SUMS](https://github.com/GuangyuZhu04/afd-plugin/releases/tag/ascend950-umdk-flash-board-20261128)

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
| `attention` | 在第一张分配的卡上执行 HcPre/HcPost 零输入回环，再以 1、2、4、8 个 Q 头复测 SAS 发布长度。通过本分支的 builder 拷贝逻辑保留完整输出，用独立缓冲区执行真实 SparseFlashMla，比较每个输出元素与 causal attention 的数值参考。最后执行独立的 QLI metadata 算子。 |
| `hccl` | 先用前两张设备，再用所选的整个通信组，执行 INT32 和 BF16 HcclAllreduce 并校验求和结果。该阶段可以单独定位报告中出现的 AICPU/HCCL 初始化错误。 |
| `roundtrip` | 四卡 1、65 行。分配八卡时继续执行 1、64、4096 行 Flash 回环，逐步核对 BF16 结果、路由记录、零行成员和跨 DP 不同层。 |
| `model` | 使用现有八卡模型验收入口，加载真实 checkpoint，检查两个服务、八个通信 rank、四个 W4A8MXFP executor、十路请求及进程清理。可用 `AFD_NPU_E2E_STARTUP_TIMEOUT` 调整模型启动时间，默认 2400 秒。 |

`all` 在第一个失败阶段退出，返回原命令的非零退出码。每次运行都有独立目录，含 `manifest.txt`、`packages.txt`、`exit-status.tsv`、阶段日志、torchrun rank 日志、`plog/` 和 `extra-info/`。模型阶段的 `workers/<角色>-<PID>.log` 保存子进程 Python 日志及 traceback，`.stderr` 保存 vLLM 日志初始化后的原始标准错误。更早的错误在启动日志中。模型证据在 `model/` 下的 `run.json`、`attention.log`、`ffn.log` 和 `evaluation/completions.json`，以 `acceptance_passed=true` 及清理结果判定完整通过。

## 11 月 28 日结果的适用范围

测试在 `ce86c6c` 上记录了八个 UMDK 缓冲区和四个 FFN executor。按报告对 SAS 做定长拷贝后，原有 19 次尺寸错误降为零。强制打开 A5 的 `RUNTIME_CUSTOM_OPS` 会触发缺失的 `aclnnRmsNormCast`；保留默认开关时，保留的首错为 `libccl_kernel.so / RunAicpuIndOpThreadInit / HcclAllreduce`。将共享常量改为 2048 的那次运行在加载权重前退出，存档未包含首个子进程 traceback。上述运行均未通过模型验收。
[算子缺口报告](../../../../docs/testing/ascend950_umdk_cann_operator_gap_20261128/README.md) · [2048 对照实验](../../../../docs/testing/ascend950_umdk_dsa_metadata_2048_20261128/README.md)

目标 Attention 进程显式加载 HC、压缩比 4 所需的 QLI 原生绑定，沿用 vLLM-Ascend 的 RMSNorm 开关和 BF16 KV 选择。`bd42eea` 仓内 SAS 实现有一个 1024 字消费端检查，而本模型的 BF16 adapter 调用独立安装的 `cann_ops_transformer` 生产者和消费者。报告中的 1024 字截断只验证到了更晚的启动阶段。新版 Ascend 已将 SAS 与 QLI 容量拆开，因此这里也只按 BF16 SAS 生产者实际发布的 1024 或 2048 个 INT32 调整 SAS 缓冲区，完整拷贝所有字。保留 eager 延迟任务持有的 Tensor 对象，同宽度连续执行时复用其存储。QLI 走原有的独立 1024 字路径。
[固定版本的 BF16 adapter](https://github.com/vllm-project/vllm-ascend/blob/bd42eea01d8312c4df469fabaed45f77a12df4d3/vllm_ascend/attention/sparse_flash_mla.py) · [固定版本的 QLI 绑定](https://github.com/vllm-project/vllm-ascend/blob/bd42eea01d8312c4df469fabaed45f77a12df4d3/csrc/torch_binding.cpp)

## 与主仓 vLLM 0.34 的关系

截至 11 月 28 日，官方 afd-plugin `e4ddab17` 固定 vLLM `0.34.0 / 62d4bc3c` 和 vLLM-Ascend `7dcdb690`，安装文档列出的测试工具链为 CANN 9.3.0。#655 已移除 Ascend 的 0.32 兼容路径。MRV2 的 ubatch 准备需要传递 `batch_desc`、`dp_sync_state`，原生执行探针传递 `randomize_inputs`。新版本 Ascend 原生为 SparseFlashMla/MixedQuant SAS 选择 2048 字、为旧 SAS 与 QLI 选择 1024 字，并移除了本分支用于 0.32 的共享常量。
[主仓安装基线](https://github.com/vllm-project/afd-plugin/blob/e4ddab1762a4ba5fee5da492fc13b470d1ff872a/docs/installation.md#ascend-npu-installation) · [0.32 兼容层清理](https://github.com/vllm-project/afd-plugin/pull/655) · [新版 SAS/QLI 容量选择](https://github.com/vllm-project/vllm-ascend/blob/7dcdb690caf879a17ab07382880d5773f5202e20/vllm_ascend/attention/dsa_v1.py#L814-L823)

本分支和安装脚本固定于上文的 0.32 配对。合入 0.34 时应采用其原生 DSA builder，按新 MRV2 和 FFN 启动接口移植插件，使用匹配工具链重新验证扩展、四个通信算子及八卡模型验收。主仓升级记录中的 Ascend 板测对象为 A3。[官方升级记录](https://github.com/vllm-project/afd-plugin/pull/654)
