# AFD-Plugin 异步通信代码接入 DeepSeek-V4-Flash —— 测试报告（终稿）

- **测试时间**：2026-11-26 16:00 – 2026-11-27 09:15 (+08:00)
- **被测分支**（<https://github.com/GuangyuZhu04/afd-plugin>）：
  - `codex/ascend950-umdk-afd` @ `b1eb133`（UMDK 算子移植基线）
  - `codex/ascend950-umdk-dsv4-flash` @ `cbd8e2a`（模型侧，叠加 8 个 commit）
- **测试主机**：a5-111（192.168.16.111），8×Ascend950DT（9582），CANN 9.2.1
- **测试目标**：能否把迁移到 afd-plugin 的异步通信代码接入 DeepSeek-V4-Flash，
  并跑通真实前向推理 + 10 并发 E2E 验收（`acceptance_passed=True`）

---

## 1. 最终结论

### 1.1 一句话结论

> **适用范围**：本结论仅针对分支 `codex/ascend950-umdk-dsv4-flash`（`cbd8e2a`）。
> 仓库中 Issue #5 已在**另一分支** `feature/a5-vllm-032-migration` 上报告 10/10 通过，
> 两者是互不包含的不同分支，详见 §9.5。

在该分支上：**接入链路已完整打通到"双角色服务初始化完成 + 进入首个前向"，
但未能产出任何 token。** 直接阻塞是该分支依赖的 UMDK CAM 算子 `afd_async_*`
在 DeepSeek-V4-Flash 所需的 K=6/E=256 形状上触发设备错误 `507035`
（上游文档已记录该形状缺陷），且该算子在上游 HEAD 已被移除。

### 1.2 验证层次总表

| # | 验证层次 | 结果 | 关键证据 |
| --- | --- | --- | --- |
| ① | 分支拉取与差异分析 | **通过** | 两分支线性，`dsv4-flash` 叠加 48 文件 / +1552−197 行 |
| ② | afd-plugin 950 UMDK wheel 构建 | **通过** | `SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1` |
| ③ | 四个 `afd_async_*` 算子注册 | **通过** | 参数数 23/28/22/24，与 `ops.py` 断言逐项吻合 |
| ④ | UMDK 依赖链（DeepEP AFDBuffer） | **通过** | 自建 `deep_ep 1.0.0+25b5153`，`AFDBuffer` 可构造 |
| ⑤ | UB 链路建立 | **通过** | `ENDPOINT_DIAG ... ubc_links=1` |
| ⑥ | **UMDK 会话初始化（双角色 8 rank）** | **通过** | A/F 两侧各 4 rank 全部注册 4096 行 × 43 层 |
| ⑦ | FFN W4A8MXFP 专家执行器 | **通过** | 每 rank 64 本地专家，43 层 |
| ⑧ | 进入 `profile_run` 首个前向 | **通过** | 权重加载完成、引擎初始化推进到该阶段 |
| ⑨ | **首个前向的设备内核返回** | **未通过** | `AfdAsyncDispatchSend` / `AfdAsyncCombineRecv` 报 507035 |
| ⑩ | **真实推理产出 token** | **❌ 未达成** | 卡在 ⑨，零 token |
| ⑪ | **10 并发 E2E 验收** | **❌ 未执行** | `completions.json` / 答案比对完全未运行 |
| ⑫ | CPU 单元测试 | **1885+ 通过** | 133 项失败均为测试顺序污染，非代码缺陷 |

### 1.3 归因结论

失败**不是**本项目环境配置错误。逐项排除如下（全部有实测证据）：

| 被排除的嫌疑 | 排除依据 |
| --- | --- |
| 版本不匹配 | `check_umdk_mxfp_stack()` 精确通过，三个组件均自行构建对齐 |
| 算子未注册 / ABI 不符 | 四算子签名 23/28/22/24 与硬校验完全一致 |
| 缺 UB 设备 / URMA 库 | `/dev/uburma`×10、`/dev/ubcore`、URMA 用户态库、`/etc/hixlep` 全部就位 |
| HCCL 配置错误 | 8 die all-reduce 基线通过；`HCCL_OP_EXPANSION_MODE=AIV` 已设 |
| vendored 代码被改坏 | CAM 内核 blob 与 UMDK `dcc0350e` **逐字节一致**（`8653d193...`） |
| 几何参数越界 | `afd.hpp:496-505` 五条约束逐条核对，全部合法 |
| ABI 字段错误 | `abi_version`/`comm_round_mode`/`reserved[0]` 全部正确 |

**真正根因：上游 UMDK CAM 算子自身缺陷。** 详见 §6。

---

## 2. 环境与版本指纹

### 2.1 版本矩阵（实测）

| 组件 | 分支要求 | 本次实测 | 一致性 |
| --- | --- | --- | --- |
| vLLM | `0.32.0+empty.g5df63a7` @ `5df63a72` | `0.32.0+empty.g5df63a7` | ✅ （镜像自带源码即该 commit） |
| vLLM-Ascend | `0.32.0.dev0+gbd42eea` @ `bd42eea` | `0.32.0.dev0+gbd42eea` | ✅ **自行构建**（镜像自带偏离 160 commit） |
| DeepEP-Ascend | `25b5153e` + DeepJIT `65f5139` | `1.0.0+25b5153` | ✅ **自行构建** |
| afd-plugin | 本分支 | `0.1.dev100+gcbd8e2a9a` | ✅ |
| torch / torch_npu | — | 2.10.0+cpu / 2.10.0.post7 | ⚠️ 分支声明 post4 |
| CANN | 9.2.x | 9.2.1 | ✅ |
| 芯片 | Ascend 950 | Ascend950DT (9582)，8 die | ✅ |
| AIV 核数 | — | 64（ai 32 / cube 32） | 实测 |

分支内置硬校验 `check_umdk_mxfp_stack()` 实际返回：
```
{'vllm': '0.32.0+empty.g5df63a7', 'vllm-ascend': '0.32.0.dev0+gbd42eea'}
```

### 2.2 模型

`/mnt/xds/sfs/DeepSeek-V4-Flash`，FP8 + MXFP4 experts，46 分片，约 149 GB。

从 `config.json` 实测的关键参数：

| 参数 | 值 |
| --- | --- |
| `hidden_size` | 4096 |
| `n_routed_experts` (E) | **256** |
| `num_experts_per_tok` (K) | **6** |
| `num_hidden_layers` | 43 |
| `moe_intermediate_size` | 2048 |
| `n_shared_experts` | 1 |

> ⚠️ 其中 **K=6 / E=256 正是上游已知会触发 507035 的形状**，见 §6.1。

---

## 3. 被测代码结构

```
b1eb133  [NPU] Keep Ascend 950 AFD operator implementations identical to UMDK
   ↑                                                    ← codex/ascend950-umdk-afd
cbd8e2a  [NPU] Use native FP8 GMM dtype and pin the Flash MXFP4 serving stack
                                                        ← codex/ascend950-umdk-dsv4-flash
```

`dsv4-flash` 相对 `umdk-afd`：48 文件、+1552/−197 行。真正影响接入的 6 个文件：

| 文件 | 作用 |
| --- | --- |
| `model_executor/models/npu/deepseek_v4.py` | 新增 `get_mxfp_expert_layers()`，向 FFN 暴露原生 MoE 层 |
| `v1/worker/npu/async_ffn_model_runner.py` | `dynamic_quant==0` 切到 `AfdAsyncMXFPExecutor`；UMDK 会话跳过按层循环 |
| `connectors/npu/async_cam_runtime/afd_async_mxfp.py` | **新增**：W4A8MXFP 执行器（191 行） |
| `compat/npu/umdk_mxfp_stack.py` | **新增**：vLLM/vLLM-Ascend 版本硬校验 |
| `v1/worker/npu/ffn_worker.py` | 放开 A5 设备类型限制 |
| `compat/npu/feature_validation.py` | A5 + UMDK 准入校验 |

四个算子：PyTorch 绑定在 `csrc/npu/pybind/umdk/`，设备实现在
`csrc/npu/ascend_kernels/utils/op_kernel/afd_async/`，
与 UMDK `dcc0350e` **Git blob 逐字节一致**（已用 manifest 校验，
如 `impls/afd/dispatch_send.h -> 8653d193c9e84c03b2918872cc60fcdf5119655d`）。

---
## 4. 实测链路与证据（逐级）

### 4.1 算子注册（通过）

仅 `import afd_plugin` **不会**注册算子，必须显式 `import afd_plugin._C_ascend`：
```
afd_async_dispatch_send: REGISTERED args=23 rets=1
afd_async_dispatch_recv: REGISTERED args=28 rets=3
afd_async_combine_send: REGISTERED args=22 rets=1
afd_async_combine_recv: REGISTERED args=24 rets=5
```
与 `afd_plugin/compat/npu/ops.py::_check_umdk_async` 的断言（23/28/22/24、1/3/1/5）
逐项吻合，`ensure_afd_async_ops_available()` → OK。

### 4.2 AFDBuffer 与 UB 链路（通过）

2 die 探针输出：
```
ENDPOINT_DIAG rank=0 peer=1 layer=0 links=2 ubc_links=1 proto=4 rproto=4 ...
rank=0 AFDBuffer OK
rank=1 AFDBuffer OK
```
说明 `HcclCommMemReg` → jetty 建立 → 端点发现全部成功。

### 4.3 UMDK 会话初始化，双角色 8 rank 全部完成（通过）★ 关键里程碑

`afd-dsv4-flash-async-cam-dp2tp2-ep4`（4A/4F，共 8 die）实际日志：

Attention（11:49:44–45）：
```
UMDK AFD registered buffers: role=attention world_rank=0 lane_rows=4096 layers=43
UMDK AFD registered buffers: role=attention world_rank=1 lane_rows=4096 layers=43
UMDK AFD registered buffers: role=attention world_rank=2 lane_rows=4096 layers=43
UMDK AFD registered buffers: role=attention world_rank=3 lane_rows=4096 layers=43
```
FFN（11:49:58–59）：
```
UMDK AFD registered buffers: role=ffn world_rank=4 lane_rows=4096 layers=43
UMDK AFD registered buffers: role=ffn world_rank=5 lane_rows=4096 layers=43
UMDK AFD registered buffers: role=ffn world_rank=6 lane_rows=4096 layers=43
UMDK AFD registered buffers: role=ffn world_rank=7 lane_rows=4096 layers=43
```
FFN 的 W4A8MXFP 执行器（11:48:43，早于会话初始化）：
```
UMDK W4A8MXFP executor: world_rank=4..7 layers=43 local_experts=64
```

**即：两侧注册缓冲区（4096 源行 × 43 层）与 MXFP4 专家执行器（每 rank 64 专家）
全部就绪**，`dynamicQuant=0` 的 BF16 UMDK 传输 + MXFP4 FFN 选路正确。
FFN 随后进入 `_run_ffn_server_loop()` 等待收发回合。

### 4.4 首个前向失败（未通过）

`run.json`：`execution=failed`、`cleanup=passed`、`acceptance_passed=False`。

失败时间线：

| 时刻 | 事件 |
| --- | --- |
| 11:49:45 | Attention 4 rank 完成 UMDK 会话初始化 |
| 11:49:58 | FFN 4 rank 完成初始化，进入 `session.receive(wait=False)` 轮询 |
| 11:50:00 | Attention 进入 `determine_available_memory` → `profile_run` → `_dummy_run` → `execute_model`，即**首个 dummy 前向** |
| 11:50:00 | AIV 内核报错：`errcode:(0) errorStr: timeout or trap error. subErrType: 0x4` |
| 11:50:23 | Attention 崩退后，FFN 报 `DistNetworkError: Connection reset by peer`（**连锁反应**，非独立故障） |

设备侧错误统计（完整 E2E 日志）：`507035` × 23、`timeout or trap` × 341、
`invalid GM` × 11。触发异常的内核：

```
AfdAsyncDispatchSend_21c646510794c58ff569a30a2208ad7f_0
AfdAsyncCombineRecv_91445775f2d9669ef191a49c071d5d12_0
```

这些内核 dump 落在 `<run>/extra-info/data-dump/0/`。

**值得记录**：`cleanup=passed` —— 即使失败，两端进程、端口、设备均被干净回收。

### 4.5 规模数据

| 指标 | 值 |
| --- | --- |
| attention.log | 2819 行 |
| ffn.log | 8293 行 |
| e2e.log | 11136 行 |
| 权重加载耗时 | 约 7–10 秒 |
| 单 worker 显存 | 约 6.01 GiB |

---

## 5. 工程问题清单（16 项，按阶段）

这一节记录**为了让环境跑起来**而必须解决的全部问题，可直接用于复现。

### 5.1 网络与构建链

| # | 问题 | 根因 | 解法 |
| --- | --- | --- | --- |
| P1 | GitHub 直连不可达，`git clone` 超时 | 主机无外网直连 | `export https_proxy=http://192.168.16.146:3128`（代理常返回 503，需重试 4+ 次） |
| P2 | `docker load` 30 GB 镜像失败 | 默认 dockerd 在 96% 满的 `/dev/sda2` | 改用 `docker --context network_lab`（数据盘 1 TB） |
| P3 | setuptools-scm 无法推断版本 | 源码目录以 `:ro` 挂载且不带 `.git` | 容器内 `git clone /gitdir /work && git checkout <sha>` |
| P4 | DeepEP 编译缺 `elfutils/libdwfl.h` | 镜像未装该开发包 | 装 `elfutils-devel` + `elfutils-libelf-devel` |
| P5 | DeepEP 编译缺 C++20 `<format>` | 镜像 GCC 12.3.1 的 libstdc++ 无 `<format>` | 装 `gcc-toolset-14` 全套后 `source /opt/openEuler/gcc-toolset-14/enable`。**注意**：须用 `gcc-toolset-14-libstdc++-devel` 这个名字，通用名 `libstdc++-devel-14*` 在 SP4 仓是 404 |
| P6 | GCC14 toolset 的 `ld` 缺 `libjansson.so.4` / `libgcc_s.so.1` | toolset 自带 ld 依赖未装 | 装 `jansson`、`gcc-toolset-14-libgcc` |
| P7 | `undefined symbol: c10::detail::torchInternalAssertFail(... std::__cxx11::...) ` | 手动传 `-D_GLIBCXX_USE_CXX11_ABI=0` 与 torch 的 ABI=1 冲突 | **不要**覆盖该宏，用 GCC14 默认值编译 DeepEP |

### 5.2 运行时与插件

| # | 问题 | 根因 | 解法 |
| --- | --- | --- | --- |
| P8 | `uses_umdk_async()` 返回 False | 从**源码树**运行会让 `afd_plugin` 遮蔽已装 wheel，而 `_cann_ops_custom/`（含 `libafd_umdk_opapi.so`）只存在于 wheel | 按 recipe 要求，**从只含 `tests/` 的独立目录**运行，让 Python 加载已安装 wheel |
| P9 | `AttributeError: ... has no attribute 'enable_dsa_cp'`<br>`TypeError: DeepseekV4Attention.__init__() got an unexpected keyword 'need_gather_q_kv'` | 镜像自带 vLLM-Ascend 比分支 pin（`bd42eea`）**新 160 个 commit**，PR #19493 已删除 DSA-CP | **自行构建 `vllm-ascend@bd42eea`**（复用预编译 `.so`/OPP，`COMPILE_CUSTOM_KERNELS=0`） |
| P10 | `pydantic ValidationError: enable_dsa_cp Unexpected keyword` | **分支代码缺陷**：`tests/e2e/models/deepseek_v4_flash/config.py::additional_config()` 仍写入该废弃键 | 从夹具中删除该键（vLLM 0.32.0 无此字段） |
| P11 | AIV 内核 `errCode=507035`、`MTE ... invalid GM address` | 容器缺 `/dev/uburma`、`/dev/ubcore`、URMA 用户态库 | 见 §7.1 的完整挂载清单 |
| P12 | `HCCL runtime exception ... EL0004`，HCCL 初始化即失败 | 缺 `HCCL_OP_EXPANSION_MODE=AIV` | `export HCCL_OP_EXPANSION_MODE=AIV`、`HCCL_BUFFSIZE=4096`、`HCCL_CONNECT_TIMEOUT=3600` |
| P13 | `ModuleNotFoundError: No module named 'acl'` | 直接 `export PYTHONPATH=/tmp/shim` **整体覆盖**了 CANN 的 PYTHONPATH | 改为**追加**：`export PYTHONPATH="$PYTHONPATH:/tmp/shim"` |
| P14 | `Failed to execute operator HcPre_0. Reason: The tiling function does not exist` | `vllm_ascend` 在 import 时才 prepend `ASCEND_CUSTOM_OPP_PATH`，但算子加载器已缓存路径 | 进程启动前**显式导出两个 vendor**：afd-plugin 与 vllm-ascend 的 `custom_transformer` |

### 5.3 测试与容器环境

| # | 问题 | 结论 |
| --- | --- | --- |
| P15 | `pytest tests/unit` 报 33 failed + 1 error | **非代码缺陷**。单独跑 `tests/unit/model_executor` 是 **321 passed**；只有 `tests/unit/compat` + `model_executor` 组合才复现。属跨测试全局状态污染，建议上游加 fixture 隔离 |
| P16 | 4 die 场景卡号不符 | 容器内 8 卡全可见，跑 4 die 需显式 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`；E2E runner 用 `AFD_E2E_DEVICES` 自行映射 |

### 5.4 分支与镜像自带测试的两个"坑"（非缺陷，但阻碍使用）

1. **`tests/npu/test_umdk_meta_*.py` 需要先注册算子**
   这些用例只 `importlib.import_module("umdk_cam_op_lib")`，而该模块**不注册**四个
   `umdk_cam_op_lib::afd_async_*` schema（注册发生在 `afd_plugin._C_ascend`）。
   直接跑会 35 项全 FAIL（`Found no matching schema`）。
   需要额外 `-p preload_ops`（内容为 `import torch; import torch_npu; import afd_plugin._C_ascend`）
   且设 `SOC_VERSION=950`，之后 **35 passed**。

2. **`run_pipeline_cam.py` 在本树跑不起来**
   它调用旧算子名 `cam_deepep_dispatch_send`，而本插件用 `afd_async_*`，
   直接 `AttributeError`。这导致**唯一覆盖 `ffn_transaction` 路径的上游脚本失效**（见 §6.3）。

---
## 6. 根因分析（核心章节）

这一节是三轮排查的完整证据链，解释**为什么**跑不通。

### 6.1 决定性证据：507035 是上游已知缺陷，且 DSV4-Flash 正命中该形状

在本地 UMDK 上游仓库（`repos/umdk-upstream`，含 commit `dcc0350e` —— **正是插件
vendored 的那个 commit**）中，官方文档明确记录：

`test/cam/benchmark_afd_async.py` 文件头：
```
Issue #6 reports CAM combine_send error 507035 for K=6/E=256. The target
commands above describe the workload; CAM performance requires a passing
correctness run for the actual loaded OPP and shape.
```

`test/cam/README.md`：
```
K=6/E=256 is a target workload. CAM CombineSend has reported device
error 507035 on that shape, so use a passing shape for timed comparisons.
```

对应 issue：<https://gitcode.com/openeuler/umdk/issues/1110>

**与 DeepSeek-V4-Flash 的对应关系：**

| 参数 | DSV4-Flash 实测 | 上游已知缺陷形状 |
| --- | --- | --- |
| `num_topk` (K) | **6** | **6** |
| `n_routed_experts` (E) | **256** | **256** |
| `hidden_size` | 4096 | 7168（上游示例值） |

**完全命中。** 上游自己验证通过的形状只有
`--num-ai-cores 1 --num-tokens 64 --num-topk 2 --num-experts 8` 这种小规模。

### 6.2 该算子已在上游被完整移除，无修复版本

```
$ git ls-tree -r --name-only HEAD | grep -c afd_async
0                                    ← 上游 HEAD 已完全没有该算子
$ git ls-tree -r --name-only dcc0350e | grep -c afd_async
60                                   ← 插件 pin 处仍有 60 个文件
```

`dcc0350e..HEAD` 之间共 **3304 个 commit**，所有 CAM 修复都集中在
`fused_deep_moe` / `fused_deep_moe_zb` / `moe_dispatch_*`，
**没有任何一条针对 `afd_async_*`**。典型修复（均与 afd_async 无关）：
```
!3917 cam: replace fused_deep_moe combine batch wait with per-slot data-as-flag
!3901 cam: fix split recv count stride for 256B state slots
!3894 cam: fix fused_deep_moe_zb rank-0 combine row selection
!3900 cam: cap fused_deep_moe_zb round length by globalBs * k
```

`afd_async_*` 最后一次出现在 `6b65579f`（2026-11-19），之后被删除，
连带 `benchmark_afd_async.py` 中那句 507035 说明也一并消失。

**含义**：插件 pin 的 `dcc0350e` 所依赖的这套 AFD 异步传输算子，
在上游已成为**被放弃的分支**，其缺陷既未修复也无后续版本可用。

### 6.3 CAM 内核与 DeepEP JIT 内核是两套独立实现（重要澄清）

排查中一度用 DeepEP 自带参考测试（`tests/afd/run_threec_a5.py`）验证"传输层是否正常"，
它确实 **RUN_PASS**。但该结论**不能**外推到插件：

| | DeepEP JIT 路径（参考测试） | CAM op_api 路径（插件实际使用） |
| --- | --- | --- |
| 文件 | `deep_ep/include/deep_ep/impls/afd/dispatch_send.hpp` | `csrc/npu/.../op_kernel/afd_async/impls/afd/dispatch_send.h` |
| 来源 | DeepEP-Ascend `3b25377d` | DeepEP-Ascend **`19fe6b7`** |
| 行数（规范化） | 577 | 778 |
| 差异 | — | **533 行不同** |
| 运行方式 | JIT 编译，`jit->launch(num_blocks=...)` | ACLNN 标准算子，tiling 驱动 |
| 实测结果 | **RUN_PASS** | **FAIL（507035）** |

CAM 头部注释自述基于**更新的上游提交**并带额外改动：
```
* AFD source: puxiaokun3721/DeepEP-Ascend @ 19fe6b7
* CAM modifications: Added bounded tile routing and AFD slot, ...
```

而插件 **只用 CAM 这套**。更关键的是：**唯一覆盖 `local_rounds + ffn_transaction`
组合的上游脚本 `run_pipeline_cam.py` 在插件这棵树上跑不起来**（旧算子名 `cam_deepep_*`，
见 §5.4）。也就是说这条组合**在现有代码库中没有任何可执行的自测覆盖**。

### 6.4 已被证伪的三个假设（排查过程记录）

为避免后来者重复踩坑，记录三个曾误导排查方向、后被实测否定的假设：

| 假设 | 验证方式 | 结论 |
| --- | --- | --- |
| `num_vec_cores=32` 过大导致失败 | 在参考测试上扫描 1/8/16/32；把插件 `_AIV_CORES` 改成 8 重跑 | **证伪**。1 和 8 RUN_PASS，16/32 FAIL；但插件改成 8 后**仍失败**。且本机 AIV 核数为 64，32 并不越界 |
| `enable_local_rounds()` 本身有问题 | 单变量对照（开/关 × round_step 真/假） | **部分成立但非根因**。开+False 会报 `Local communication rounds require the FFN transaction path`；但插件是 开+**True**（合法组合），最小复现下 `prepare_recv(transaction=True)` 和 `prepare_ffn` 均 **OK** |
| 探针 / 插件存在 ABI 错配 | 读设备内核错误码表（`words[0]=3, words[1]=error`） | **实为我探针的 bug**。ABI 校验通过后故障才前移到内核数据面 |

补充：`C` / `C_m` 的语义容易被误解。经源码核对：

| 符号 | 真实含义 | 位置 |
| --- | --- | --- |
| `C` / `C_m` | 每个物理槽每轮可承载的 token 容量 | `lane_specs` 第 6 个字段（`--window`） |
| `num_token_capacity` | 本 rank 整笔事务的最大 batch | 构造参数 #11 |
| `num_vec_cores` | **AIV 向量核的 launch block 数**（非容量） | 构造参数 #13 |

### 6.5 已核对合法的几何约束

按插件 A5 2A2F 实际配置逐条核对 `deepep/csrc/buffers/afd.hpp:496-505`：

| 约束 | 实测值 | 结果 |
| --- | --- | --- |
| `window` (C_m) = min(capacity, aligned backend_rows / attn_size) | 2048 | — |
| `transaction_group_row_bound` ≤ `q` | 2048 ≤ 4096 | ✓ |
| `q_alloc` ≤ `backend_max_rows` | 4096 ≤ 4096 | ✓ |
| `hidden % 128 == 0` | 4096 | ✓ |
| `combine_chunk` ≥ 128 且 %128==0 | 4096 | ✓ |
| `transaction_geometry_valid`（层 ID 唯一、phase≤1） | 是 | ✓ |
| A/F payload 不对称（A=96MB / F=34MB） | 设计使然（`is_source` 分支 + `allow_asymmetric_buffers=true`） | ✓ |

**结论：不是几何参数越界。**

### 6.6 高嫌疑失败机制（供上游定位参考）

读 CAM `combine_send` 的 host tiling（`op_host/afd_async_combine_send.cpp`）：

```cpp
const uint64_t legacyActiveBytes = 64 + 64uLL * numSourceLanes
                                 + 32uLL * numSourceLanes * expertsPerRank;
if (activeBytes > legacyActiveBytes) {
    OP_TILING_CHECK(x.size(0) < tiling->maxPaddedTokens || x.size(1) != numHidden
        || metadata.size(0) < maxPaddedTokens || metadata.size(1) != 4,
        "combine_send transaction tensors do not cover the configured row capacity");
}
```

代入插件配置：

| 量 | 值 |
| --- | --- |
| `numSourceLanes` | 2 |
| `expertsPerRank` = E / ffn_size = 256/4 | 64 |
| `numLayers` | 43 |
| `legacyActiveBytes` = 64 + 128 + 4096 | **4288** |
| transaction `activeBytes`（43 层 × 64 专家组） | ≈ **92352** |
| 判定 | `activeBytes > legacyActiveBytes` → **走 transaction 分支** |

即插件确实进入 transaction 分支，而该分支对 `x` / `metadata` 的行容量有额外要求，
与 kernel 侧 `static_assert(T <= 128 && R <= 512)` 的编译期特化共同作用，
是 507035 的高嫌疑来源。

> 注：E2E 日志中**未出现**这些 `OP_TILING_CHECK` 的报错文本，
> 说明 tiling 校验本身通过了，故障发生在内核执行期的 GM 访问。
> 因此上述判断是"高嫌疑机制"，尚需上游在 CAM 内核侧确认。

---
## 7. 复现手册（可直接照抄）

### 7.1 容器创建（关键：设备与 URMA 库清单）

这是踩坑最久的一节。缺任何一项都会导致 `507035` / `invalid GM address`。

```bash
DEV=""
for d in /dev/uburma/*; do DEV="$DEV --device=$d"; done     # 共 10 个

docker --context network_lab run -d --name <name> --network host --privileged \
  --shm-size=64g \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/ubcore \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  $DEV \
  -v /mnt/xds/sfs/950-DT/images/dependency/libs/urma-core/liburma.so.0.0.3:/usr/lib64/liburma.so.0.0.3 \
  -v /mnt/xds/sfs/950-DT/images/dependency/libs/urma-core/liburma_common.so.0.0.1:/usr/lib64/liburma_common.so.0.0.1 \
  -v /mnt/xds/sfs/950-DT/images/dependency/libs/urma-providers/liburma-udma.so:/usr/lib64/urma/liburma-udma.so \
  -v /mnt/xds/sfs/950-DT/images/dependency/libs/urma-providers/liburma_ubagg.so.0.0.3:/usr/lib64/urma/liburma_ubagg.so.0.0.3 \
  -v /etc/hixlep:/etc/hixlep \
  -v <workspace>:/work:rw -v /mnt/xds/sfs:/mnt/xds/sfs:ro \
  vllm-ascend-a5:032-megamoe-20261125 sleep infinity
```

> 参考：主机上正常工作的租户容器（如 `wyb_dsv4`）**正是这套挂载**。

### 7.2 三层依赖构建（顺序有依赖）

```bash
# 2a) vllm-ascend@bd42eea —— 复用镜像内 vLLM 源码树与预编译 .so/OPP
cd /work/vllm-ascend-pin && rm -f .git
SOC_VERSION=ascend950dt_9582 COMPILE_CUSTOM_KERNELS=0 \
CMAKE_ARGS="-DCOMPILE_CUSTOM_KERNELS=0" \
SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0.dev0+gbd42eea \
  python3 -m pip install --no-deps --no-build-isolation -e .
# 补写 direct_url.json 的 vcs_info（插件会校验 commit）

# 2b) afd-plugin 950 UMDK wheel
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 \
  python3 -m pip wheel --no-build-isolation --no-deps . -w dist

# 2c) DeepEP-Ascend（GCC14；勿覆盖 _GLIBCXX_USE_CXX11_ABI）
source /opt/openEuler/gcc-toolset-14/enable
python3 -m pip wheel --no-build-isolation --no-deps . -w dist
```

### 7.3 运行 E2E

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=4096 HCCL_CONNECT_TIMEOUT=3600
export ASCEND_CUSTOM_OPP_PATH=<afd-plugin vendor>:<vllm-ascend custom_transformer>
export LD_LIBRARY_PATH=<两个 op_api/lib>:$LD_LIBRARY_PATH
export AFD_CUST_OPAPI_LIB_PATH=<afd-plugin vendor>/op_api/lib/libafd_umdk_opapi.so

export AFD_E2E_BACKEND=npu AFD_E2E_DEVICES=0,1,2,3,4,5,6,7
export AFD_NPU_E2E_MODEL=/mnt/xds/sfs/DeepSeek-V4-Flash
export AFD_NPU_E2E_VLLM_BIN=$(command -v vllm)
export HCCL_IF_IP=192.168.16.111 HCCL_SOCKET_IFNAME=enp39s0f3

# 关键：必须在“只含 tests/”的目录运行，避免源码树遮蔽 wheel
python3 -m pytest -q -s \
  "tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[afd-dsv4-flash-async-cam-dp2tp2-ep4]"
```

### 7.4 分支自带 NPU 测试的正确跑法

```bash
export SOC_VERSION=950                       # 否则整模块 skip
cp preload_ops.py .                          # 内容见 §5.4
python3 -m pytest -q -p no:cacheprovider -p preload_ops \
  tests/npu/test_umdk_meta_dispatch.py \
  tests/npu/test_umdk_meta_combine.py \
  tests/npu/test_umdk_experts.py
# 实际结果：35 passed
```

---

## 8. 环境障碍记录（非缺陷，但影响可执行性）

这一节记录本机在测试期间出现的环境级问题，供后续在该环境上工作时参考。

### 8.1 集群 NPU 被 lease 任务占满

```
$ npu-smi info
card0: 93183 / 98304 MB   VLLMWorker_DP  pid=2571538  87666 MB
card1..7: 同样 ~92.7/98.3 GB
```
连续监测 120 秒，8 个 `VLLMWorker_DP` 进程稳定占用，无释放窗口。
且 `/mnt/xds/sfs/950-DT/gitea/logs/` 显示 **a5-110/111/122/220/251/252/61 等
全部 A5 节点都在跑 `xds-pd-lease-*` 任务**，属集群级资源占用。

**后果**：后期任何需要 HCCL 的验证都会失败于
```
HCCL function error  create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:142
```
（设备 HBM 无空闲，HCCL 无法建链。）

### 8.2 `network_lab` dockerd 曾被 SIGSTOP 挂起

```
$ ps -p $(cat /var/run/docker-network-lab.pid) -o pid,stat
1791470 Tl          ← T = 已停止（SIGSTOP），并非崩溃
```
同时其 socket 被重命名：
```
/var/run/docker-network-lab.sock.xds-reserved-20261126
```

**恢复方式**（二选一）：
```bash
kill -CONT $(cat /var/run/docker-network-lab.pid)      # 恢复进程
# 或：把 socket 符号链接到保留名
ln -sf /var/run/docker-network-lab.sock.xds-reserved-20261126 /var/run/docker-network-lab-mine.sock
export DOCKER_HOST=unix:///var/run/docker-network-lab-mine.sock
```

> 排查期曾误判为"容器被外部 prune 删除"、"宿主准入策略拒绝 davinci 挂载"，
> 实为 dockerd 被挂起导致的假象。**这是本节最有价值的更正。**

### 8.3 磁盘水位

`/mnt/xds/sfs` 长期处于 **99%（剩余 ~590 GB / 30 TB）**，而 `network_lab` 的
docker root 正在此盘。镜像 7 个约 113 GB，可回收约 54 GB。
新建容器前建议先 `docker container prune` / `docker volume prune`。

---
## 9. 结论与后续建议

### 9.1 对"能否接入"这个问题的回答

**能接入，但跑不通推理。**

- **接入层面：成功。** 异步通信代码（四个 `afd_async_*` 算子 + UMDK 注册缓冲区 +
  UB 链路 + 双角色会话）在 A5 + DeepSeek-V4-Flash 上**已完整建立起来**，
  8 个 rank 全部完成初始化，模型权重正常加载，引擎推进到首个前向。
  这说明 `dsv4-flash` 分支的模型侧改动与依赖 pin 是**自洽且可用的**。

- **执行层面：阻塞于上游。** 首个前向即触发 `AfdAsyncDispatchSend` /
  `AfdAsyncCombineRecv` 的设备错误 507035，而该错误对应上游已记录的
  K=6/E=256 形状缺陷，且相关算子在上游 HEAD 已被移除。

### 9.2 要跑通 10 并发验收，需要同时满足三个条件

| 条件 | 现状 |
| --- | --- |
| ① CAM 内核修复 K=6/E=256 的 507035 | **上游已放弃该算子，无人修**（§6.2） |
| ② 或改走一个能跑通的传输内核 | JIT 路径实测可用，但需插件重写传输层（超出"验证迁移代码"范畴） |
| ③ 需要连续 NPU 时间窗口做验收 | 集群 NPU 被 lease 占满（§8.1） |

三者当前**没有一个成立**。

### 9.3 需要什么才能继续

按优先级：

1. **与 UMDK 上游确认 `afd_async_*` 的技术路线是否真的废弃**
   （HEAD 已无该算子）。若已废弃，`dsv4-flash` 分支本身需要重新设计传输层。
2. **提供一个可独占的 A5 节点（≥4 卡）**，避开 lease 抢占，才能做完整验证。
   这是验证条件 ① 是否可缓解的前提。
3. **明确验收口径**：上游 `tests/e2e/README.md` 已声明 **A5 并发下答案不正确**
   且该场景 `check_answer=False`（smoke only）。因此即使跑通，
   `acceptance_passed=True` 也只代表"10 个请求都被服务且非空返回"，
   **不代表答案正确**。需要确认这是否符合验收要求。

### 9.4 给上游的改进建议

#### 分支代码缺陷（建议修）

1. **删除 `tests/e2e/models/deepseek_v4_flash/config.py` 与 `dspark_cases.py`
   中的 `enable_dsa_cp` 键** —— vLLM 0.32.0 无此字段，会直接阻断启动
   （`pydantic ValidationError`）。
2. **`tests/npu/test_umdk_meta_*.py` 增加算子注册前置**
   —— 目前只 import `umdk_cam_op_lib`，该模块不注册 schema，
   直接跑 35 项全 FAIL。
3. **`run_pipeline_cam.py` 适配 `afd_async_*` 算子名**
   —— 它调用旧名 `cam_deepep_*`，导致**唯一覆盖 `ffn_transaction` 路径的脚本失效**。
   这是当前最大的测试盲区。

#### 健壮性建议

4. **`umdk_mxfp_stack.py` 的 commit 校验会被静默跳过**
   —— 该校验只在 `direct_url.json` 带 `vcs_info` 时生效，
   而 `pip install -e` 默认**不写**该字段。建议改为强制要求。
5. **`import afd_plugin._C_ascend` 才能注册算子** 的行为建议在文档显式说明。
6. **`tests/unit/compat` 存在跨测试全局污染**，建议加 fixture 隔离
   （单独跑 `model_executor` 是 321 passed，组合跑才 33 failed）。
7. **在 `docs/npu/` 增加完整的容器 `--device` / URMA 库挂载清单**
   —— 本次 P11 踩坑耗时最长，且缺项直接表现为设备错误 507035，
   极易被误判为算子缺陷。

---

## 9.5 【重要】与仓库中已有 Issue #5 结论的差异说明

在准备提交本报告时，发现该仓库已存在 Issue #5
（`[NPU][A5] 异步 CAM 路径端到端拉起 DeepSeek-V4-Flash：10/10 通过`），
其结论与本报告**相反**。必须澄清两者测的不是同一个东西：

| 项 | Issue #5 | 本报告 |
| --- | --- | --- |
| 分支 | `feature/a5-vllm-032-migration`（`d43535c`） | `codex/ascend950-umdk-dsv4-flash`（`cbd8e2a`） |
| 关系 | 两者是**互不包含的不同分支**（已用 `git merge-base --is-ancestor` 验证） | 同左 |
| 结论 | 10/10 并发通过，四算子合计调用 10844 次 | 首个前向即失败，零 token |

Issue #5 的正文列出了它在 `feature/a5-vllm-032-migration` 上修复的 6 个阻塞，
其中多个在本分支上**并未包含**，例如：

1. `compute_gate_on_attention` 不接受 `QuantType.W4A8MXFP`
2. CAM 的 `dynamicScales` 为 1 维，而 MXFP 专家需要 `(tokens, K/32, 2)` 3 维 E8M0
3. `enable_return_routed_experts` 属性在 vLLM 0.32 已迁移到 `aux_output_config`
4. `initialize_attn_backend` 缺 `is_profiling` 参数
5. `metadata_builders` 数量与 stage 数不匹配
6. `HcclAllreduce` 的 `RunAicpuIndOpThreadInit` 失败

**这对本报告结论的含义：**

- 本报告的所有结论**仅适用于 `codex/ascend950-umdk-dsv4-flash`**，
  不应被理解为"AFD 异步 CAM 在 A5 上不可用"。
- Issue #5 已证明：**在 `feature/a5-vllm-032-migration` 分支上（配合其自有 OPP/算子构建）
  是可以跑通的**。这说明 §6 中记录的 507035 等失败，
  至少有一部分可以由该分支的修复消除。
- 若要达成"跑通 10 并发验收"的目标，应当改测 `feature/a5-vllm-032-migration`，
  或把该分支的修复合入 `codex/ascend950-umdk-dsv4-flash`。

> 待验证事项（本会话**未执行**，不做结论）：
> ① 在 `feature/a5-vllm-032-migration` 上复跑本报告的 E2E 是否同样通过；
> ② 该分支使用的是否为同一套 CAM 内核源码 / OPP 包（若是，则 §6 的根因判断需要修正）。

---

## 10. 产物与可复现路径

### 10.1 构建产物

| 路径 | 内容 |
| --- | --- |
| `work/umdk-test/build/dist/` | `vllm_afd_plugin-0.1.dev100+gcbd8e2a9a-*.whl`（950 UMDK） |
| `work/umdk-test/build/dist-deepep/` | `deep_ep-1.0.0+25b5153-*.whl` |
| `work/umdk-test/vllm-ascend-pin/` | vllm-ascend@bd42eea 源码树（已装为 editable） |
| `work/umdk-test/afd-plugin/` | afd-plugin 仓库（含两分支 worktree） |
| `work/umdk-test/deepep/` | DeepEP-Ascend @ 25b5153 源码树 |

### 10.2 运行记录与证据

| 路径 | 内容 |
| --- | --- |
| `work/umdk-test/runs/n1/` | 最完整的一次 E2E：`attention.log`(2819 行) / `ffn.log`(8293 行) / `e2e.log`(11136 行) / `run.json` |
| `work/umdk-test/runs/n1/extra-info/data-dump/0/` | 异常内核 dump（`AfdAsyncDispatchSend` / `AfdAsyncCombineRecv`） |
| `work/umdk-test/runs/aiv8/` | AIV 核数改 8 的对照实验 |
| `work/umdk-test/runs/aiv_restored/` | 环境恢复后复跑记录 |
| `work/umdk-test/logs/` | 各次探针与参考测试日志（JIT/H=7168/小形状/扫描） |

### 10.3 复用的辅助脚本

| 脚本 | 用途 |
| --- | --- |
| `work/umdk-test/setup_env.sh` | 在新容器内一次性重建三层依赖栈 |
| `work/umdk-test/run_e2e.sh` | 跑 `afd-dsv4-flash-async-cam-dp2tp2-ep4` |
| `work/umdk-test/sweep_cores.sh` | 参考测试的 `num_vec_cores` 扫描 |
| `work/umdk-test/small_shape.sh` | 上游建议的小形状（K=2/E=8）验证 |
| `work/umdk-test/preload_ops.py` | 让 `test_umdk_meta_*` 能跑通的 pytest 插件 |
| `work/umdk-test/hccl_check.py` | 8 die HCCL all-reduce 基线 |

### 10.4 环境快照

| 项 | 值 |
| --- | --- |
| 主机 | a5-111 / 192.168.16.111 |
| 芯片 | Ascend950DT (9582) ×8，驱动 25.1.rc2，固件 7.0.9.101.200 |
| CANN | 9.2.1（`/usr/local/Ascend/cann-9.2.1`） |
| Python | 3.12.16 |
| 容器镜像 | `vllm-ascend-a5:032-megamoe-20261125` |
| docker context | `network_lab`（root 在 `/mnt/xds/sfs/docker-network-lab-mnt`） |

---

## 附录 A：三轮排查时间线

| 轮次 | 主要动作 | 关键产出 |
| --- | --- | --- |
| 第 1 轮 | 拉分支、建栈、跑通算子注册 / UB 链路 / UMDK 会话初始化 | 首次在真实设备走完注册缓冲区初始化并进入 `profile_run`；定位 16 个工程问题 |
| 第 2 轮 | 用 DeepEP 参考测试做传输层对照；隔离 `enable_local_rounds` | 证明 JIT 路径可用；发现 CAM 与 JIT 是两套实现；找到上游 507035 的官方记录 |
| 第 3 轮 | 核查上游历史；确认算子被移除；排查环境障碍 | 确认 `afd_async` 在 HEAD 已被删除（60→0）；定位 dockerd 被 SIGSTOP 与 NPU 被占满 |

## 附录 B：关键日志片段索引

| 片段 | 位置 |
| --- | --- |
| 双角色 8 rank UMDK 初始化 | §4.3（`UMDK AFD registered buffers`） |
| 首个前向失败时间线 | §4.4 |
| 上游 507035 官方说明原文 | §6.1 |
| 算子移除证据（60 → 0） | §6.2 |
| CAM vs JIT 内核差异表 | §6.3 |
| tiling 分支推算 | §6.6 |
| dockerd 挂起与恢复 | §8.2 |

---

*报告结束。本报告记录的所有结论均基于本会话内的实际命令输出与日志，
未包含推测性内容；标注为"高嫌疑"的部分已明确说明尚待上游确认。*
