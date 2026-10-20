# A5 异步通信算子相对 A3 的新增特性与性能优化

A5 在 A3 的 routed-only 异步通信框架上，新增 Ascend 950 适配，并围绕路由统计、数据搬运、接收复用、向量同步和 UB 内存使用实现了多项优化。**最新整合分支 `GuangyuZhu04/a5-async-cam-full-features` 已将 CCU 连续段传输、SIMT 描述符和 epoch/credit 多槽控制接入实际调用链，同时保留此前的累计优化。**

默认配置仍使用优化后的 MTE 四算子和 compact-v2 协议。显式设置 `a5_full_features=true` 后，DS/DR/CR 在隔离槽位内复用优化内核，CS 采用设备描述符和 custom CCU；外层协议为 `AFDA5SLT` version 1。当前槽粒度是完整通信轮次，按任意 contribution 到达顺序提前消费和提前归还小块槽位仍未启用。

比较范围固定如下，便于区分不同分支已有的能力：

- **A3 原版**：`bf5bd280b7481358fc0df4a8ec5897b12be73802`，即 A5 迁移前的仓库版本。
- **A5 迁移基线**：`GuangyuZhu04/a5-async-cam-baseline` 的 `7d773dc`，仅包含 B0 硬件适配和资源检查。
- **A5 累计优化实现**：`GuangyuZhu04/a5-async-cam-payload-tiling` 的 `4d134dc955c58727c08511c6043ee01279eb6279`，包含第 2.1～2.10 节优化；此时 CCU、SIMT 描述符和多槽协议仍为独立实验。
- **A5 最新整合实现**：[`GuangyuZhu04/a5-async-cam-full-features`](https://github.com/GuangyuZhu04/afd-plugin/tree/GuangyuZhu04/a5-async-cam-full-features)，实现提交为 [`06bd14b7015a5935d2276de15099ded801f3391c`](https://github.com/GuangyuZhu04/afd-plugin/commit/06bd14b7015a5935d2276de15099ded801f3391c)，在 `4d134dc` 上完成第 2.11～2.14 节集成。

本文按截至 2026-10-20 的上述源码快照说明。**“优化”指已实现的开销削减机制，尚无 A5 实机延迟、带宽或吞吐提升数据。** 第 2.1～2.10 节源码链接保留 `4d134dc` 的历史位置，整合新增代码链接固定到 `06bd14b`；分阶段提交和启用配置见第 4 节，完整运行说明见[全特性集成指南](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/docs/npu/CAM_ASYNC_A5_FULL_FEATURES.md)。

## 1 新增特性和优化概览

DS、DR、CS、CR 分别表示 dispatch send、dispatch recv、combine send、combine recv。AIV 是执行这些内核的向量核，UB 是核内暂存区，MTE2/MTE3 分别承担这里的读入与写出搬运。

一轮数据流为 Attention DS → FFN DR → expert 计算 → FFN CS → Attention CR。下文 B 为单源 token 数，H 为 hidden 宽度，K 为 top-k，M 为 FFN/MoE rank 数，R 为每 rank 的 routed expert 数；q0/q1 表示动态量化关闭/开启。

| 类别 | 新增特性或优化 | 相对 A3 的变化和收益来源 | 实现状态 |
| --- | --- | --- | --- |
| 1 | A5 窗口适配与资源检查 | 适配 950 的 HCCL 窗口结构，增加构建注册、UB 峰值与容量检查，修正 64 位对齐计算 | B0 基线 |
| 2 | CS 和 DR 的 MTE 双缓冲 | 两个 UB 槽交替读写，减少逐 tile 或逐行等待 Scalar 的串行依赖 | A5 常规路径，按资源选择 |
| 3 | DS 的 SIMT 计数和前缀扫描 | 将路由统计并行化，保留稳定 token 布局及原 AIV 前缀格式 | A5 常规路径，按 batch 和 UB 选择 |
| 4 | CR 输入预取与同步精简 | 重叠下一块 MTE2 读取和当前块向量运算，去掉无 Scalar 消费者的两处显式等待 | A5 常规路径，UB 不足回退 |
| 5 | DR 两行缓存 | 重复 assignment 命中同一 wire row 时复用已有 UB 数据，减少 MTE2 读取 | A5 常规路径，限局部缓存 |
| 6 | 小 batch 使用较少 source AIV | DS 和 CR 同时采用 `min(platform_aiv, 48, B)`，减少空 token 分区 | A5 常规路径 |
| 7 | DS 稀疏 rank 管理 | 位图加访问列表替代每 token 清零、扫描全部 MoE rank | A5 常规路径 |
| 8 | DS 两槽 token 流水 | 跨 token 安排输入预取、量化和发送，以整 token 生命周期替代逐目标等待 | A5 常规路径，资源不足回退 |
| 9 | DR 连续行批量搬运 | 合并同源、同 expert 的连续 wire rows，减少小 DMA 次数 | A5 常规路径，有连续段才选择 |
| 10 | CR 按 hidden tile 分配 FP32 scratch | 三份 full-H scratch 改为 tile 大小，并预存每 token 的 K 个地址和权重 | A5 常规路径，优于旧 tile 规划才选择 |
| 11 | CS 的 CCU 连续段传输 | 设备描述符提供动态 offset 和 length，由每个 peer 的有界 CCU 任务承担传输，目标是减少 AIV 搬运占用 | 已接入全特性路径，显式启用 |
| 12 | SIMT 稳定描述符生产者 | 并行生成固定槽描述符，以有界 tile 批量写回；UB 不足选择 scalar prepare | 已接入 CS keys 120/121，按资源选择 |
| 13 | epoch 和 credit 多槽控制 | 2～8 个完整轮次槽，隔离 frame、匹配 epoch 回收并公平选择完整 TP 组 | 已接入四算子；细粒度乱序消费仍为参考模型 |
| 14 | 运行时、资源所有权和构建集成 | 增加原生会话、三阶段初始化、ticket 传递、同 stream 完成发布及 HCOMM ABI 隔离 | 已接入 connector、PyTorch binding 和 wheel |

A3 已经具备异步 dispatch/combine、routed-only 传输、同 token 同 rank 的 payload 去重、动态 INT8 量化和按 top-k 顺序的 FP32 加权累加。这些是 A5 保留的原有能力。

“全特性”表示整合分支包含累计优化及新增运行路径，不表示所有机制在每次调用中同时执行：默认 CS 仍走 MTE 双缓冲，全特性 CS 走 CCU；SIMT、缓存、连续行及 tile scratch 仍按各自门禁选择。第 2.1～2.10 节描述的默认路径继续保留，DS/DR/CR 的适用优化也由全特性路径复用。

## 2 按特性对应代码改动

下面代码位置中，四算子目录及 `utils/` 均相对于 `csrc/npu/ascend_kernels/`。原实验模块位于 `csrc/npu/experimental/`；整合版本将 P2/P3/P4 共用算法提升到 `utils/op_kernel/cam_async_*`，原实验头文件转发到共享实现。

### 2.1 A5 硬件适配和资源边界检查

**A3 原行为**：内核直接解释 `HcclOpResParam`，本地读取 `localWindowsIn`，远端通过 `remoteRes[rank].nextDevicePtr` 取得窗口；四个 async 算子只注册 `ascend910_93`。

**A5 改动**：把窗口读取集中到 `AsyncDataWindow`。A5 使用 SDK 提供的 `AscendC::HcclCombineOpParam.windowsIn[rank]`，编译期确认窗口数组至少有 64 项；A3 保留原结构。业务 metadata 和 ready flags 继续位于 data window，未迁移到 `windowsOut`。

| 代码位置 | 具体更改 |
| --- | --- |
| [utils/op_kernel/cam_async_window.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_window.h#L26) | 新增 `AsyncDataWindow::Init/GetDataWindow`，集中隔离 A3/A5 窗口 ABI。 |
| 四个 `afd_async_*/op_kernel/afd_async_*.h` | `Init` 和 `GetPeerAddrByRankId` 接入窗口适配；`MathCeil` 改为 `uint64_t`，避免超过 4 GiB 的容量在对齐计算时被截断；向量 barrier 使用 SDK 的 `PipeBarrier<PIPE_V>()`。 |
| [utils/op_host/cam_async_host_tiling.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_host/cam_async_host_tiling.h#L12) | 新增四算子的 UB 峰值计算、`GetActiveAivCount` 和窗口尾部 cursor 预留常量；统一 active AIV 上限为 48。 |
| [DS op_host](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_host/afd_async_dispatch_send.cpp#L73)、[DR op_host](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_host/afd_async_dispatch_recv.cpp#L74)、[CS op_host](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_send/op_host/afd_async_combine_send.cpp#L68)、[CR op_host](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_recv/op_host/afd_async_combine_recv.cpp#L67) | `SetHcommCfg` 为 A5 选择 `HcclAccelerator::AIV` 对应的命名常量 `MTE_HCCL_ACCELERATOR`；tiling 增加 A5 world、形状、UB 和窗口检查，使 launch blockDim 与 tiling aivNum 一致；增加 `AddConfig("ascend950")`。 |
| [operator_registry.json](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/operator_registry.json#L14)、[pybind/CMakeLists.txt](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/pybind/CMakeLists.txt#L54) | 为 950 同时启用四算子的 CANN 包和 PyTorch binding 编译，避免只编出算子包或只注册 Python 名称。 |

A5 域要求 Attention 和 FFN rank 为正、world 不超过 64，并限制 hidden 对齐和单源 UINT16 可表示范围。DR 的地址 slice 按实际 active AIV 分配预算，FFN 窗口额外保留尾部 32B cursor。此阶段主要建立可构建、可检查的迁移基线；ACLNN 仍沿用 MTE server。

验证入口：[test_cam_async_host_tiling.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_host_tiling.py#L1)、[test_cam_async_build.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_build.py#L1)。

### 2.2 CS 和 DR 的 MTE 双缓冲

**A3 原行为**：CS 每个连续 tile、DR 每个 gather row 都要完成读入、写出及 Scalar 等待后，再复用同一 UB buffer。

**A5 改动**：CS 把可复用 UB 按完整行划成两个 tile；DR 保留原输入行和完整地址 slice，只从剩余 UB 切出第二行。两个槽分别拥有 ready/free event，使 tile n 的 MTE3 写出与下一槽的 MTE2 读入具备重叠条件。

- [afd_async_combine_send/op_kernel/afd_async_combine_send.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_send/op_kernel/afd_async_combine_send.h#L186)：`CombineSend` 增加双槽规划、交替 DMA 和尾部 `Drain`，仍按 expert/TP 段与 AIV 区间限制 tile。
- [afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h#L660)：`DispatchRecvRouted` 选择流水路径，`DispatchRecvRowsPipelined` 执行逐行读写；INT8 payload 与其 FP32 scale 使用同一槽生命周期。
- [utils/op_kernel/cam_async_mte_pipeline.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_mte_pipeline.h#L13)：新增 `PlanCombineSendBuffers`、`CanUseReceiveSecondSlot`、`PipelineEvents` 和 `TwoSlotPipeline`。

event 使用 `AllocEventID/ReleaseEventID` 租用，避免复用仍在途的编号。所有槽写出完成后，通过 MTE3→MTE2→Scalar 依赖收尾，再进入原有通知和窗口释放逻辑。A3、单行及 UB 不足时保留串行路径；最低 UB 要求没有因双缓冲而提高。

验证入口：[test_cam_async_mte_pipeline.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_mte_pipeline.py#L1)。

### 2.3 DS 路由统计和前缀扫描使用 SIMT

**A3 原行为**：`PreProcess` 由 Scalar 遍历 top-k，计算每个 rank 的唯一 payload 数、每个 expert 的 assignment 数，再生成各 AIV 的累计前缀。

**A5 改动**：128 个 SIMT 线程按 token 分工，使用 UB 整数原子计数 assignment；每个 token 用线程局部 rank 位图统计唯一 payload 目的地。前缀扫描按完整统计列分工。原 payload 和地址表仍按 token 顺序产生，原子计数不参与分配输出位置。

- [DS kernel 的 PreProcess](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h#L683)：接入 `count_routing_tokens` 和 SIMD/MTE 与 VF 之间的事件交接，保留两次 `SyncAll` 及原统计矩阵。
- [afd_async_dispatch_send_simt.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send_simt.h#L45)：新增 `count_routing_tokens_vf` 和 `prefix_routing_columns_vf`。
- [cam_async_routing_columns.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_routing_columns.h#L24)：共享 `CountTokensForLane`、`PrefixColumnsForLane`，保留 inclusive per-AIV prefix，使 CR 仍能定位本核之前的 occurrence。
- [cam_async_simt_routing_budget.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_simt_routing_budget.h#L14)、[DS op_host 的选路](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_host/afd_async_dispatch_send.cpp#L221)、[DS kernel 入口](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.cpp#L1)：新增资源门禁和 SIMT keys `110/111`，保留 scalar keys `100/101`。

启用需要每 AIV 至少 32 token、`1 ≤ topk ≤ 16384`，以及原峰值 UB 之外可保留 136 KiB 的 SIMT 预算。prefix staging 不足时仍可采用 SIMT 计数加 scalar prefix。32 token 是结构性门槛，136 KiB 是保守资源预留，均不代表实测最佳配置。

验证入口：[test_cam_async_simt_routing.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_simt_routing.py#L1)。

### 2.4 CR 输入预取和向量到标量同步精简

**A3 原行为**：每个 top-k contribution 读一行，转换、乘权、累加后推进；初始化和累加后存在显式 `V_S` 等待。

**A5 改动**：先保留三份 full-H FP32 scratch，只从剩余 UB 增加第二个输入槽。按 token→top-k→hidden tile 遍历，预取下一块后计算当前块。后续提交去掉 `Duplicate` 和 `Add` 后两处没有 Scalar 消费者的显式 `V_S` 等待，并按值保存当前 tile 的 weight。

- [CR kernel 的 CombineRecvPipelined](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h#L537)：新增有序预取循环及 `currentScale` 捕获；保留 `Cast→PipeBarrier→Muls→PipeBarrier→Add`。
- [CR kernel 的 LoadCombineTile](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h#L502)：只在新 contribution 时推进 expert occurrence，同一行的后续 hidden tile 复用地址和 weight。
- [utils/op_kernel/cam_async_combine_prefetch.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_combine_prefetch.h#L28)：新增 `PlanCombinePrefetch`、`CombineTileCursor`、`PipelineEvents`；用 `MTE2_V/V_MTE2` 交接输入槽。

第二槽不足 2048B 时回退；完整行只有一次 contribution 时也不强制流水。最后一个 K 的输出复用输入槽，因此保留 `V_MTE3/MTE3_V` 保护；尾部 drain 后才进入原 `SyncAll` 和 flags reset。减少的是无必要的 Scalar 等待，向量依赖和输出保护仍然存在。

验证入口：[test_cam_async_receive_optimization.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_receive_optimization.py#L1)，其中包含异步算术、参数捕获和同步遗漏负例。

### 2.5 DR 两路缓存复用同一 wire row

**A3 原行为**：同一 token 命中某 rank 的多个 expert 时，发送端虽只传一份 payload，DR 仍按每个 assignment 重新读入该 wire row。

**A5 改动**：在既有两行 UB 槽中增加两路 LRU 缓存，以实际 source 地址和 wire index 作为 key。命中时省去 MTE2 读入，仍完整写出当前 assignment 的 payload 和 scale。

- [utils/op_kernel/cam_async_receive_cache.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_receive_cache.h#L19)：新增 `CanUseCache`、`RowKey`、`TwoRowCache::Select/Remember`。
- [DR kernel 的 DispatchRecvRowsPipelined](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h#L660)：在命中判断后仍等待该槽上次 MTE3 消费完成，再执行 ready handoff 和本次输出。

缓存只在一次 TP/source/chunk gather 内有效。单 selected expert 或 AIV 区间完全处于初始 expert 内时，采用原 P0 交替槽，省去 cache 查询。它减少局部重复读取，不改变网络去重规则；大 batch 的 expert-major 遍历可能缺少缓存命中。

验证入口：[test_cam_async_receive_optimization.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_receive_optimization.py#L1)。

### 2.6 小 batch 的 DS 和 CR 使用相同的较少 AIV

**A3 原行为**：source 侧按平台核数划分 token，小 batch 会产生空分区。A5 B0 先统一为 `min(platform_aiv, 48)`，接收优化阶段进一步考虑 B。

**A5 改动**：同一 source 的 DS 和 CR 一起使用 `min(platform_aiv, 48, B)`，例如 B=7 和 B=5 的 source 分别使用 7 核和 5 核。FFN 侧 DR/CS 仍使用 `min(platform_aiv, 48)`。

- [GetSourceActiveAivCount](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_host/cam_async_host_tiling.h#L38)：新增 source 核数策略。
- [DS op_host](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_host/afd_async_dispatch_send.cpp#L215)、[CR op_host](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_recv/op_host/afd_async_combine_recv.cpp#L200)：共同更新 launch 和 tiling，保持 token 分区及 DS→CR 前缀解释一致。
- [tests/npu/async_cam_common.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/npu/async_cam_common.py#L1)：harness 报告各 source 的预期 DS/CR 核数及 FFN 核数；`--aiv-count` 仍是资源声明，不是 native launch 覆盖参数。

窗口仍预留 48 行统计，payload offset 不变；UB 下限仍按平台 cap 校验，不因少用核而扩大原接受范围。收益目标是减少空分区，奇数核数下的同步仍需设备验证。

验证入口：[test_cam_async_host_tiling.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_host_tiling.py#L1) 和 [test_async_cam_harness.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/compat/npu/test_async_cam_harness.py#L1)。

### 2.7 DS 只管理当前 token 访问过的 rank

**A3 原行为**：每 token 清零 M 个 flag，处理 top-k 后再扫描 M 个 rank，推进已发送 rank 的 wire offset。

**A5 改动**：用一个 64 位位图记录是否访问，用原 `uint8[M]` flag 区保存访问列表；token 结束只推进 U 个实际访问 rank，`U ≤ min(K, M)`。rank 管理开销由每 token 的 O(M) 清零和扫描变为 O(1) 重置及 O(U) 更新，top-k 遍历仍需 O(K)。

- [utils/op_kernel/cam_async_sparse_ranks.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_sparse_ranks.h#L35)：新增 `TokenRankTracker::BeginToken/RecordRank/AdvanceOffsets` 及 A5 域检查。
- [DS kernel 的 DispatchSend](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h#L498)：首次命中 rank 才发 payload，每个 assignment 都写地址；同 token 全部 assignment 处理完后才推进一次该 rank 的 wire offset。

未增加 UB buffer。合法 A5 域中 Attention 至少占一个 rank，因此 MoE rank 最多 63。此改动优化去重管理的 Scalar 工作量，发送字节数与 A3 相同。

验证入口：[test_cam_async_sparse_ranks.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_sparse_ranks.py#L1)。

### 2.8 DS 跨 token 组织输入和发送流水

**A3 原行为**：一个 token 读入并按需量化后，向每个目标 rank 发送，每次 payload 发送都显式等待 `MTE3_S`。

**A5 改动**：从 spare UB 增加第二个输入行槽，每槽让输入和 wire 输出原位重叠，不复制共享 FP32 量化 scratch。当前 token 输入就绪后先安排下一 token 读入；当前槽向全部目标发送完毕后，统一交还 MTE3→MTE2 credit。

- [utils/op_kernel/cam_async_dispatch_pipeline.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_dispatch_pipeline.h#L26)：新增 `plan_dispatch_buffers` 和 `PipelineEvents`，管理两槽输入、量化结果、scale 与整 token 释放。
- [DS kernel 的 DispatchSend](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h#L504)：增加 next-token 预取，流水路径取消逐 peer 的 Scalar 等待，尾部 drain 后才 flush 地址 metadata 和发布 flags。
- [DS kernel 的 QuantProcess](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h#L967)：增加 A5 私有的 pipeline/slot 参数，仅调整尾部 ready 交接；量化路径为每槽租用独立 `V_MTE3` 和 `S_MTE3` event。

本 AIV 至少有两个 token、第二行可容纳且 wire extent 不超过输入行时才启用。地址表 staging 保留自身的 `S_MTE3/MTE3_S` 保护；payload credit 不能代替 metadata 完成边界。收益目标是跨 token 重叠读取、量化和发送，原量化中的 Scalar 读取仍可能限制重叠程度。

验证入口：[test_cam_async_dispatch_pipeline.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_dispatch_pipeline.py#L1)。

### 2.9 DR 将连续 wire rows 合成批量 DMA

**A3 原行为**：每个 assignment 独立读取一行；即使相邻 wire index 连续，也不会合并搬运。

**A5 改动**：保持 expert→TP→token 的输出顺序，在同一 source、同一 expert、同一 AIV 区间内识别连续 wire index。两个额外的连续 UB tile 交替搬运 run；非连续的 singleton 仍按原顺序处理。

- [utils/op_kernel/cam_async_receive_runs.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_receive_runs.h#L30)：新增 `plan_receive_runs`、`ReceiveRunCursor`、`has_contiguous_run`，限定 tile 为至少 2 行、最多 128 行且不超过 64 KiB；UINT16 索引末尾不回卷合并。
- [DR kernel 的 dispatch_recv_runs_pipelined](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h#L595)：一个 run 用一次 MTE2 读入；q0 批量连续写出，q1 按行去掉 32B wire padding，并逐行复制原 4B scale。
- [plan_payload_copy](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_receive_runs.h#L56)：明确 `DataCopyPad` 的 UB source stride 使用 32B block 单位，量化行的 source stride 为 1。

选路优先尝试连续 run，其次是两行缓存/P0 流水，最后回退串行。没有连续段或 UB 不足时，不使用 run 路径；64 KiB 和 128 行是结构上限。优化减少 DMA 调用次数，不减少需要交付的 assignment 数据量。

验证入口：[test_cam_async_receive_runs.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_receive_runs.py#L1)。

### 2.10 CR 用 tile 大小的 FP32 scratch 替代 full-H scratch

**A3 原行为**：持有三份 `float[H]`，分别用于输入转换、乘积和累加；大 H 会挤压输入预取可用的 UB。前述 CR 预取阶段仍保留这些 full-H buffer。

**A5 改动**：每 token 先固定 K 个 source 地址和 FP32 weights，然后按 hidden tile→K 遍历，复用三份 tile 大小 scratch 和两个输入槽。对每个 hidden 元素，贡献顺序仍为 top-k 下标 0…K−1。

- [utils/op_kernel/cam_async_combine_tile_plan.h](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_combine_tile_plan.h#L60)：新增 `plan_tile_scratch` 和溢出检查，预算 K 个 UINT64 地址及 FP32 weights；`capture_token_sources` 使 occurrence 每贡献只推进一次。
- [CR kernel 的 Init](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h#L260)：在原 metadata 之后选择 tile scratch 布局；不满足条件时保留 full-H 布局和原预取规划。
- [CR kernel 的 combine_recv_tiled](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h#L644)：采用 `HiddenKCursor`、`load_tiled_input` 和 `retire_token_descriptor_use`；token 末等待旧描述符消费结束，之后才允许下一 token 覆写。

三份 FP32 scratch 从 `12H` 字节降为 `12h` 字节，h 为 tile 宽度；另有两个输入槽和约 `12K` 字节加对齐的描述符成本。只有 `h < H`、输入 tile 至少 2048B，且能扩大旧预取 tile 或旧预取不可用时才选择。原 full-H 最低资源校验仍须通过，不由此扩大支持的 shape。

计算仍分别执行 Cast、Muls、Add，最后 `CAST_RINT`，未引入 FMA 或改变 K 的累加顺序。收益目标是改善受 UB 限制时的预取粒度，代价包括描述符读写、分块 DMA 和 token 退休等待。

验证入口：[test_cam_async_combine_tile_plan.py](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/tests/unit/package/test_cam_async_combine_tile_plan.py#L1)。

### 2.11 CCU 连续段传输接入实际 CS

**A3 原行为**：CS 的载荷经过 GM→UB→peer window，由 AIV 调度 MTE 搬运。

**早期阶段**：`d859336` 增加独立 CCU planner、AIV 描述符生产者和 host 提交封装；截至 `4d134dc`，该实验未接入四算子。

**最新整合改动**：`06bd14b` 将链路接为“FFN 输出复制到稳定注册的 staging → 设备生成连续段描述符 → custom CCU 搬运 → 完成标记 → ready 发布”。32B 描述符保存 source offset、destination offset、byte count 和 destination rank；动态 counts、目标 TP 组和地址规划留在设备上。

| 代码位置 | `06bd14b` 的具体更改 |
| --- | --- |
| [utils/op_kernel/cam_async_combine_segments.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_combine_segments.h) | 将 `CombineSegmentCursor::Init/Next` 提升为生产与实验共用的 checked planner，保留 metadata、TP 组和容量检查。 |
| [utils/op_kernel/cam_async_a5_prepare.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_a5_prepare.h) | 新增 `PrepareCombine`，按全部 Attention rank 建立固定槽表，非参与 peer 清为零长度；目标 offset 包含当前 frame 偏移。 |
| [csrc/npu/runtime/cam_async_a5_transport.cpp](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/runtime/cam_async_a5_transport.cpp) | `afd_a5_transport_create/launch` 接入真实 HCOMM 资源和通道；`BatchKernel::Algorithm` 在每个 peer 的一个有界 CCU 任务内循环处理 expert 段。 |
| [csrc/npu/pybind/torch_binding_cam_async_a5.cpp](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/pybind/torch_binding_cam_async_a5.cpp) | `AsyncCamA5Context::combine_send` 管理 staging，按 PrepareCombine→CCU→PublishCombine 顺序提交，所有阶段进入 torch-npu 任务队列并使用同一 runtime stream。 |
| [CS kernel 入口](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/afd_async_combine_send/op_kernel/afd_async_combine_send.cpp)、[cam_async_a5_control.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_a5_control.h) | 分离描述符准备与完成发布两次 launch；`PublishCombine` 同时检查规划成功状态和 CCU retirement marker，仅在最后一个 expert chunk 发布最终 ready。 |

规划状态和传输完成标记使用不同的 word，标记排在所有 CCU 任务之后，避免部分 host 提交失败时把上一轮成功当作本轮完成。相对 A3，收益目标是减少 AIV 数据搬运占用；相对早期逐 expert 的 CCU 提交方案，整合版本收敛为逐 peer 提交。FFN 输出到 staging 的额外复制也必须计入成本。该路径已完成编译链接，实际 CCU 执行、跨引擎顺序和远端可见性仍待真机验证。

验证入口：[test_cam_async_ccu.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_ccu.py)、[test_cam_async_a5_full.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_a5_full.py)。

### 2.12 SIMT 稳定生成 CCU 描述符

**早期阶段**：`fbf3a95` 实现独立 SIMT 描述符生产者。lane 先处理完整 TP 行，把 UB counts 原位转换成 inclusive prefixes，再独占 `tp*R+expert` 固定描述符槽；每 256 槽、8 KiB tile 连续写回，避免 atomic ticket 改变布局。

**最新整合改动**：生产 CS 的 prepare 阶段直接调用上述共享算法，并编译独立 scalar/SIMT 入口。全局描述符表先清零，再写当前 TP 组，避免残留的其他 DP 组描述符被发送。

- [utils/op_kernel/cam_async_stable_descriptors.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_stable_descriptors.h)、[cam_async_descriptor_simt.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_descriptor_simt.h)：提升 `PrefixTpRowsForLane`、`DescribeSlotsForLane` 和两个 VF 为共享实现，保留稳定 expert→TP 源顺序。
- [cam_async_a5_prepare.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_a5_prepare.h)：`PrepareCombine` 先运行完整 checked planner，再选择 scalar cursor 或 SIMT prefix/descriptor VF，完成 MTE3 写回后发布规划状态。
- [CS op_host](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/afd_async_combine_send/op_host/afd_async_combine_send.cpp)、[CS tiling key 定义](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/afd_async_combine_send/op_kernel/afd_async_combine_send_tiling.h)：新增 **120（scalar prepare）/121（SIMT prepare）**；保留默认 MTE 的 100/101。
- [cam_async_descriptor_budget.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_descriptor_budget.h)：完整 SIMT UB 预算为 **213024B**；不足时使用 **73760B** scalar arena，连 scalar 预算也不足则拒绝该 prepare。

SIMT 描述符与第 2.3 节的 DS SIMT 路由统计是两处独立选路，不要求同时启用。它减少串行描述符生成和零散写回，每个 descriptor 恢复 source prefix 仍有 O(TP) 读取，收益尚未实测。

验证入口：[test_cam_async_pipeline.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_pipeline.py)、[test_cam_async_a5_full.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_a5_full.py)；实际 A5 编译产物已检出 CS SIMT entry。

### 2.13 epoch 和 credit 多槽控制接入四算子

**A3 原行为**：compact-v2 通过原 ready/chunk flags 管理串行 round 的窗口复用，没有 epoch、credit 或多批在途的 wire slot。Python 层 stage 不能替代这些协议字段。

**早期阶段**：`fbf3a95` 提供独立 `AFDSLOT4` version 1 参考协议、`BoundedSlotProtocol` 状态机和 `OrderedCombineGate` 有序归约模型，未直接驱动四算子通信。

**最新整合改动**：使用独立的 **`AFDA5SLT` version 1** 外层协议，将原 compact routed 布局放入 2～8 个隔离 frame，默认 2 槽。控制记录携带 domain、slot、epoch、request、状态和边界字段；同一 source 的 epoch 单调递增，耗尽后拒绝继续使用。

| 代码位置 | `06bd14b` 的具体更改 |
| --- | --- |
| [cam_async_a5_contract.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_a5_contract.h) | 新增调用 tag、`RuntimeArgs`、`PlanLayout/FrameOffset` 和记录匹配规则，明确 full profile 的协议标识及每槽 payload 容量。 |
| [cam_async_a5_control.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_a5_control.h) | `BeforeDispatch` 等待原 epoch 的 credit；`BeforeReceive` 公平轮询完整 TP 组；`PublishCombine` 发布最终 ready；`BeforeCombine/AfterCombine` 等待贡献并归还 credit。增加 abort 轮询及 `BeforeCall/AfterCall` 控制边界。 |
| 四个 `afd_async_*/op_host`、`op_kernel` 及 [cam_async_window.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_window.h) | host 容量检查由整个窗口改为单 frame payload；kernel 在数据阶段前后调用控制逻辑，窗口寻址加 frame offset，保留必要的多 AIV 同步。 |
| [torch_binding_cam_async_a5.cpp](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/pybind/torch_binding_cam_async_a5.cpp) | 维护 host 槽位预留和轮换，将设备 ticket 从 DS 传至 CR、从 DR 传至 CS；host 释放预留后仍由同 stream 顺序和设备 credit 约束实际复用。 |
| [cam_async_ring_layout.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_ring_layout.h)、[cam_async_slot_protocol.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_slot_protocol.h) | 提升共享 ring、128B 控制记录及校验实现；`BoundedSlotProtocol/OrderedCombineGate` 继续作为参考模型保留。 |

**槽位按完整轮次持有**：DR 仍可按完整 expert 分 chunk，非末 chunk 继续持有 lease；CS 在最后 chunk 发布最终 ready，CR 按原 K 顺序完成归约后返回 credit。`OrderedCombineGate` 的任意 contribution 乱序捕获、提前消费和提前归还小块槽位未接入本调用链。多槽机制已实现，已有验收入口按轮次依次轮换槽位，并发在途轮次及压力性能仍需另行真机验收。

记录发布遵循“清 commit→写 body→写 epoch commit”，接收端通过 MTE2 重读 commit/body/commit。当前 CPU 控制测试覆盖这些生产逻辑，但 8B commit 的设备原子性、远端可见性和跨引擎顺序尚无实机证明。`AFDA5SLT`、实验 `AFDSLOT4` 和默认 compact-v2 使用不同的协议合同，不能混用通信域。

验证入口：[test_cam_async_slot_protocol.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_slot_protocol.py)、[test_cam_async_slot_wire.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_slot_wire.py)、[test_cam_async_a5_control.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_a5_control.py)。

### 2.14 运行时、资源所有权和构建集成

**此前的边界**：截至 `4d134dc`，累计优化未修改 `afd_plugin/` 下的 Python runtime，独立实验也没有完整会话和 connector 调用链。`06bd14b` 补齐以下代码层级：

| 代码位置 | 具体更改 |
| --- | --- |
| [afd_plugin/connectors/npu/async_cam.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/afd_plugin/connectors/npu/async_cam.py) | 解析 `a5_full_features/a5_slot_count`，使用专用 `afd_async_cam_a5_full` 通信组；在 transfer state 中传递 `transport_ticket`，将四个操作路由到原生会话，安排关闭顺序。 |
| [afd_plugin/compat/npu/cam_async_a5.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/afd_plugin/compat/npu/cam_async_a5.py) | 新增可在 CPU 环境导入的类型化接口、`create_async_cam_a5_context` 和 `initialize_async_cam_a5_context`，使用既有 process-group Store 完成三阶段初始化。 |
| [torch_binding_cam_async_a5.cpp](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/pybind/torch_binding_cam_async_a5.cpp) | 注册 `torch.classes.afd_ascend.AsyncCamA5Context`；对象字段持有 MC2 窗口、内存注册、staging、descriptor、status、ticket 和 NPU event，初始化后绑定首次数据调用的 stream。 |
| [torch_binding_cam_async.cpp](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/pybind/torch_binding_cam_async.cpp)、[torch_binding_cam_async.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/pybind/torch_binding_cam_async.h) | 原四算子 binding 增加 full-call tag 和 session-owned `RuntimeArgs` 大小检查，将 tag 传给现有 ACLNN 入口；保留默认调用及算子 schema。 |
| [cam_async_a5_transport.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/runtime/cam_async_a5_transport.h)、[cam_async_a5_transport.cpp](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/runtime/cam_async_a5_transport.cpp) | 通过仅含 C/POD 参数的边界隔离 HCOMM ABI0 和 torch 2.8 ABI1；查询拓扑并选择 layer 0 的 UBC_TP/UBC_CTP 通道。 |
| [pybind/CMakeLists.txt](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/pybind/CMakeLists.txt)、[setup.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/setup.py)、[pyproject.toml](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/pyproject.toml) | 950 构建新增 `libafd_cam_async_a5_transport.so`，与 PyTorch 扩展分别编译并链接，补齐 inplace 安装和 wheel 依赖库打包。A3 不注册新会话类。 |
| [cam_async_checked_arithmetic.h](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_checked_arithmetic.h) | 新增 `CheckedMultiply64`，使用 32 位分段实现 UINT64 溢出检查，避免 CANN 3101 实际 kernel 链接依赖不可用的 `__multi3`。 |

初始化按以下边界推进：

1. `prepare()` 查询 MC2 窗口和实际分配大小、注册内存并取得 token，各 rank 经 Store 交换资源、shape/dtype 和 rank 身份。
2. `configure()` 校验全域配置、清零本地 frame 并分配缓冲；全部 rank 确认完成后才注册通道。
3. `initialize()` 注册 CCU 通道与程序并完成初始化同步；全部 rank 确认后才允许数据调用。

Store 只用于初始化，不增加每 batch/chunk 的 Store 请求或 CCU counts D2H。原 FFN worker 为选择模型层等用途已有的 metadata CPU 读取继续保留。

正常关闭顺序为 `quiesce()` → 销毁 HCCL process group → `release_after_group_destroyed()`。`abort()` 使用独立 DMA stream 通知设备轮询者退出并禁止继续提交；它不签发 credit、不重置 epoch，也不取消或证明 CCU 任务已退休。设备错误后需整域收尾或进程退出，不能原地重发和复用槽。

验证入口：[test_cam_async_a5_context.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/compat/npu/test_cam_async_a5_context.py)、[test_async_cam_a5_wiring.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/connectors/test_async_cam_a5_wiring.py)、[test_cam_async_build.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/unit/package/test_cam_async_build.py)。

## 3 保持的数值行为与两条路径的接口差异

| 项目 | 默认累计 MTE 路径 | 全特性路径（`a5_full_features=true`） |
| --- | --- | --- |
| 对外接口 | 保留四个 `torch.ops.afd_ascend.afd_async_*` 名称、schema 和 ACLNN 签名。 | 新增 `AsyncCamA5Context` 会话接口和 ticket；复用原 ACLNN 入口，通过 full-call tag 和会话参数选择新控制逻辑。 |
| 通信协议 | `cam_async_routed_only_compact_v2`。 | `AFDA5SLT` version 1 外层 frame，每槽内部承载 compact routed 布局；使用独立通信域。 |
| 路由与输出 | 同 token 同 MoE rank 一份 payload，每个 assignment 保留地址，DR 按 expert→source TP→token 输出。 | 保留相同去重和稳定输出顺序；shared experts 同样留在 Attention。 |
| 动态量化 | 逐 token FP32 amax，按 `127/amax` 量化，存储 FP32 倒数 scale，保留 round-to-even。 | 复用相同量化公式、scale 表示和舍入路径。 |
| combine 算术 | FP16/BF16 转 FP32，各 K 分别乘权、顺序相加，最后一次舍入回原 dtype。 | 保持每个元素的逻辑 K 顺序及分步乘加；CCU 和分槽不改变归约公式。 |
| 完成与复用 | 保留原 ready/chunk flags、空 MoE 通知和窗口保活要求；UB 双缓冲属于内核内部机制。 | 增加完整轮次 lease、epoch/credit、CCU 完成门禁及 event 保活；最后 chunk ready 后由 CR 归还 credit。 |
| 运行时范围 | 保留原 CAM async 约束。 | 已修改 Python connector 和 native binding；当前要求 eager、单会话单提交 stream，未新增 ACL graph、native DBO 或 A3/A5 混合通信保证。 |

历史数值路径可比较原 [A3 DS](https://github.com/GuangyuZhu04/afd-plugin/blob/bf5bd280b7481358fc0df4a8ec5897b12be73802/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h#L350) 与 [累计 A5 DS](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h#L450)，以及 [A3 CR](https://github.com/GuangyuZhu04/afd-plugin/blob/bf5bd280b7481358fc0df4a8ec5897b12be73802/csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h#L290) 与 [累计 A5 CR](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h#L395)。最新整合已修改 `torch_binding_cam_async.cpp` 并新增会话 binding；四个 `op_api/` 包装实现仍未修改。

## 4 分支和提交索引

这些阶段形成连续的父子提交链。表中分支均带 `GuangyuZhu04/` 前缀；后续分支包含前面阶段。CCU、SIMT 描述符和多槽控制直到 `full-features` 阶段才接入运行时。表中记录实现提交，后续仅文档更新不改变这些源码验证基准。

| 分支后缀 | 提交 | 本阶段引入的内容 |
| --- | --- | --- |
| `a5-async-cam-baseline` | [7d773dc](https://github.com/GuangyuZhu04/afd-plugin/commit/7d773dcb5f1e73987d80b61f1191da2069057a34) | B0 硬件适配、资源检查和构建注册。 |
| `a5-async-cam-mte-pipeline` | [1269434](https://github.com/GuangyuZhu04/afd-plugin/commit/12694341a2fa3d709d63c37c26020033ab76fc10) | CS/DR 的 MTE 双缓冲。 |
| `a5-async-cam-simt-routing` | [45f86a1](https://github.com/GuangyuZhu04/afd-plugin/commit/45f86a1d9b1f647809fd11d8ca2fab2d40798d71) | DS SIMT histogram 和 prefix。 |
| `a5-async-cam-ccu-combine` | [d859336](https://github.com/GuangyuZhu04/afd-plugin/commit/d85933657c776ea7563363d8a57a2e71a670557e) | CCU 连续段实验模块。 |
| `a5-async-cam-simt-ccu-slots` | [fbf3a95](https://github.com/GuangyuZhu04/afd-plugin/commit/fbf3a956061b7f17e30de71e3f0bac01901e7ce3) | SIMT 描述符和多槽参考协议。 |
| `a5-async-cam-recv-optimization` | [52504ac](https://github.com/GuangyuZhu04/afd-plugin/commit/52504ac3ce1eace526bd1f5148f5045fe7efaa25) | CR 预取、DR 两行缓存、小 batch 核数选择。 |
| `a5-async-cam-sync-sparse` | [730ec5d](https://github.com/GuangyuZhu04/afd-plugin/commit/730ec5d7a06e0ba8ece5aa7c49229ed5434c4f1f) → [3a265b4](https://github.com/GuangyuZhu04/afd-plugin/commit/3a265b45aa0b21b9d33038ec30d5a220668456cb) | 先精简 CR Scalar 等待，再优化 DS rank 管理。 |
| `a5-async-cam-payload-tiling` | [5574c9e](https://github.com/GuangyuZhu04/afd-plugin/commit/5574c9e0fd76a1e952e44359740b2dd5ce9982c8) → [fbb268b](https://github.com/GuangyuZhu04/afd-plugin/commit/fbb268be5362827bc69f984685f595eae3f5b53e) → [4d134dc](https://github.com/GuangyuZhu04/afd-plugin/commit/4d134dc955c58727c08511c6043ee01279eb6279) | 依次加入 DS token 流水、DR 连续行、CR tile scratch。 |
| **`a5-async-cam-full-features`** | [06bd14b](https://github.com/GuangyuZhu04/afd-plugin/commit/06bd14b7015a5935d2276de15099ded801f3391c) | **最新整合实现**：在 `4d134dc` 上接入 CCU、scalar/SIMT 描述符、完整轮次多槽、connector 会话、三阶段初始化、完成门禁及 ABI 隔离构建。 |

### 4.1 启用最新整合路径

在每个 Attention 和 FFN 进程既有的 `connector_extra_config` 中增加：

```json
{
  "connector_extra_config": {
    "a5_full_features": true,
    "a5_slot_count": 2
  }
}
```

该片段只展示新增开关，其他模型和拓扑配置沿用原配置。全部 rank 的模型、拓扑、dtype、quant、最大 token 容量和 slot 配置须一致，并显式设置与实际通信组窗口分配相同的 `HCCL_BUFFSIZE`。不设置开关时继续使用默认累计 MTE 路径。

全特性路径要求 950 构建、匹配的 CANN/HCOMM、拓扑 layer 0 的 UBC_TP 或 UBC_CTP 通道，以及 eager 模式下每会话一个串行提交者和一个 NPU stream。初始化结束后才绑定首次数据调用的 stream。资源或通道不满足时初始化失败；切回默认路径需要重新启动通信域。

`a5_slot_count` 为 2～8，默认 2。`HCCL_BUFFSIZE` 对应整个域窗口，槽数增加会减少每槽 payload 容量；四算子以单槽容量校验 shape。FFN 另持有稳定注册的 FP16/BF16 staging，约占 `receive_capacity × hidden_size × 2` 字节，还需计入 descriptor/status 和 event 保活的在途 tensor。完整配置、布局公式和构建命令见[全特性集成指南](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/docs/npu/CAM_ASYNC_A5_FULL_FEATURES.md)。

## 5 已有验证和性能结论的范围

以下引用已有验收记录，本次文档更新未重新编译或运行测试。

| 阶段 | 已有验证结果 |
| --- | --- |
| [B0 基线](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/docs/npu/CAM_ASYNC_A5_BASELINE.md) | A5 完整算子包、PyTorch 扩展及 A3 六算子针对性回归构建；115 passed、46 subtests passed。 |
| [累计 payload-tiling](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/docs/npu/CAM_ASYNC_A5_PAYLOAD_TILING.md) | 267 passed、46 subtests passed、无 skip；覆盖共享 helper CPU 模型、故障负例和真实 dispatcher/Meta。 |
| [最新 full-features 实现 `06bd14b`](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/docs/npu/CAM_ASYNC_A5_FULL_FEATURES.md) | 全新源码目录下 A5 完整 CANN 包、扩展与 CCU 依赖库构建链接通过；实际 wheel 内容检查和独立解包目录加载通过；**298 passed、46 subtests passed、无 skip**。 |
| 整合后的 A3 针对性回归 | 独立目录构建 A2E/E2A 和四个 async 算子，扩展及 **20 项 Meta** 检查通过；不包含未改动的 GMM 构建资格。 |
| 最终源码一致性 | 格式调整后定向检查 **34 passed**；已核对提交源码与远程编译快照的 **616 个文件，零差异**。 |

最终 A5 编译产物包含以下实际入口：

| 算子 | Tiling keys | SIMT 入口 |
| --- | --- | --- |
| DS | 100、101、110、111 | 已检出 |
| DR | 100、101 | 不适用 |
| CS | 100、101、120、121 | 已检出 |
| CR | 100、101、102、103 | 不适用 |

整合版本的远程编译环境为 aarch64、CANN 9.0.0/BiSheng、Python 3.11、torch 2.8.0+cpu、torch-npu 2.8.0.post4，实际目标为 `__NPU_ARCH__=3101/dav-c310-vec`。**服务器无 NPU 卡，未进行设备通信、模型 E2E 或性能资格验证。** 编译、Meta 和 CPU 状态机检查不能换算成加速比例，也不能证明设备原子性、远端可见性或实际重叠。

默认路径继续使用 [async_cam_precision.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/npu/async_cam_precision.py) 和 [async_cam_performance.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/npu/async_cam_performance.py)。全特性新增 [async_cam_a5_full.py](https://github.com/GuangyuZhu04/afd-plugin/blob/06bd14b7015a5935d2276de15099ded801f3391c/tests/npu/async_cam_a5_full.py)，复用独立 CPU golden、metadata 校验和容差，按完整 DS→DR→CS→CR 轮次轮换槽位；该入口尚未执行真机测试，也不等同于并发轮次压力或性能基准。

评估收益时，先在真机完成精度、槽位复用、多个 expert chunk、多 DP、空 MoE、错误退出及并发轮次验收，再在同一 A5 硬件上比较 B0、累计 MTE 和全特性路径，保持拓扑、输入、dtype、quant、窗口及计时边界一致。重点观察各算子与完整 cycle 的跨 rank 最大延迟、p50/p95，以及 DMA 次数、缓存命中、event 等待、staging 复制成本和实际重叠。A3 与 A5 跨硬件的时间差需要单独报告，不能全部归因于代码优化。
