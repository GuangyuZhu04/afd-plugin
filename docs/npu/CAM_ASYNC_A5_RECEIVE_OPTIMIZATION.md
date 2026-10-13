# A5 async CAM：接收预取、两行复用与小批量核心选择

2026-10-13；从 P3/P4 提交 `fbf3a956061b7f17e30de71e3f0bac01901e7ce3` 派生，分支为 `GuangyuZhu04/a5-async-cam-recv-optimization`。本轮优化接入 A5 原四算子的正常路径；P2 CCU、P3 描述符与 P4 多槽协议仍是独立实验模块。原阶段分支和工作区保持不变。

compact-v2 布局、PyTorch/ACLNN ABI、operator keys、量化和 FP32 top-k 顺序均保持。远端只有编译环境，没有 NPU；编译和 CPU 模型不能证明设备同步、通信正确性或性能提升。

## 实现与预期收益

| 改动 | 实际路径 | 收益来源与限制 |
| --- | --- | --- |
| combine recv 输入预取 | A5 原 key 内自动选择；UB 不足沿原串行路径 | 两槽 MTE2/V 流水，尝试重叠下一块读取与当前块向量运算；分块和事件也增加开销 |
| dispatch recv 两行缓存 | P0 双缓冲可用，且 AIV 区间跨 expert 时选择 | 邻近 assignment 指向同一 wire row 时省去 MTE2；两路容量有限，大 batch 的 expert-major 遍历可能几乎无命中 |
| 小 batch source 核数 | 同源 DS/CR 使用 `min(platform_aiv, 48, B)` | 减少空 token 分区；奇数核数和 `SyncAll` 需要 A5 真卡资格化 |

入口：[CR kernel](../../csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h)、[预取规划/事件](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_combine_prefetch.h)、[DR kernel](../../csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h)、[两行缓存](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_receive_cache.h)、[source 核数策略](../../csrc/npu/ascend_kernels/utils/op_host/cam_async_host_tiling.h)。没有增加运行时配置、共享 mutable global state 或新通信协议。

### CR：有序 hidden tile 与双槽输入

保留原一行输入和三行完整 H 的 FP32 scratch，只从剩余 UB 分配第二个输入槽。tile 宽度取完整输入行与 32B 对齐剩余空间的较小值；第二槽不足 2048B 时回退原循环。这是避免大量微小 DMA 的结构门槛，尚未测量最优阈值。最后尾块可小于 2048B。只有至少两次 contribution，或一行需要多个 tile 时启用。

遍历顺序为 token → top-k slot → hidden tile。每个 hidden 元素保持原逻辑 K 顺序，执行独立 `Cast → Muls → Add`，最后 K 使用 `CAST_RINT` 写回原 dtype；不改为 FMA，不重新关联归约。expert occurrence 每 contribution 前进一次，后续 hidden tile 复用同一物理地址和 weight。非 `UseDispatchPrefix` key 先统计本核区间之前的路由。

每槽独立租用 `MTE2_V` ready 与 `V_MTE2` free event。先安排下一块读取，再计算当前块。最后 K 的输出复用当前输入槽，因此必须等 `MTE3_V` 后才归还 free。尾部消费全部 free，再用 `MTE2_S` 将 V→MTE2→Scalar 依赖闭合，随后才释放事件、进入原 `SyncAll` 并清零 completion flags。原 `PreProcess` 的远端 ready 等待保持；本地 event 不增加远端可见性保证。

### DR：完整 wire row 的两路复用

复用 P0 已有的两个 UB row，不增加最低 UB。key 包含实际 source wire 地址和 wire index；先使用无效槽，之后按两路 LRU 淘汰。miss 读取完整 payload 与量化 scale，hit 省去读取。两者都先等该槽此前的 MTE3-free，再经过 ready handoff，完整写出每个 assignment 的 payload 和精确 4B FP32 scale。

只有真实 miss load、ready、全部输出读取和 `MarkWritten` 已排队之后，才记住 key。hit 不能跳过 ownership 依赖。缓存只在单次 TP/source/chunk gather 内有效，下一次调用重新构造；保留 P0 Drain 和原 ClearFlags。单 selected expert，或本 AIV 完全位于初始 expert 内时，直接采用 P0 交替槽，不执行 tag/key 工作；恰好从 expert 边界开始时保守允许缓存。

两路缓存只能命中局部重复 row，不能实现跨所有 expert 的全局去重。密集大 batch 可能付出查询成本而无收益，必须按命中率和完整 cycle 延迟评估，不能仅凭省略的模拟 DMA 判定保留。

### source 核数与资源合同

DS/CR host 在确认 `B >= 1` 后共同选择 source 核数，`blockDim` 与 tiling `aivNum` 一致。同一 TP 组的不同 source rank 可分别使用不同核数，例如默认 `B=7/5` 各用 7/5 核。FFN 侧 DR/CS 仍用 `min(platform_aiv, 48)`。

窗口头部仍固定预留 48 行，dispatch/combine payload offset 不变。最低 UB 和 DS SIMT peak eligibility 仍按平台上限计算，不因较小 launch 核数扩大原接受范围。A3 保持原核数和串行 CR 路径。

harness 的 `--aiv-count` 是平台 capped capacity 的声明值，不是 launch override。JSON 分别给出 source 各 rank 的预期 DS/CR 核数和 FFN DR/CS 核数；这些是代码策略计算值，没有查询设备。原窗口与 UB 预检保持。

## 验证与复现

环境：aarch64、CANN 9.0.0、BiSheng clang15.0.5（2026-02-12），A5 实际宏 3101 / dav-c310-vec；Python3.11.15、torch2.8.0+cpu、torch-npu2.8.0.post4。

| 最终检查 | 结果 |
| --- | --- |
| A5 ACLNN 包 | A2E/E2A 与四个 async 完整编译、打包通过 |
| A5 PyTorch 扩展 | 编译通过，真实 dispatcher/Meta 加载回归通过 |
| A3 targeted 回归 | 独立副本编译 A2E/E2A 与四个 async 通过；未改原 registry，未包含本轮无关 GMM |
| 全部测试 | **203 passed、46 subtests passed，无 skip**；新增 24 项接收模型与 3 项 host/harness 检查 |
| dry-run / lint | precision、performance 和 SIMT 候选预检通过；Ruff check/format、Markdown 与 diff whitespace 通过 |
| 输入一致性 | 426 个编译/测试源码文件与本地最终输入指纹一致；Shell 统一 LF，文档在验证后补齐 |

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
AFD_BUILD_JOBS=4 bash csrc/npu/build_aclnn.sh "$PWD" 950
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 AFD_SKIP_ACLNN_BUILD=1 \
  MAX_JOBS=2 python setup.py build_ext --inplace

python -m pytest -o addopts='' -q -ra \
  tests/unit/package/test_cam_async_receive_optimization.py \
  tests/unit/package/test_cam_async_host_tiling.py \
  tests/unit/compat/npu/test_async_cam_harness.py
```

[新增 24 项模型测试](../../tests/unit/package/test_cam_async_receive_optimization.py)执行真实 CR Plan/Cursor/PipelineEvents、DR TwoRowCache/CanUseCache 和 P0 events。注入 backend 构建独立 MTE2/V/MTE3 FIFO，每个正例尝试 64 种交错。覆盖 FP16/BF16、H7168/H12288、hidden 尾块、K1/K8、非零 owned 起点、occurrence、AA/ABA/LRU、相同 index 不同地址、量化 scale、调用间重置和缓存筛选边界。独立 oracle 从原输入与路由重建数据；`-ffp-contract=off` 保持 FP32 分步舍入。

九个故障负例检出遗漏 ready/free、输出槽过早复用、错误输出槽、尾部 fence、FMA、pending hit 无 credit、提前归还 scale 和跨调用 stale cache。临时 SDK header 仅声明注入模型所需名称，不提供实现，不作为 CANN 支持证明。host/harness 检查执行共享 C++ 核数 selector，并验证原 UB/window 下限没有缩小。

完整回归还包含 `test_ascend_build_files`、`test_cam_async_build`、P0/P1/P2/P3/P4 package 测试和 `test_async_cam_performance`、`test_async_cam_golden`、`test_cam_async_ops`。扩展的无 Git 隔离快照使用进程局部版本 `0.0.0+a5.recv.opt.fbf3a95`；没有更换 CANN 或 Python 安装。

本地工作树为 `.cache/a5-recv-opt-worktree/`，证据在原仓库忽略目录 `.cache/a5-recv-opt-validation/`；远端目录为 `/home/zhuguangyu/afd_async_a5_recv_opt_20261013_6jRLgl`，A5 在 `source/`，A3 在 `a3_validation/`。原工程没有覆盖。日志为 `aclnn_a5_recv_opt_build.log`、`aclnn_a3_recv_opt_regression.log`、`binding_a5_recv_opt_build.log`、`recv_opt_checks.log/xml`，`artifact_manifest.json` 与 `source_fingerprint_verification.json` 保存指纹核验。二进制和日志不提交 Git。

临时启动的 `cam_guangyu_dev_90` 在验证结束后已恢复为原 `exited` 状态。

| 产物 | 字节 / SHA256 |
| --- | --- |
| AFD_ascend950.run | 790017；808e7bab261faf6b54b79dd20cd2a198699db047e63d31155df37a5e09c2ef68 |
| A3 targeted 包 | 643022；8ac5c1e9cf2fe3e8007a6beb159d724a97b0d215908fdd9879d6b6932ae4f1cb |
| _C_ascend.cpython-311-aarch64-linux-gnu.so | 399352；52c01f03c0f40dac5c34fbba38db44e1f06d44b35b10b29a6a85afb89e512a39 |

## A5 真卡对照

先跑精度，再测同配置下的 per-op 与完整 cycle；对照父分支 `GuangyuZhu04/a5-async-cam-simt-ccu-slots`，也保留 B0 对照。不同 checkout 使用独立进程与匹配 vendor。至少覆盖 B1/B5/B7/B48/B49 和较大 B、不同 H/K、quant0/1、FP16/BF16、TP1/2/4、多 expert/chunk、空 MoE、热点/稀疏及重复窗口复用。重点验证奇数 launch 的 `SyncAll`、prefix 与非 prefix、CR partial tile 输出、DR hit 的 payload/scale lifetime。

下面的 48/196608 是可通过预检的示例声明值，运行时需替换为目标平台核对过的值：

```bash
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_precision.py --soc-version 950 \
  --aiv-count 48 --ub-budget-bytes 196608 \
  --lengths 7,5 --hidden-size 7168 --capacity 16 \
  --dtype bf16 --dynamic-quant 1

timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_performance.py --soc-version 950 \
  --aiv-count 48 --ub-budget-bytes 196608 \
  --lengths 512 --hidden-size 7168 --capacity 1024 \
  --dtype bf16 --dynamic-quant 1 --measure-mode cycle \
  --warmup 5 --iterations 50 --output .cache/recv-opt-cycle.json
```

Profiler 确认路径选择、MTE2/V 重叠、DMA/tile 数、event 等待、DR cache 命中与 AIV 利用率。以相同拓扑、输入、window、warmup/iterations、计时边界比较 p50/p95 和每轮跨 rank 最大延迟；CPU 模型和 dry-run 没有提供加速幅度。
