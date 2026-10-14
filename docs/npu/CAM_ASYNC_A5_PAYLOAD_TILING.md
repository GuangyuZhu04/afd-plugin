# A5 async CAM：发送流水与接收分块

2026-10-14；从 `3a265b45aa0b21b9d33038ec30d5a220668456cb` 派生，分支为 `GuangyuZhu04/a5-async-cam-payload-tiling`。本轮按 DS token 双缓冲、DR 连续行批量搬运、CR tile 级 FP32 scratch 三项独立提交推进。已有阶段分支和工作树保持不变。

所有改动限定 A5 正常算子路径，保留 compact-v2、PyTorch/ACLNN ABI、operator keys、原资源接受范围、量化算术与 FP32 top-k 顺序。服务器只有编译环境，无 NPU；以下模型与编译证据不提供设备通信资格或性能增益结论。

## DS：两槽 token 流水

原 DS 在每次向目标 rank 发送 payload 后等待 Scalar。本轮从 mandatory arena 之后的 spare UB 分配第二个输入行槽；两个槽都沿原布局，将 FP16/BF16 输入与原位 wire 输出视图重叠。共享 FP32 量化 scratch 不复制。只有本 AIV 至少两个 token、第二行可容纳、wire 行不超过输入行 extent 时启用，已扣除 SIMT reserve 的 arena 才参与规划；其余沿原路径。

先加载当前槽，等待 input ready，再安排下一槽读取。量化仍每 token 调用原算法一次。全部目标 rank 按原首次命中顺序读取当前 wire 行，每 assignment 的地址项、稀疏 rank 位图和 token 末 offset 更新保持。直到全部 peer payload/scale 源读取排队后，才归还该槽的 MTE3→MTE2 credit。

q0 的 input ready 为 `MTE2_MTE3`，q1 为 `MTE2_V`。q1 另为每槽租用独立 `V_MTE3` wire ready 与 `S_MTE3` scale ready；取消逐 peer Scalar 等待后不能连续借用同一个尚未消费的事件号。slot 的 free→next input ready 依赖保护这些租用事件的下一次 signal。原 QuantProcess 的数值操作、V/S 读取依赖保持；只有 A5 private 调用的尾部 handoff 使用独占 ID，A3 沿原签名和同步。

地址表 staging 与 payload 槽是独立生命周期。`FlushExpertTokenOffset` 仍使用原 `S_MTE3/MTE3_S` 等待，不能用 token payload credit 证明 metadata staging 已可重写。调用尾部先 drain 两槽，闭合 MTE3→MTE2→Scalar 并释放所有事件，再执行原尾部地址 flush 和 SetFlags。

收益候选来自跨 token 重叠读取、量化和发送，以及减少逐目标 Scalar 等待；元数据 flush 和原量化 Scalar 读取仍可能限制重叠。小 batch 每核只有一 token 时回退，不能据默认小样本判断流水速度。

入口：[DS kernel](../../csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h)、[生产 buffer plan/events](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_dispatch_pipeline.h)。真实 SDK `kernel_tpipe_impl_c310.h` 的 Alloc 在 event bitmap 中占位、Fetch 只取未占位 ID、Release 清位，因此原 metadata Fetch 不会选到已租用的 scale ready ID。

[新增10项 DS 模型](../../tests/unit/package/test_cam_async_dispatch_pipeline.py)执行生产 plan/events，分别调度 MTE2/V/MTE3 与 Scalar 参数/metadata 依赖。独立 raw-route/opaque-wire oracle 检查多 peer 发送和每 assignment 地址；覆盖 owned slice、原始/量化 wire，并用提前 credit、晚读参数、metadata fence 遗漏、shared scratch 覆写、tail drain 遗漏、borrowed ready 重用负例检出生命周期错误。wire 量化 bytes 是 fixture，不模拟或修改原量化算术，不作为 CANN API 或硬件完成证明。

| DS 独立阶段 | 结果 |
| --- | --- |
| A5 / A3 targeted | A2E/E2A 与四 async 完整编译、打包通过；A3 使用独立副本 |
| A5 扩展 / 完整回归 | 扩展构建及真实 dispatcher/Meta 加载通过；**236 passed、46 subtests passed，无 skip** |
| 检查 | precision/performance/SIMT 候选 dry-run、Ruff、Markdown、diff whitespace 通过 |
| 产物及输入 | 实际 ELF 保留 DS100/101/110/111、CR100/101/102/103；432个编译/测试源码指纹一致 |

DS 包为841567B，SHA256 `70f03f4e21a807bdcb4df1e1a7487b3a9cf44e850e97c95ed87e2be19b0402f8`；A3包647750B，`1542994f4fe115523cbf87cbf88940ee01d8079c04f45fbf344cfc1f1ce5d0ba`；扩展399352B，`844d41f9a3dccc3a03026449d0addd612f588e3e42b64c833184d03f96c419de`。

## DR：同源、同 expert 的连续行

入口：[DR kernel](../../csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h)、[run plan/cursor](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_receive_runs.h)。保留 mandatory 输入行及完整地址 slice，只从其后的 spare UB 分配两个独立连续 tile；不能把旧输入行和 P0 第二行当作相邻多行内存。

每 tile 至少容纳两行，结构上限为64KiB/128rows，不是实测最优值。预检存在同 expert 内连续 wire index 后才选择新路径；全随机、单行或空间不足沿 cache/P0。cursor 保持原 expert→TP→token 输出位置，不跨 expert、source call 或 AIV 区间；65535之后不能回卷为0并合并。混合 singleton 仍按原顺序复制。

连续 wire 行用一次 MTE2 读取。q0 payload 连续批量写回；q1 用 `DataCopyPad{rows,H,1,0,0}` 跳过每行32B padding。现场 dav_c310 `CopyUbufToGmAlignV2` 明确 UB source stride 为32B blocks、GM destination stride为bytes，不能把32填入该source参数。每行 scale 仍从完整 wire row 的 H 偏移复制原4B，全部 payload/scale 读取排队后才归还 tile credit，Drain后进入原 chunk flags/ClearFlags。

[新增16项 DR 模型](../../tests/unit/package/test_cam_async_receive_runs.py)执行生产 plan/cursor 和 P0 events，使用独立 raw expert/peer byte oracle。覆盖 empty expert/AIV、expert边界、partial/tail、多个AIV slice、混合 singleton、不同peer、random/UB fallback和uint16边界；遗漏reuse wait、提前scale credit、错误payload/scale stride及tail drain负例均可检出。模型不证明硬件DMA远端可见性或速度。

| DR 独立阶段 | 结果 |
| --- | --- |
| A5 / A3 targeted / 扩展 | 六算子完整编译、打包和扩展构建通过；真实 dispatcher/Meta 加载通过 |
| 完整回归 | **252 passed、46 subtests passed，无 skip**；包含 DS10 和 DR16 项 |
| 静态 / 预检 | Ruff、Markdown、diff whitespace及三项dry-run通过 |
| 实际 keys / 输入 | DS/CR key集合保持；435个编译/测试源码指纹一致 |

DR 包为851454B，SHA256 `5fca0af85212de3568f142cf65d26337391f79a73ed1e6c9a6ef332775da4783`；A3包651077B，`4b51e1cc9e4fcedad865d220f7ee726c3086bd0ad595453f9c3eba921bb5d334`；扩展399352B，`6f2bcff455b0e44fd5bf2b48470e56d78202552f39afb5b712cd21ed737afaea`。

## CR：每个 hidden tile 保持原 K 顺序

入口：[CR kernel](../../csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h)、[tile plan/cursor/捕获与退休](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_combine_tile_plan.h)。原三个完整 H 的 FP32 scratch 改为三个 tile 大小 scratch；两输入槽也按 tile 分配。K 个 UINT64 地址和 FP32 weights 有 checked、32B 对齐 UB 预算，没有新增固定 top-k 上限。旧 mandatory full-H 布局和预处理峰值仍须满足，不因较小 scratch 放宽 host 接受范围。

仅在 tile<H、输入 tile 至少2048B、且比旧 spare-only 预取 tile 更大时选择新路径；完整行已可双缓冲时保留旧快路径。2048B 是沿用的结构门槛，未测量最优阈值。K 描述符预算不足或不能改善 tile 时沿旧预取/串行路径。

每 token 先按 K 顺序固定 source row/weight，expert occurrence 每贡献仅推进一次。随后 hidden→K 遍历，每元素仍执行独立 `Cast→Muls→Add`，最后 K 使用 `CAST_RINT`；不引入 FMA或重新关联归约。prefix key 保留原 seed，非 prefix 先统计本 AIV 起点之前的路由。

输入槽仍通过 MTE2_V/V_MTE2 租用事件交接，最后 K 的输出复用该输入槽，因此保留 V_MTE3/MTE3_V。每 token 结束消费两槽 free 并 MTE2_S fence，才可重写下一 token 的描述符；event ID 在整调用内保持租用，最终 Drain 后才进入原 SyncAll/flags reset。三份 scratch 在 hidden tile 间复用，不能在旧最终 Cast 读取完之前 Duplicate 下一份 sum。

[新增15项 CR 模型](../../tests/unit/package/test_cam_async_combine_tile_plan.py)执行生产 plan/capture/cursor/retirement/events，独立 raw routing 重建 occurrence、wire 地址和 FP32 分步舍入 golden。覆盖 FP16/BF16、prefix/非prefix、partial hidden、K1/K8、大于4GiB地址和资源回退；晚读地址/weight、缺少token退休、提前输出credit、遗漏tail drain、错误K顺序与FMA负例均可检出。临时 SDK header 是声明 stub，不提供 API 实现。

收益候选来自较大的预取 tile 和较少的 full-H scratch 占用，代价包括 K 描述符读写、token 退休和分块 DMA。部分资源边界下相对原串行路径可能增加 DMA 数，实机必须同时检查新旧路径和整体 cycle。下面是通过原 harness 资源预检的边界例子；48/262144是声明值，需要替换成目标平台核对后的值，不能据 dry-run 推断实际选路：

```bash
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_precision.py --soc-version 950 \
  --aiv-count 48 --ub-budget-bytes 262144 --hidden-size 12288 \
  --lengths 512 --experts-per-rank 256 --top-k 2 --capacity 1024 \
  --dtype bf16 --dynamic-quant 1
```

| CR 独立阶段 | 最终结果 |
| --- | --- |
| A5 / A3 targeted / 扩展 | 六算子完整编译、打包及扩展构建通过；真实 dispatcher/Meta 加载回归通过 |
| 完整回归 | **267 passed、46 subtests passed，无 skip**；新增 DS10、DR16、CR15 共41项 |
| 静态 / 预检 | Ruff check/format、Markdown、diff whitespace及三个原dry-run、上述资源边界dry-run通过 |
| keys / 输入 | 实际 DS/CR key集合保持；438个编译/测试源码指纹一致 |

最终 A5包为864369B，SHA256 `6fbba18c2aea91efe75429c73521bef0e8c5cd2a9e8f229664855c27c6196909`；A3包655857B，`0823cfcd00db28022a3e2f23cd9fcf7ec5b78e0fe92d8aa6053d4c34a36fbc22`；扩展399352B，`4a05984023fbde1ee3476bb4b3901f63772b7ceb28888c4f190754626a3d441f`。没有 NPU 执行或可归因的 ISA/IR 指令对照；实际选路、完成语义及收益仍需设备证据。

## 构建与隔离

环境：aarch64、CANN9.0.0、BiSheng clang15.0.5（2026-02-12）、A5 宏3101/dav-c310-vec；Python3.11.15、torch2.8.0、torch-npu2.8.0.post4、CPU g++13.3.0。没有更换运行时或 SDK。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
AFD_BUILD_JOBS=4 bash csrc/npu/build_aclnn.sh "$PWD" 950
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 AFD_SKIP_ACLNN_BUILD=1 \
  MAX_JOBS=2 python setup.py build_ext --inplace
```

每项编译快照只包括其父提交与本项文件白名单；三项测试和源码分阶段加入，不将其他未提交改动混入验证。A3 在各阶段独立副本只选择 A2E/E2A 与四 async，未改仓库 registry，未包含无关 GMM。Git archive 使用 `core.autocrlf=false`，Linux 文本按 LF 核对。

独立阶段父提交分别为3a265b4、5574c9e、fbb268b；完整回归包括已有 build/host/P0/P1/P2/P3/P4、receive/sparse 模型、harness/performance/golden 和真实 dispatcher/Meta 测试。本项新增 wrapper 的 SDK header 仅用于 CPU解析；三轮实际 CANN包单独证明 SDK API可编译，不将 stub 当作 SDK证据。

本地工作树为 `.cache/a5-payload-tiling-worktree/`，证据在原仓库忽略目录 `.cache/a5-payload-tiling-validation/`；远端为 `/home/zhuguangyu/afd_async_a5_payload_tiling_20261014_2T0rjz/stage{1,2,3}/`。各阶段保留独立源码、日志、XML、包与扩展，以及 input/artifact/verification manifest。日志和二进制不提交 Git。

日志为各 stage 下的 `aclnn_a5_payload_stageN_build.log`、`aclnn_a3_payload_stageN_regression.log`、`binding_a5_payload_stageN_build.log`、`payload_stageN_checks.log/xml`；`artifact_manifest.json`、`input_manifest.json`、`verification.json`、`git_verification.json` 及 `package_keys.json` 保存指纹和keys。SDK stride/event pool摘录与 `tile_boundary_plan.json` 保存在本地证据目录。

临时启动的 `cam_guangyu_dev_90` 在三项验证和证据收集结束后已恢复为原 `exited` 状态。

## 实机对照与保留条件

在独立进程使用各提交配套 vendor，先跑现有 independent golden 精度，再以相同输入、拓扑、window、warmup/iterations 和计时边界比较每算子及整轮跨 rank 最大延迟。覆盖 FP16/BF16、quant0/1、TP1/2/4、大小 batch、不同 H/K、同 rank 多 expert、稀疏/热点/均衡路由、empty MoE、多 chunk 和重复窗口复用；精度 tolerance 不放宽。

DS 特别检查所有 peer/scale 读取结束前不能覆写槽位、slot-specific ready ID 不提前重用、metadata flush 不因 payload drain 而省略。Profiler 确认路径选择、MTE/V 重叠、DMA 数量和等待占比；CPU 调度重叠不能换算为硬件加速百分比。

这些改动位于 plugin 自有后端，不新增 vLLM runtime patch。实机资格完成后再决定保留范围，并评估向算子来源贡献；SDK、窗口资源或完成域合同变化时重新审查依赖。CCU 正式接入仍需 A5 实机验证注册、stream 交接、源消费和远端可见性，当前独立实验不自动发布 ready 或 credit。
