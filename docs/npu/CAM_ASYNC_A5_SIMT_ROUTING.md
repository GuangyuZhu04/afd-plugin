# A5 async CAM：P1 SIMT 路由统计与前缀

2026-10-12；本轮从P0提交 `12694341a2fa3d709d63c37c26020033ab76fc10` 派生，分支为 `GuangyuZhu04/a5-async-cam-simt-routing`。B0和P0分支保持原提交；新代码仅在独立worktree实现。

这是设计路线第三步的首版混合实现：dispatch send的计数和可容纳的前缀扫描使用SIMT，载荷、量化、地址表生成、ready协议以及combine recv的加权顺序沿用P0。P0的combine send/dispatch recv MTE流水继续保留。服务器没有NPU卡，设备正确性和性能尚未验证。

## 实现范围

| 阶段 | 实现与合同 |
| --- | --- |
| Histogram | 每AIV保持原token分区，将完整token的ID tile通过MTE2读入64KiB UB；128个SIMT线程按token分工，UB整数原子计数每个assignment，lane内64位rank mask为每token/rank仅计一次payload |
| Prefix | 保留原MTE统计聚合与两次SyncAll；SIMT扫描本AIV负责的完整列，写入原aligned UB staging，然后由原MTE3写统计矩阵；所有AIV统一检查最坏列数预算，不足时使用原scalar prefix |
| 稳定布局 | 原payload和地址表遍历保持token次序；原 `M+E` 矩阵stride与inclusive per-AIV prefix保持不变，combine recv继续读取前一AIV行和最后active行 |
| 空分区 | 写入零计数并参与两次barrier；无token的AIV仍可能负责prefix列 |
| 其他算子 | dispatch recv/combine send继承P0；combine recv未改；不引入CCU或新的wire协议 |

实现入口：[dispatch send](../../csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h)、[VF wrapper](../../csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send_simt.h)、[共享整数算法](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_routing_columns.h)、[资源门禁](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_simt_routing_budget.h)。

统计原子操作只决定整数总数，不分配payload/address slot。SIMT仅访问本AIV的UB，GM窗口读写仍由原MTE路径完成，避免引入不同AIV共享cacheline的GM写回问题。64KiB ID tile按 `floor(16384/topk)` 个完整token划分；全局ID偏移先按64位计算，完成后使原GetIds page缓存失效，防止payload阶段复用tile误读。

## UB门禁与编译隔离

- A5、`floor(batch/active_aiv) >= 32`、`1 <= topk <= 16384`，且原dispatch-send峰值UB加136KiB可容纳时选择SIMT。32 token是一个warp的结构门禁，尚无实测crossover。
- 136KiB由公开文档Data Cache上界128KiB加编译器8KiB保留构成，是保守预算；不是当前芯片默认/实测cache配置。SIMT key在TPipe InitBuffer前减去该空间，后续全部分配只使用剩余arena。
- Scalar keys `100/101` 保留完整原UB；SIMT keys `110/111` 经预处理条件单独引入VF声明和调用。原operator ABI与tiling结构布局保持不变。
- Histogram可用与prefix staging可用分别判断；exact资源边界允许SIMT计数加scalar prefix，不强行分配额外scratch，不提高host/harness原最低资源要求。

MTE2加载到VF读取使用MTE2_V；每个计数VF后用V_MTE2归还ID tile，下一tile才可覆盖。统计/prefix的MTE3写入前使用V_MTE3。Scalar与Vector的原初始化、读取依赖继续显式同步。事件编译通过仅证明接口可用，VF完成、UB原子和硬件流水仍需实卡资格。

## 验证

环境：aarch64、CANN9.0.0、BiSheng clang15.0.5（2026-02-12）、A5实际宏3101/dav-c310-vec；Python3.11.15、torch2.8.0+cpu、torch-npu2.8.0.post4。

| 检查 | 最终结果 |
| --- | --- |
| A5 ACLNN完整包 | A2E/E2A与四个async编译打包通过；两个dtype的DS均包含100/101/110/111四key |
| 真实SIMT入口 | 两个DS对象实际包含count/prefix `simt_entry`，函数大小分别728B/488B；该结果不表示运行时选路或实机性能 |
| Scalar隔离 | 两个dtype的100/101共四个native entry，`.text`函数字节SHA256及逐entry `.ascend.meta`内容与P0完全一致，证明原scalar code/resource未受VF污染 |
| A3针对性回归 | 在独立副本仅构建A2E/E2A与四async，DS仍只有100/101，无SIMT入口；不包含本轮未修改的GMM |
| A5 PyTorch扩展 | 构建、真实加载及dispatcher/Meta通过 |
| 针对性测试 | **140 passed、46 subtests passed，无skip**；含13项SIMT生产整数算法probe、原CPU golden、真实Meta及P0事件模型 |
| dry-run与lint | 原样例及B2048/H7168/q1候选预检通过；Ruff check/format、Markdown和diff whitespace通过 |

CPU测试使用普通C++编译器执行生产共享算法，以独立token-first set oracle比较所有原始计数、每个inclusive prefix与combine recv起点；乱序lane调度、64KiB多tile、lane多token/多column、空/不均匀AIV分区、active48、rank63、385列的9/8列aligned stride、padding sentinel、UINT16边界和预算溢出均有覆盖。此模型不验证NPU原子和事件行为。

复现测试：

```bash
SOC_VERSION=950 python -m pytest -o addopts='' -q -ra \
  tests/unit/package/test_ascend_build_files.py \
  tests/unit/package/test_cam_async_build.py \
  tests/unit/package/test_cam_async_host_tiling.py \
  tests/unit/package/test_cam_async_mte_pipeline.py \
  tests/unit/package/test_cam_async_simt_routing.py \
  tests/unit/compat/npu/test_async_cam_harness.py \
  tests/unit/compat/npu/test_async_cam_performance.py \
  tests/unit/compat/npu/test_async_cam_golden.py \
  tests/unit/compat/npu/test_cam_async_ops.py
```

正常checkout构建沿用 [B0记录](CAM_ASYNC_A5_BASELINE.md)。无.git的隔离快照仅在扩展构建进程设置 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_AFD_PLUGIN=0.0.0+a5.simt.routing.1269434`，不修改系统安装。

## 隔离与产物

本地新worktree：`C:/D_disk/Github/afd-plugin/.cache/a5-simt-worktree`。远端新目录：`/home/zhuguangyu/afd_async_a5_simt_routing_20261012_ZGua3P`；source/为A5快照，a3_validation/为独立A3副本。远端既有A5项目和P0/B0验证目录未覆盖。临时启动的容器完成工作后恢复为exited状态。

本地产物与日志位于原工作区忽略目录 `.cache/a5-p1-validation/`，未纳入Git；远端上述新目录也保留证据。183个最终csrc/npu源码快照指纹与本地输入逐文件一致，Shell脚本仅在Linux快照统一LF行尾。

| 产物 | 大小 / SHA256 |
| --- | --- |
| AFD_ascend950.run | 762672B；305cc5c1c744d2cb99c6858952d224ded1ebd9fc4c8b3e60628eda552390e7dc |
| _C_ascend.cpython-311-aarch64-linux-gnu.so | 399352B；f72eedaf8b96a0474895419ab0221eb0c28bdaf2a14bd461fb65481816368fbc |
| A3回归包（远端） | 632d3da33896f7d4e0b05180b45658feb05dfb2ad403865b96dbe42c334c912b |

日志：`aclnn_a5_p1_build.log`、`aclnn_a3_p1_regression.log`、`binding_a5_p1_build.log`、`p1_checks.log/xml`、`key_resources.json/log`、`scalar_isolation.json`及`artifact_manifest.json`。CANN通过预处理后的tiling key比较枚举入口；本版让A5的新key比较在infer阶段可见，只在选定key的编译中展开VF主体。二进制核验避免将整包编译成功误当作SIMT已进入产物。

## 实机对照

先完成B0设备roundtrip，再用相同拓扑、输入、window、dtype和计时边界分别运行B0/P0/P1。P1继承P0，因此P1对P0可定位本轮SIMT净影响，P1对B0可比较累计变化；各版本独立进程加载对应vendor，避免相同operator混装。

官方内存模型把用户UB区域、8KiB reserved和SIMT Data Cache按低址到高址排列，因此本版限制低址arena并留136KiB。现有ELF元数据没有DC实际配置/边界字段，不能据此证明default cache大小或实际物理边界；实机须核验arena base/end与reserved/cache区域没有重叠。

实机需验证100/101与110/111选路、UB不足fallback、prefix staging不足fallback、多tile ID复用、不同active AIV、稀疏/空expert、quant0/1、TP1/2/4、UINT16边界、窗口复用及延迟注入。harness的 `--ub-budget-bytes` 使用原baseline的完整native每核UB预算（减SIMT reserve之前）；扣除由native key完成，调用者不重复扣除。Profiler核对VF和MTE同步、UB/DC资源和route阶段时长。32-token门禁不是已证明最快的阈值；SIMT原子争用、VF启动、reserve减少arena和保留的scalar payload遍历都可能抵消收益。实测后再调整选择策略。

下面是通过CPU预检的候选样例（默认2A2F/TP2/R4/K2），48个AIV和320KiB UB为声明的资源，需要先在实机核对；原DS峰值137824B加reserve139264B共277088B，2048/48至少42 token。

```bash
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_precision.py --soc-version 950 \
  --aiv-count 48 --ub-budget-bytes 327680 --hidden-size 7168 \
  --lengths 2048 --capacity 1024 --window-mb 64 --dtype bf16 --dynamic-quant 1

# 精度通过后分别运行B0/P0/P1，并由Profiler确认实际key与VF。
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_performance.py --soc-version 950 \
  --aiv-count 48 --ub-budget-bytes 327680 --hidden-size 7168 \
  --lengths 2048 --capacity 1024 --window-mb 64 --dtype bf16 --dynamic-quant 1 \
  --measure-mode cycle --warmup 5 --iterations 50 --output .cache/p1-cycle.json
```

预算资料：[CANN9.0内存模型](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00019.html)、[CANN9.0混合内存与保留空间](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10073.html)、[SIMT callee约束](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10052.html)。实现语法以现场3101/CANN9.0 SDK为准。
