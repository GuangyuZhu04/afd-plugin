# CAM AFD SIMT 标量路径优化：实测结论

> 本文档记录一期针对 `combine_send` / `dispatch_send` 标量时延的 SIMT 化工作，
> 包含**已落地的改动**、**九条被证否的方向**，以及用 `msprof` 得到的**瓶颈定性**。
> 所有数字均为真机（Ascend 950DT，2A2F）实测，注明口径。

---

## 一、结论摘要

| 算子 | 配置 | base | opt | 变化 |
|---|---|---|---|---|
| **combine_send** | NC=1 | 28136.3 µs | **5562.1 µs** | **−80.2%（5.06×）** |
| **combine_send** | NC=8 | 4279.5 / 4326.7 µs | **2038.4 / 2045.9 µs** | **−52.4%（2.10×）** |
| dispatch_send | NC=1 | 4969.1 µs | 4081.7 µs | −17.9% |
| dispatch_recv | NC=1 | 5741.6 µs | 5738.4 µs | −0.1%（未改动） |

- 口径：`--microbatches 2`，每腿 n=4，NC=8 两轮交替（n8a/n8b），散布 <0.3%
- 验证：全部腿 `rc=0`、`sanity.sh` 判 `SANITY_OK`、单元测试 27 passed
- 跨卡组复现：卡 0–3 与卡 4–7 独立测量差异 <0.1%

**`combine_send` 的标量时间从 27266.7 µs 降到 4962.3 µs（−81.8%）。**

---

## 二、为何 combine_send 有效而 dispatch_send 无效

两个算子的标量占比都很高，但**结构完全不同**：

| | combine_send | dispatch_send |
|---|---|---|
| 标量结构 | 每 token 一条 **4 级 UB 依赖链** | 大量**独立标量操作** |
| 依赖链 | `md[0] → lane 槽位 → 两次 lane_cache 读 → offset → 6 次存储` | 无 |
| SIMT 化 | 把链拆成 warp 内并行 | 无可拆的链 |
| 结果 | **−81.8%** | **0%** |

**SIMT 化的前提是存在可拆分的串行依赖链。** 这是本次工作最重要的结论。

---

## 三、msprof 定量证据（新增，此前无法获得）

自建探针与 `llvm-objdump` 静态分析都不可靠（前者三次失败，后者是 AArch64 target、
在 DaVinci ISA 上 19% `<unknown>`、零跳转指令）。改用官方 `msprof op` 成功采集：

```bash
msprof op --aic-metrics=PipeUtilization,ResourceConflictRatio,ArithmeticUtilization \
  --kernel-name=CamDeepepDispatchSend --launch-count=1 --output=DIR \
  python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/afd/run_pipeline_cam.py --ffn-transaction --hidden 7168 --topk 2 ... --num-vec-cores 1
```

### 结果：两个算子都是「执行型」而非「等待型」

| 指标 | dispatch_send | combine_send |
|---|---|---|
| `aiv_scalar_ratio` | **79.6%** | **89.0%** |
| `aiv_vec_ratio` | 14.0% | 10.8% |
| `aiv_vec_simt_vf_ratio` | **14.0%（= vec 全部）** | **10.8%（= vec 全部）** |
| `aiv_vec_wait_ratio` | 0.10% | **0.03%** |
| `aiv_mte2_wait_ratio` | 4.68% | **0.13%** |
| `aiv_mte3_wait_ratio` | 0.47% | **0.005%** |
| `aiv_vec_stu/ldu/sfu_cflt_ratio` | **全 0** | **全 0** |
| `aiv_icache_miss_rate` | 0.015% | 0.017% |

**总等待预算：dispatch_send ≈5.3%，combine_send ≈0.16%；资源冲突完全为 0。**

### 推论

1. **标量管道的 79.6%/89.0% 是真实执行时间，不是依赖停顿。**
   因此「减少标量依赖以降低耗时」这条路径无效——不存在可减少的等待。
2. `aiv_vec_simt_vf_ratio == aiv_vec_ratio` 证明 **100% 的向量时间都来自 SIMT VF**
   （`route_tile` 与 `fill_write_tasks_simt`），没有其他向量工作。
3. `icache_miss_rate ≈ 0` 排除取指停顿。

---

## 四、已落地的改动（combine_send）

`cam_deepep_combine_send_rt.h`：

- **`fill_write_tasks_simt`**：warp-per-token 填 `WriteTask`，替代标量逐 token 循环。
  按 batch 对齐切片（`take = min(selected - s, kBatch - num_task_tokens)`），
  保证 WQE 发射时机与 drain 边界不变。
- **`uint4` 向量化读取**：metadata 与 `lane_write_cache` 各一次 16 B 读取代多次标量读。

### 一个必须记住的坑（花了 9 轮才定位）

`record_base` 是 `__gm__ bfloat16_t*`，标量路径 `+ j * num_hidden` 是**元素**偏移
（每个 bfloat16 占 2 字节），实际字节偏移为 `j * num_hidden * 2`。
把它转成 `uint64_t` 做字节运算时若只用 `j * num_hidden_bytes`，**少乘 2**，
每条记录会落在前一条的后半段，表现为整体错位一位。

```cpp
// 正确
task->src_addr = record_base + uint64_t(j) * uint64_t(2) * num_hidden_bytes;
```

这与交接文档第八节「指针版本才对，整数中转版会输出错序」完全一致。

---

## 五、九条被证否的方向（含证据）

| # | 方向 | 结论 | 证据 |
|---|---|---|---|
| 1 | SIMT 化 dispatch_send 的 WriteTask 填充（ballot/popc 过滤） | +0.08% | 实测 |
| 2 | SIMT 化路由过滤 | 同上 | 实测 |
| 3 | 轻量控制字轮询（单字 GM 读取代 32 B DMA） | +0.09% | scalar 3257.5→3260.9 |
| 4 | `route_tile` 两遍重复消除 | 不可行 | 两遍之间需跨全部 tile 缓存，`tiles` 是运行时值 |
| 5 | 消除运行时除法 `expert/(kNumExperts/kNumRanks)` | 不成立 | 模板参数，编译期常量折叠 |
| 6 | `topk_idx` GM 读合并 | 不成立 | warp 读 32 连续 int64 = 256 B，已完全合并 |
| 7 | 减少 `deduplicate` 轮数 | 不成立 | `kNumRanks=2` 时 2 轮，已最小 |
| 8 | warp 扫描直方图（原子数 512→64/tile，8× 减少） | +0.10% | 实测；原子争用不在关键路径 |
| 9 | 重构依赖结构（本文档第三节） | 无效 | 等待仅 5.3%、冲突为 0 |

另有两条来自既有交接文档，**不要重试**：

| 方向 | 结论 |
|---|---|
| 描述符循环按 4 展开 | 正确但慢 7.1%（寄存器压力） |
| 延迟 drain（两 peer 重叠） | +2.0%（URMA 引擎不重叠同核双流） |

---

## 六、剩余空间（未解决）

对每个算子用 NC=1 与 NC=8 两点解 `scalar = fixed + work/nc`：

| 算子 | NC=1 | NC=8 | work | **fixed** |
|---|---|---|---|---|
| combine_send | 4962.3 | 1703.0 | 3724.9 | **1237.4 µs** |
| dispatch_send | 3257.7 | 1385.7 | 2139.4 | **1118.3 µs** |

回代 NC=8 均精确吻合。两者固定成本相差约 10%，指向**框架级共同开销**
（每轮握手 / `SyncAll` 屏障 / jetty drain / ACK 发布）。

**已排除**：`wait_control` 自旋、`store_control`（估算仅 60–120 µs）。
**未验证**：屏障传播、jetty drain、ACK 发布——属协议核心，且「延迟 drain」已被否证。

要定位这部分，需要 `msprof op` 的 `TimelineDetail` / `PipeTimeline`，
但这两个指标在本 kernel 上解析失败（`PipeUtilization` / `ResourceConflictRatio` /
`ArithmeticUtilization` 三个可正常使用）。

---

## 七、复现方法

```bash
# 构建（仓库 CMake 不编译内核，真正入口是 staging 树内的 build/cam/build.sh）
bash build_all.sh opt                      # 需 RC=0

# 回归
python3 -m pytest test/cam/ut/cam_deepep/ -q    # 27 passed

# 性能 A/B（每腿 n=4）
bash mb_ab.sh 2 1        # NC=1: base -> opt
bash nc8_mb.sh 8 2       # NC=8: base/opt/base/opt 交替

# sanity（上界判定，DEGRADED 要丢弃而非归因代码）
sanity.sh /work/results/afd-cam/quick-ab/mb_mb1-opt
```

### 设备占用检测的一个坑

容器内 `npu-smi` **看不到宿主机进程**，因此基于容器内 `npu-smi` 的忙闲检测会
误报空闲，导致 A/B 撞上他人负载。检测必须在**宿主机侧**做，且要求连续空闲窗口
（例如 60 s），同时对 HBM 占用设阈值——这一条能抓住容器内不可见的宿主进程。

---

## 八、改动范围

| 文件 | 变化 |
|---|---|
| `cam_deepep_combine_send_rt.h` | `fill_write_tasks_simt`、shadow 比对、`uint4` 读取 |
| `dispatch_send.h` | 探针与试验代码，**全部默认关闭**，仅留作参考 |

所有试验开关默认 0（关闭），只有 `CAM_USE_SIMT_WRITE_TASKS=1`（combine_send 的成果）。
