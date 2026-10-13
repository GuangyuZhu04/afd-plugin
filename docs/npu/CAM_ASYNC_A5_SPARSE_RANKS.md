# A5 async CAM：dispatch send 稀疏 rank 管理

2026-10-13；分支 `GuangyuZhu04/a5-async-cam-sync-sparse`，第二项从 [CR 同步精简提交](https://github.com/GuangyuZhu04/afd-plugin/commit/730ec5d7a06e0ba8ece5aa7c49229ed5434c4f1f) 独立追加。已有分支和工作树保持不变；第一项的 kernel/model 文件在本次提交没有再修改。

## 实现与预算

[DS payload 循环](../../csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h)原来每 token 清零并扫描全部 M 个 MoE rank。本轮在 A5 使用 [TokenRankTracker](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_sparse_ranks.h)：`BeginToken` 只重置 uint64 位图和访问数量；首次出现的 rank 写入访问列表，同 token 后续重复 rank 仅查位图。token 结束后只遍历实际访问的 U 个 rank，更新其 wire offset 一次，U≤min(K,M)。

访问列表直接复用原 `tokenSentFlagTensor_` 的 uint8[M] UB 区域，不新增 TBuf、UB 槽或 host 最低预算。仅增加 Scalar 状态；没有新的 mutable global state、运行时配置或 operator key。Init、SIMT reserve、host/harness/window 资源检查保持。

现有 A5 host 要求 A、M 为正、world=A+M 且 world≤64，因此合法生产 M≤63、rank 位索引为0…62。共享 helper 在移位前检查 rankCount/rank 边界，域校验不增加新的接受形状。A3 和未进入该 A5 域的原循环仍保留。

去重和网络字节数保持原行为：仍在首次命中 rank 时发送 payload，每 assignment 都输出地址项。wire index 在本 token 的所有 assignment 期间保持不变，只有 token 末尾更新；expert occurrence 仍逐 assignment 增加。一次 token 输入读取/量化、INT8 wire scale、16-entry 地址表 flush、PreProcess/SIMT metadata 及完成 flags 都保持原行为。

预期收益来自减少遍历未访问 rank 的 Scalar/UB 工作；小 M 或稠密访问时位图/list 开销可能抵消收益，需要实卡对照。没有声称新增 payload 去重、减少网络流量或测得吞吐提升。

## 独立路由 oracle

[16 项测试](../../tests/unit/package/test_cam_async_sparse_ranks.py)通过 C++17 编译真实共享 tracker。oracle 从原 expert ID 路由独立建立每 rank 的唯一 token 序列、每 expert 的 assignment 序列和 prefix；逐项比较首次 payload 的顺序/bytes、全部 assignment 地址 bytes、rank/expert 最终 offsets、owned slice 的前序 seed、未使用区域，以及输入只 materialize 一次/token。

覆盖单 rank、rank62、完整63-rank域、K>M、非零 owned 起点、空 slice、跨 token 重置和 stale list 尾部；uint16 边界在同 token 两次命中时保持65534，token末恰一次推进到65535。portable helper 的 bit63 测试只检查 uint64 边界，不将 M64 纳入生产域。非法配置/rank/slice 检查是 helper/oracle 边界测试，不构成原设备输入校验资格。

两个故障负例检出首次 payload 后提前更新 offset，以及下一 token 未重置位图。检查 list 的读写次数和 counter 更新次数只与真实 visited owners 相关。量化 wire 是 opaque bytes/4B scale fixture；测试验证原样传递，不重新定义或模拟量化算术、DMA 完成和 NPU 时间。

```bash
python -m pytest -o addopts='' -q -ra \
  tests/unit/package/test_cam_async_sparse_ranks.py
```

## 远程构建与隔离

现场为 aarch64、CANN9.0.0、BiSheng clang15.0.5（2026-02-12）、A5 宏3101/dav-c310-vec；Python3.11.15、torch2.8.0+cpu、torch-npu2.8.0.post4。服务器无 NPU 卡。

第二项构建快照使用第一项正式提交源码树，覆盖 DS kernel/helper/probe/pytest 四文件；Linux 快照的四文件和 Shell 使用 LF。Windows `core.autocrlf=true` 归档会输出 CRLF，因此 Git 源码与编译输入按 LF 规范化核对；只有行尾差异，无代码内容差异。A5 构建 A2E/E2A 与四 async；A3 在独立副本仅选择这六算子，不修改仓库 registry，不包含本轮无关 GMM。

| 第二项检查 | 最终结果 |
| --- | --- |
| A5 ACLNN / 扩展 | 六算子完整编译、打包与扩展构建通过，真实 dispatcher/Meta 加载回归通过 |
| A3 targeted | 六算子独立构建通过 |
| 完整回归 | **226 passed、46 subtests passed，无 skip**；含第一项七个新增场景和本项16项 |
| native ELF keys | readelf 核对 DS=100/101/110/111、CR=100/101/102/103，集合保持 |
| 静态 / dry-run | Ruff check/format、Markdown、diff whitespace，以及 precision/performance/SIMT 候选预检通过 |
| 输入指纹 | 429 个编译/测试源码文件与最终快照一致；文档在验证后补齐 |

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
AFD_BUILD_JOBS=4 bash csrc/npu/build_aclnn.sh "$PWD" 950
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 AFD_SKIP_ACLNN_BUILD=1 \
  MAX_JOBS=2 python setup.py build_ext --inplace
```

本地工作树为 `.cache/a5-sync-sparse-worktree/`，证据在原仓库忽略目录 `.cache/a5-sync-sparse-validation/`。远端为 `/home/zhuguangyu/afd_async_a5_sync_sparse_20261013_Gx25yu`：第一项产物保存于 `stage1_artifacts/`，第二项于 `stage2_artifacts/`；日志带 `sync_stage1`/`sync_stage2` 标记，两个提交分别保留输入与产物指纹，二进制和日志不提交 Git。

第二项日志为 `aclnn_a5_sync_stage2_build.log`、`aclnn_a3_sync_stage2_regression.log`、`binding_a5_sync_stage2_build.log`、`sync_stage2_checks.log/xml`。本地 `stage2/artifact_manifest_stage2.json`、`stage2/source_verification.json`、`stage2/git_tree_verification.json` 与 `stage2/package_keys_stage2.json` 保存证据；扩展隔离构建版本为 `0.0.0+a5.sync.stage2.730ec5d`，未更换系统 CANN/Python。

两项验证及证据收集结束后，临时启动的 `cam_guangyu_dev_90` 已恢复为原 `exited` 状态。

| 第二项产物 | 字节 / SHA256 |
| --- | --- |
| AFD_ascend950.run | 795801；bcc2e910e2de139e2a9b45075943364617ca99d62ea4a1fb571863f63af1a9b3 |
| A3 targeted 包 | 644403；ea164b3ab40fa2a766db644ead891194da0c18ec45abcf9749c8d22fdd5d727e |
| _C_ascend.cpython-311-aarch64-linux-gnu.so | 399352；6f2d9de039a04edd0f0ea5eb25c56b11735fc114582a8927a767a70d2fc242e7 |

## 真卡对照与后续

先以第一项提交为对照，通过原 independent golden 精度，再比较 M1/少量/63、不同 K、同 rank 多 expert、稀疏/热点/均衡路由、B1/小B/大B、FP16/BF16、quant0/1、TP1/2/4、多 chunk 和窗口重复复用。协议、ABI、精度 tolerance 均不放宽。

使用独立进程和匹配 vendor，保持输入、拓扑、window、warmup/iterations 和测量边界相同。Profiler 查看 bookkeeping 与通信/量化时间占比，以单算子和完整 cycle 跨 rank 最大延迟决定收益；Scalar 操作次数的 CPU 模型不能当作硬件加速比例。

改动位于 plugin 自有后端，不新增 vLLM runtime patch。实机证据完成后再决定保留范围，并评估向算子来源贡献；新的 world/metadata/SDK 合同需重新审核 mask 与计数边界。
