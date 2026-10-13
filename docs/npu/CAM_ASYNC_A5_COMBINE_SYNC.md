# A5 async CAM：combine recv 向量到标量同步精简

2026-10-13；从接收优化提交 `52504ac3ce1eace526bd1f5148f5045fe7efaa25` 派生到 `GuangyuZhu04/a5-async-cam-sync-sparse`。本轮第一项只修改 A5 CR 预取路径，独立于下一项 dispatch send 稀疏 rank 管理提交。已有分支和工作树保持不变。

## 改动与依赖

[CombineRecvPipelined](../../csrc/npu/ascend_kernels/afd_async_combine_recv/op_kernel/afd_async_combine_recv.h)中，`Duplicate` 和 `Add` 的 FP32 scratch 没有 Scalar 消费者，因此移除两处显式 `V_S` 等待。每个当前 tile 的 weight 在安排下一 tile 前以值保存为 `currentScale`；地址、长度和输出位置仍由当前 cursor 快照给出。

保留 `Cast → PIPE_V → Muls → PIPE_V → Add`。下一次 Muls 前的 PIPE_V 也完成此前的 Add 和当前 Duplicate，独立 hidden 区间的 scratch 不需要 Scalar 观察结果。最后 K 仍在 PIPE_V 后 `CAST_RINT`，保留 `V_MTE3` 输出 ready、`MTE3_V` 槽位保护和 `V_MTE2` free。尾部 drain 闭合 V→MTE2→Scalar 后才释放 event、进入原 `SyncAll` 和重置远端 flags。

A3、原串行 fallback、GetIds 分页、PreProcess 的真实 Scalar 读取及其等待保持原行为。operator keys、compact-v2、ABI、最低 UB、三份 full-H FP32 scratch 和原 K 顺序均保持。没有 FMA、额外运行时配置或可变共享状态。

预期收益来自 Scalar 可以更早安排后续 DMA/向量命令；编译器及向量 API 内部可能仍有同步成本，实际收益需要真卡 profiling。未测量或承诺等待指令数、延迟和带宽下降。

## 回归模型

[接收模型测试](../../tests/unit/package/test_cam_async_receive_optimization.py)增加七项：FP16、BF16、partial H、K8 四个异步正例，以及 live scale、live address、Scalar 提前改写 product 三个故障负例。每个场景尝试 64 种 MTE2/V/MTE3 FIFO 交错。

新模型使用与生产相同的三个共享 full-H FP32 scratch，分别排队 Cast、Muls、Add 和最终 Cast，不把整段计算当作不可分割操作。按相同 seed 对照原逐 tile Scalar 等待模型与新模型；正例均使用原输入导出的独立 FP32 分步舍入/FP16或BF16最终舍入 golden。模型确认无逐 tile Scalar fence、issue-time 参数捕获，以及 Scalar 可在 V 工作未完成时继续发出命令。故障负例证明如果异步命令晚读可变参数，或 Scalar 改写 live scratch，模型会检出错误。

模型执行共享生产 Plan/Cursor/PipelineEvents；向量算术通过 CPU backend 模拟，不能替代 NPU 执行或通信完成证据。临时 SDK header 只声明模型解析所需类型。实际 CANN 编译单独验证接口和完整算子包。

## 编译与证据范围

现场为 aarch64、CANN9.0.0、BiSheng clang15.0.5（2026-02-12），实际 A5 宏3101/dav-c310-vec；Python3.11.15、torch2.8.0+cpu、torch-npu2.8.0.post4。服务器无 NPU。

第一项编译快照为父提交全部文件加三个 CR/kernel/model 文件，DS 文件指纹仍等于父提交；第二项未混入本次构建。

| 第一项检查 | 最终结果 |
| --- | --- |
| A5 ACLNN / 扩展 | A2E/E2A 与四 async 完整构建、扩展编译及真实 dispatcher/Meta 加载回归通过 |
| A3 targeted 构建 | 独立副本 A2E/E2A 与四 async 通过；不修改原 registry，不包含无关 GMM |
| 完整回归 | **210 passed、46 subtests passed，无 skip**；含新增七项异步算术模型 |
| dry-run / 静态检查 | precision/performance/SIMT 候选预检及 Ruff check/format、Markdown、diff whitespace 通过 |

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
AFD_BUILD_JOBS=4 bash csrc/npu/build_aclnn.sh "$PWD" 950
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 AFD_SKIP_ACLNN_BUILD=1 \
  MAX_JOBS=2 python setup.py build_ext --inplace
python -m pytest -o addopts='' -q -ra \
  tests/unit/package/test_cam_async_receive_optimization.py
```

额外尝试检查生成代码：安装包 llvm-objdump 无法解码这些 A5 opcode，输出 `<not available>`；BiSheng CCE driver 拒绝通用 `-S`、LLVM 输出、保存临时汇编及 pass dump 选项。本轮没有取得可归因的 ISA/IR 对照，不能据空输出或命令退出码推断实际消除了多少指令。正确性论证使用公开 API 的按值参数、PIPE_V 与事件依赖合同，并由独立模型和真实 SDK 构建分别核对；硬件资格仍待补齐。

证据目录为原仓库忽略目录 `.cache/a5-sync-sparse-validation/`；远端为 `/home/zhuguangyu/afd_async_a5_sync_sparse_20261013_Gx25yu`，第一项源码在 `source/`，A3 针对性回归在 `a3_validation/`。日志与二进制不提交 Git。

426 个编译/测试源码文件的指纹与第一项快照一致，DS 仍为父提交；文档在验证后补齐。第一项产物另存 `stage1_artifacts/`，日志为 `aclnn_a5_sync_stage1_build.log`、`aclnn_a3_sync_stage1_regression.log`、`binding_a5_sync_stage1_build.log`、`sync_stage1_checks.log/xml`，本地 `stage1/artifact_manifest_stage1.json` 与 `stage1/source_verification.json` 保存完整指纹。

| 第一项产物 | 字节 / SHA256 |
| --- | --- |
| AFD_ascend950.run | 789126；c415e2bf941ce9201fd255bf9185e1a6585e47840688656f5388a18ec8b2250e |
| A3 targeted 包 | 643186；294c373796bc0ebb1fb53ce40819253654fc07801f3e33c44b4c0dbf60624762 |
| _C_ascend.cpython-311-aarch64-linux-gnu.so | 399352；6f2d9de039a04edd0f0ea5eb25c56b11735fc114582a8927a767a70d2fc242e7 |

## A5 实机资格

对照父提交，使用独立进程和配套 vendor。运行 FP16/BF16、K1/K2/K8、不同 H、partial tile、prefix/非 prefix、小 batch 奇数核数、多 source/TP/chunk、空 MoE 和窗口重复复用；payload、scale、路由合同及 combine tolerance 不放宽。重点确认 Scalar 提前推进不改变动态地址/weight 捕获，shared scratch 没有跨 tile/token 竞争，output 槽在 MTE3 消费完成前不能复用。

精度通过后，以相同输入、拓扑、window、warmup/iterations 和计时边界对照 per-op 与每轮跨 rank 最大延迟；Profiler 检查 Scalar 等待及 MTE2/V 重叠。无卡测试不能提供加速结论。

改动位于 plugin 自有后端算子，不扩展 vLLM runtime patch。实机资格完成后再决定保留范围，并评估向算子来源贡献；未来 SDK 参数或流水合同变化时重新审查依赖，不能沿用当前证据。
