# 507035 原始测试日志与 Kernel Dump

本目录是 [Ascend 950 UMDK DSV4-Flash 接入测试报告](../../ascend950_umdk_dsv4_flash_test_report.md)
的原始证据集合。

- 测试时间：2026-11-26
- 被测分支：`codex/ascend950-umdk-dsv4-flash` @ `cbd8e2a`
- 环境：a5-111，8×Ascend950DT (9582)，CANN 9.2.1

---

## ⚠️ 关于证据完整性的说明

请先阅读本节，它决定了这些文件能证明什么、不能证明什么。

| 证据类型 | 状态 | 说明 |
| --- | --- | --- |
| E2E 服务端日志（attention/ffn/e2e） | ✅ **本次完整保留** | 含全部 507035 原始报错栈 |
| 507035 提取片段 | ✅ **本次** | 由上述日志机械提取 |
| 最小复现探针日志 | ✅ **本次** | rt10/rt11/cam_launch 等 |
| **ACL plog（`/root/ascend/log/debug/plog`）** | ❌ **已丢失** | 原文件在测试容器内，容器被集群回收后随之删除 |
| **本次 kernel dump（`extra-info/data-dump`）** | ❌ **已丢失** | 同上，落在容器内 `/tmp/.../extra-info/` |
| 历史 kernel dump | ⚠️ **非本次** | 见下方 `2026-11-20_kernel_dump_historical/` |

**为什么 plog 和 dump 会丢**：测试期间宿主机存在周期性 `docker prune`
与 NPU lease 抢占（详见报告 §8），运行容器被回收时其可写层一并删除，
而 ACL 的 plog 与 kernel dump 默认写入容器内路径。

**如何避免**：在启动 `vllm serve` 前设置
```bash
export ASCEND_PROCESS_LOG_PATH=/work/plog      # /work 为持久挂载
export ASCEND_DUMP_PATH=/work/extra-info
```
（或把 `/root/ascend/log` 与工作目录挂载到宿主持久盘。）

---

## 目录结构

```
2026-11-26_e2e_507035/          ← 本次 E2E 原始日志（完整）
├── attention.log               ← Attention 角色服务日志（553 KB，2819 行）
├── ffn.log                     ← FFN 角色服务日志（674 KB，8293 行）
├── e2e.log                     ← E2E 驱动完整输出（1.3 MB，11136 行）
├── run.json                    ← 用例执行记录（execution/cleanup/acceptance）
├── 507035_occurrences.txt      ← 507035 出现的所有行号与行内容
├── 507035_context.txt          ← 507035 前后各若干行的上下文
└── aivec_error_context.txt     ← AIV 内核错误（aivec error）上下文

2026-11-26_probes/              ← 最小复现探针日志
├── rt10.log                    ← task 字段未正确打包（BAD_ABI）
├── rt11.log                    ← 修正打包后，故障前移到内核数据面
├── cam_launch.log              ← 插件真实形状下的 AFDBuffer 构造
├── jit_txn.log                 ← DeepEP JIT 路径（enable_local_rounds 对照）
└── min_txn.log                 ← 最小 transaction 路径初始化

2026-11-20_kernel_dump_historical/   ← ⚠️ 历史 dump，非本次
└── data-dump/{0,1,2,3,...}/
    ├── AfdAsyncCombineRecv_d2a9f391...json    ← kernel 元数据
    ├── AfdAsyncCombineRecv_d2a9f391....o      ← 设备 ELF
    ├── ..._100_host.o
    └── exception_info.61.2897.202611201346...  ← 异常现场（AIC_INFO 寄存器/参数）
```

### 关于历史 kernel dump

这批 dump 来自 **2026-11-20 的另一次测试**，不是本报告 §6 结论所依据的那一次。
使用时请注意：

| 项 | 本次 (11-26) | 历史 dump (11-20) |
| --- | --- | --- |
| 触发内核 | `AfdAsyncCombineRecv_91445775f2d9669ef191a49c071d5d12`<br>`AfdAsyncDispatchSend_21c646510794c58ff569a30a2208ad7f` | `AfdAsyncCombineRecv_d2a9f391bf9a17f49b4399f97f3a3794` |
| 内核 ELF 哈希 | — （dump 已丢失） | `d2a9f391...` |

内核 ELF 哈希不同，说明**不是同一份编译产物**（可能来自不同 OPP 构建）。
因此这批 dump **只能作为"该算子族在 A5 上确实产生过设备异常"的旁证**，
不能直接用作本次失败的根因证据。

不过其中 `exception_info.*` 文件保留了真实的 `[AIC_INFO] args(0 to N)`
寄存器与参数快照，对定位内核参数错位仍有参考价值，故一并保留。

---

## 507035 的关键报错原文

摘录自 `attention.log:357`：

```
(Worker_DP1_TP1_EP3 pid=1183) ERROR 11-26 13:52:06 [multiproc_executor.py:1063]
RuntimeError: copy_between_host_and_device_opapi:
../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:94
NPU function error: device error type 3, error code is 507035
```

同文件后随的设备侧展开：

```
[ERROR] PTA call acl api failed
[Error]: The vector core execution is abnormal.
EZ9999: Internal error!
EZ9999[PID: ...]: An error occurs on the device(chipId:0, dieId:0),
  the serial number is 211, the error is aivec error, core id = 0, ...
The extend info: errcode:(95)
  errorStr: The MTE instruction accesses an invalid GM address
            or the cross-device memory access timeout. subErrType: 0x4.
```

即：异常发生在 `afd_async_combine_recv` 之后的
`o.recv_status.view(torch.int32)[:2].tolist()` 处（见 `umdk_session.py:322`），
设备侧是 AIV 内核的 MTE 访问异常。

统计：本次 `attention.log` 中 507035 出现 **23 次**，`e2e.log` 中 **23 次**，
`ffn.log` 中 **0 次**（FFN 的失败是 Attention 崩退后的
`DistNetworkError: Connection reset by peer` 连锁反应）。

---

## 相关文档

- 主报告：[ascend950_umdk_dsv4_flash_test_report.md](../../ascend950_umdk_dsv4_flash_test_report.md)
- 上游已知缺陷记录：`test/cam/benchmark_afd_async.py`（UMDK `dcc0350e`）
  > Issue #6 reports CAM combine_send error 507035 for K=6/E=256.
