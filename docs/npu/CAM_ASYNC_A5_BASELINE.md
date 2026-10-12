# A5 routed-only async CAM：B0 实现与编译记录

2026-10-12；父提交 `bf5bd280b7481358fc0df4a8ec5897b12be73802`。

四个异步通信算子的 A5 功能等价基线已实现，并通过完整 CANN 包、PyTorch 扩展和无卡验证。这里是 source/build qualification：服务器没有 NPU 卡，尚未完成 A5 设备 roundtrip，也没有通信性能结论。设计路线见 [迁移设计](../design/npu_async_a5_migration.md)。

## 实现范围

- 新增 [窗口适配](../../csrc/npu/ascend_kernels/utils/op_kernel/cam_async_window.h)：A3 保留 localWindowsIn/remoteRes 树，A5 使用公共 kernel_operator.h 暴露的 AscendC::HcclCombineOpParam.windowsIn；至少64项窗口的编译期检查。compact-v2 的 metadata/ready 仍在 data window 内，不加入 A2E 偏移，也不把业务 flags 移到 windowsOut。
- 四个 kernel 的量化、top-k 累加、payload 去重、输出次序和通知生命周期保持原逻辑；原始 PIPE_V intrinsic 改为 SDK 的 PipeBarrier<PIPE_V>()。CANN9.0 的 kernel_reg.h 对3101有正式架构分支，A3继续原流水同步。四 kernel 的 MathCeil 升为64位，与host一致，修复4GB容量跨度截成0的风险。
- 新增 [host 容量规划](../../csrc/npu/ascend_kernels/utils/op_host/cam_async_host_tiling.h)：四算子共同采用 min(platform_aiv,48)，launch/tiling一致；保留48行统计预留和 DS→CR inclusive per-AIV前缀。A5限制world≤64、hidden的32对齐和UINT16单源batch范围；按buffer别名计算各阶段UB峰值，DR地址slice使用真实active核数，CS至少容纳一行和完成prefix scratch。FFN窗口额外保留尾部32B cursor。
- A5 MC2 tiling选择命名的HcclAccelerator::AIV值3，沿用ACLNN MTE server；这些字段属于不同枚举。现有CMake门禁使A3旧SDK不需要SetCommEngine。没有新增CCU/SIMT数据路径。
- [registry](../../csrc/npu/ascend_kernels/operator_registry.json)与[pybind](../../csrc/npu/pybind/CMakeLists.txt)同时为950启用四算子，A2E/E2A保留。已有A5 A2E/E2A代码和远端已有A5工程没有修改。
- [harness](../../tests/npu/async_cam_common.py)增加A5显式资源声明，active_aiv只能为1…48，A5 world≤64，MoE owner沿用48的保守范围；允许active1/MoE2。这些参数只用于预检，不控制native launch或自动探测硬件。CPU oracle、量化scale形式、精度阈值没有修改。

## 实际验证

| 检查 | 结果与范围 |
| --- | --- |
| SDK窗口类型probe | 公共kernel_operator.h、64项windowsIn、GetHcclContext读取编译通过 |
| A5完整ACLNN包 | 950默认registry的A2E/E2A和四异步算子全部编译、打包、安装到隔离源码的AFD vendor目录 |
| A5 PyTorch扩展 | _C_ascend成功构建；确认从隔离源码目录加载 |
| ACLNN导出/内核配置 | 四算子的8个执行/GetWorkspaceSize符号和ascend950配置均存在 |
| A3回归包 | 另一个隔离目录针对A2E/E2A和四异步算子编译通过；不包含本次未修改的GMM算子 |
| 最终测试 | **115 passed，46 subtests passed，无skip**；包含选择/打包、实际C++ UB边界、harness、性能汇总、独立CPU golden及19项真实dispatcher/Meta检查 |
| A5 precision/performance dry-run | 使用明确声明的48核/196608B示例资源通过；不是A5硬件资源测量或设备执行 |
| 本地补丁/文档 | git diff --check与本地链接检查；编译相关源码与远端快照对应，README之后同步的是文档说明 |

测试命令（在同一源码和已构建扩展环境中）：

```bash
SOC_VERSION=950 python -m pytest -o addopts='' -q -ra \
  tests/unit/package/test_ascend_build_files.py \
  tests/unit/package/test_cam_async_build.py \
  tests/unit/package/test_cam_async_host_tiling.py \
  tests/unit/compat/npu/test_async_cam_harness.py \
  tests/unit/compat/npu/test_async_cam_performance.py \
  tests/unit/compat/npu/test_async_cam_golden.py \
  tests/unit/compat/npu/test_cam_async_ops.py
```

Meta检查真实加载C++扩展、注册和tensor schema，但不执行HCCL组或NPU kernel。CPU golden验证参考模型和合同，不等于设备数值正确性。

## 编译环境与复现

容器 `cam_guangyu_dev_90`，aarch64；`cam_py311_pt28`：Python3.11.15、torch2.8.0+cpu、torch-npu2.8.0.post4、numpy1.24.0。CANN路径 `/usr/local/Ascend/cann-9.0.0`，opp/version.info=9.0.0；BiSheng clang15.0.5（2026-02-12构建），CMake3.28.3。此次实编为旧A5目标dav-c310-vec，宏__NPU_ARCH__=3101；代码也识别3510，但未在其他SDK上资格化。

正常源码checkout中构建：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate cam_py311_pt28
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
AFD_BUILD_JOBS=4 bash csrc/npu/build_aclnn.sh "$PWD" 950
SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 AFD_SKIP_ACLNN_BUILD=1 MAX_JOBS=2 \
  python setup.py build_ext --inplace
```

远端验证使用无.git的源码快照，因此只在扩展构建进程设置 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_AFD_PLUGIN=0.0.0+a5.baseline.bf5bd28`，未修改项目版本或系统安装。

隔离工作区：`/home/zhuguangyu/afd_async_a5_baseline_20261012_1VYtw4`。A5在source/下，A3针对性回归在a3_validation/下。CANN vendor只安装到隔离源码的afd_plugin/_cann_ops_custom；没有安装到系统或用户已有工程。源码、包、扩展和日志保留在该目录。临时启动的容器已恢复为原来的exited状态。

## 产物和日志

已取回到本地忽略目录 `.cache/a5-baseline-validation/`：

| 产物 | 大小 / SHA256 |
| --- | --- |
| `AFD_ascend950.run` | 653516B；3c88b5aac51b7e60cbff3da022f6a4bd0250367ea9dafc4f967ee1580368ea63 |
| `_C_ascend.cpython-311-aarch64-linux-gnu.so` | 399352B；4906bb093c89818f8dac85f4546739818cfe24c30119e9f3af76370b6055ae56 |
| A3回归包（远端） | 31794a5a3458a3cb9a0d56dc22f9d2babd7ba3ed8415cfe32dc9f2f3b8fd77b0 |

安装包与扩展分别提供CANN vendor和Python注册，需要与同一源码/环境配套使用；这些arm64产物不能在本地Windows执行。

`最终测试日志`、`JUnit结果`、`A5包编译日志`、`扩展构建日志`、`A3回归日志`、`产物与源码指纹`。这些文件位于本地 `.cache/a5-baseline-validation/`，未纳入Git；对应日志也保留在上述远端隔离工作区。

## 仍需A5设备验收

1. 确认所选MTE server返回匹配窗口资源，group实际buffer大小与HCCL_BUFFSIZE一致；ready/chunk flags和尾部cursor初始状态正确。静态类型probe不能证明这些运行时条件。
2. 跑四阶段roundtrip：FP16/BF16、quant0/1、TP1/2/4、多DP、多MoE、同rank多专家命中、空专家/空MoE、多专家chunk和重复窗口复用。按现有independent CPU golden验路由、payload和scale；combine保持slot顺序与现有容差。
3. 延迟注入、窗口复用、polling活性、remote visibility与输入/scratch保活；合法调用仍要求完整DS→DR→CS→CR生命周期。wire没有epoch/credit，多批并发不是本次合同。

设备expert IDs必须有效，combine batch_info必须来自对应dispatch recv；单源occurrence/index需在UINT16范围，每个TP聚合完整专家需fit receive容量。host不能读取设备路由内容来证明这些值域。零行/非有限量化输入、重复expert IDs、任意伪造metadata、ACL graph/native DBO继续作为单独资格项。

CCU、SIMT和性能对比留在后续阶段；这版B0用作A5同平台对照。
