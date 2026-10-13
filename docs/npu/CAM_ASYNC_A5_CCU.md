# A5 async CAM：P2 CCU 连续段实验实现

2026-10-13；从P1提交 `45f86a1d9b1f647809fd11d8ca2fab2d40798d71` 派生，分支 `GuangyuZhu04/a5-async-cam-ccu-combine`。这是设计路线第四步的可编译CCU候选，不是已经启用的生产通信后端。原四算子、PyTorch ABI、compact-v2协议以及P0/P1路径保持原代码。

本轮实现设备端描述符规划、使用真实SDK的custom CCU指令编排类和host注册/提交对象。源码编排CCU读取设备GM动态offset/length，并对单peer发出 `WriteNb → WaitEvent`，无需每chunk D2H读取counts或host polling。服务器没有NPU卡，尚未运行通信任务；候选不发布compact-v2 ready，也没有被原combine send自动选择。

## 实现与布局

源码：[独立模块](../../csrc/npu/experimental/cam_async_ccu/)、[checked planner](../../csrc/npu/experimental/cam_async_ccu/combine_segments.h)、[AIV生产者](../../csrc/npu/experimental/cam_async_ccu/prepare_segments.cpp)、[CCU指令编排类](../../csrc/npu/experimental/cam_async_ccu/ccu_segment_kernel.cpp)、[提交对象](../../csrc/npu/experimental/cam_async_ccu/ccu_submission.h)。

| 部件 | 本轮行为 |
| --- | --- |
| CombineSegmentCursor | 完整预检后按expert→TP遍历，给出source/destination byte offset与长度；默认一个AIV规划整chunk，也可验证原AIV row交集 |
| SendSegment | 32B对齐，四UINT64字段srcByteOffset/dstByteOffset/byteCount/dstRank，偏移0/8/16/24；固定GM槽为TP-major的 `tp*R+expert` |
| prepare_segments | 单AIV、64KiB metadata UB上限；读取完整batch_info，预检fixed TP组/容量，清零全部R*T槽，再写非空段，最后写8B规划状态 |
| SegmentKernel | host只提供descriptorGM/sourceBase/sourceToken/destinationBase/destinationToken/statusGM六项已知资源；Algorithm源码编排CCU读取状态和描述符，尚未实际注册/Translate硬件指令 |
| 双重目标校验 | producer验证metadata的TP组等于固定channel配置；CCU验证描述符dstRank等于注册kernel的固定peer，避免错槽写到另一peer |
| SegmentSubmission | instance拥有creator/arg/handle，调用真实HcclCcuKernelRegister/RegisterFinish/Launch；无新mutable global，无隐式注册内存或stream同步 |
| ReadyNotification | planner仅描述原prefix+1的最后chunk通知，包括空MoE；本候选不执行通知写入 |

令每wire row为 `rowBytes=2H`。当前chunk source顺序为expert→TP→token；每peer destination按其此前专家count累计：

`dstByteOffset = align32(4*(M+M*R)*48) + align32(4*M) + rowBytes*(moePrefix[tp]+expertPrefix[tp])`。

planner对完整chunk与窗口容量使用checked UINT64运算，拒绝非法metadata、负count、目标TP组不匹配、源/窗口/固定槽容量不足及乘加溢出。原CS和CR均未改动。64KiB只是候选producer上限，不提高原host/harness最低UB要求；未来接入生产时超限需保留MTE fallback。

规划成功且peer匹配、length非零时，CCU才写payload。非零错误状态阻止消费旧descriptor，但状态不是epoch或producer/consumer同步：消费者必须先取得本轮producer完成，否则仍可能读到上轮成功状态。窗口仍沿compact-v2串行复用，不能当多批in-flight协议。

## SDK入口选择与限制

现场为CANN9.0.0、BiSheng clang15.0.5（2026-02-12）、A5宏3101/dav-c310-vec。公开新版资料不能代替这个安装包的具体合同。

- CCU Hccl完整specialization没有BatchWrite/Send/Recv实现。预期失败探针直接实例化BatchWrite，检查失败确为缺少该member，而不是任意include/compiler错误。已有工程中的API调用文本不能证明它可用。
- AlltoAllvWrite存在真实device实现，使用HCCL_CMD_HALF_ALLTOALLV。它每peer提供source offset/size，但仅有一个共有remoteWinOffset；原CS各peer的moePrefix/当前expert前缀可能不同，一次调用无法直接表达全部目标区间。
- 该device Hccl CCU客户端handle上限63，逐segment发sparse-array prepare不能假设任意R*T均可容纳；接口repeat固定1。现场public header描述rankDim和byte数组，而最新master示例另有2*rankDim双die布局；不能混用。域参与、数组布局/单位及server notify仍需匹配算法/实机证据，不能把‘Half’名称当作独立peer保证。
- Custom CCU公开LoadVariable可在运行时读取GM，WriteNb/WaitEvent及Register/RegisterFinish/Launch由libhcomm真实导出，因此本轮选择独立custom候选。Launch仍是host C API，不冒称AIV直接调用了host Launch。

## 编译与验证

使用匹配的CANN安装，独立构建不会安装或替换当前vendor：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash csrc/npu/experimental/cam_async_ccu/build.sh \
  /usr/local/Ascend/cann-9.0.0 "$PWD/.cache/ccu-build"
```

本轮编译/链接的是host侧CCU指令编排类，未实际调用HcclComm注册、Translate生成硬件CCU指令或执行任务。CMake直接链接真实libhcomm，启用 `--no-undefined`，采用现场库的 `_GLIBCXX_USE_CXX11_ABI=0`；不使用SDK stub、不改SDK头文件。产物包括host共享库、AIV producer对象和device Hccl客户端能力探针。构建脚本检查BatchWrite的预期失败诊断，不启动通信。

| 检查 | 结果 |
| --- | --- |
| CCU宿主共享库 | 真实SDK编译、libhcomm链接通过；--no-undefined、ldd -r及共享库实际加载通过，未实际注册或启动通信任务 |
| A5 descriptor producer | BiSheng实际编译通过，共享checked planner进入设备对象 |
| device Hccl CCU positive probe | 真实GM tiling完整加载，InitV2/SetCcTilingV2/AlltoAllvWrite/Commit/Wait/Finalize编译通过 |
| A5原算子包、扩展 | A2E/E2A与四async完整包、扩展构建和实际Meta加载通过 |
| A3针对性回归 | 在独立副本构建A2E/E2A和四async通过，不包含本轮未改的GMM |
| 测试 | **148 passed、46 subtests passed，无skip**，含新增8项CCU planner的真实C++编译/执行、原CPU golden、真实Meta和P0/P1测试 |

[8项planner测试](../../tests/unit/package/test_cam_async_ccu.py)使用独立raw-routing/token byte-scatter oracle，比对所有目标区间及未写window guard，覆盖partial/last chunk、空final/nonfinal、3/48 AIV交集、固定目标组、UINT16/容量/metadata负例、超过4GiB的目标offset及UINT64溢出。它执行共享整数算法，不模拟CCU设备或证明CCU完成语义。

复现追加测试：

```bash
python -m pytest -o addopts='' -q -ra tests/unit/package/test_cam_async_ccu.py
```

本地新worktree为 `C:/D_disk/Github/afd-plugin/.cache/a5-ccu-worktree`。远端新目录 `/home/zhuguangyu/afd_async_a5_ccu_combine_20261013_XWYh7e`；A5快照为source/，A3副本为a3_validation/，实验产物为ccu_build/及顶层producer/client对象。既有A5工程和前三阶段目录未覆盖。完整日志与指纹在原工作区忽略目录 `.cache/a5-p2-validation/`，不纳入Git。

仓库内build.sh在独立ccu_repro/目录复现通过；negative probe明确确认缺少BatchWrite member。Ruff check/format、Markdown及diff whitespace通过。205个csrc/npu及tests/unit/package快照文件与本地输入指纹一致；Shell脚本仅在Linux快照统一LF。

| 产物 | 大小 / SHA256 |
| --- | --- |
| AFD_ascend950.run | 762694B；a60f9b4e01fc8ff340626fafefddfed2ed3c1d30be1d24408a6631345e86d9a7 |
| _C_ascend.cpython-311-aarch64-linux-gnu.so | 399352B；48ff3d07d6f029894a7cebbdb80e790033b9838efd93cf86a58fd18e40c482dc |
| libafd_cam_async_ccu_experimental.so | 80328B；9af359d6d7c4cc0fc1bfe6a98df2b55cc26e174fd0fd7e18f3090f83ab7d78e6 |
| prepare_segments.o | 106216B；6fc3bde7358545782eef9451a52eec21aa9ff680158c8c70c433b8463477dcef |
| hccl_ccu_client.o | 80816B；5266c6a979c28458fd0762a259ac805d1b28112e9f8043e7c04c596bf8ed6fe3 |
| A3回归包（远端） | 22670f8d6fddedef298c91a58f102ca90a96fb124f9b1300dbbe984e61b57fe0 |

日志为aclnn_a5_p2_build.log、aclnn_a3_p2_regression.log、binding_a5_p2_build.log、p2_checks.log/xml、ccu_host_build.log、ccu_producer_build.log、ccu_client_build.log、ccu_batch_write_negative.log、ccu_link_check.log和ccu_reproduce.log；artifact_manifest.json保存指纹。临时启动的容器完成后恢复为exited状态。

## 接回原四算子的资格条件

1. 明确真实registered source/destination span与memory token来源、channel对应peer、window基址/大小，并与原compact-v2窗口核对。RegisteredSpan的非零检查不证明实际注册；不能把任意Torch tensor或原MTE context直接当CCU资源。
2. 证明当前Torch stream上producer→CCU task的依赖和任务完成回到同一stream的边界。现场HcclThreadAcquireWithStream只注释支持CPU_TS；不能假设CCU thread直接绑定stream，不能把正在等CCU的kernel后面再排CCU task。
3. 实卡确定WriteNb/WaitEvent涵盖源消费、本端完成和远端HBM可见的哪些边界，再把全部段完成接回原最后ready。不以Launch成功、8B规划状态或函数名充当远端完成证明。
4. creator、kernel args、task args、registered spans、tensor及descriptor/status GM须存活且内容不可改写或复用，直到被证明的消费完成（下一prepare/GMM也不能覆盖同一缓冲区）；注册finish仅用于setup，不能增加每chunk全域barrier。
5. 对空peer、多chunk、TP1/2/4、稀疏/热点、错误目标组、窗口复用和延迟注入做精度与活性测试，通过后才能加入显式CCU选路与MTE fallback。

性能对照分别运行B0/P0/P1/P2，核对route/packing/submit/transfer/ready等待、AIV占用及整轮跨rank延迟。CCU可能省去payload UB中转，也可能因注册资源、单段任务数、staging和完成交接增加开销；本轮没有实机加速结论。

资料：[设计P2](../design/npu_async_a5_migration.md#43-p2ccu-连续区间传输优先-combine-send)、[官方BatchWrite支持范围](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/hccl_communication/HCCL-Kernel/BatchWrite.md)、[AlltoAllvWrite说明](https://asc.gitcode.com/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/AlltoAllvWrite.html)。本轮判断以保存的9.0 SDK声明、实现、符号与实际编译结果为准。
