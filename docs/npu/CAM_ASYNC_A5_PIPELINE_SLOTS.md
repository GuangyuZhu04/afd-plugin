# A5 async CAM：P3 SIMT 描述符与 P4 多槽实验协议

2026-10-13；从P2提交 `d85933657c776ea7563363d8a57a2e71a670557e` 派生，分支为 `GuangyuZhu04/a5-async-cam-simt-ccu-slots`。用户确认本轮同时推进P3和P4。原四算子、compact-v2运行路径、PyTorch ABI和前四阶段分支保持不变；新增内容在独立experimental模块中。

本轮完成可编译的SIMT描述符生产者，以及带独立版本、epoch/credit、abort/drain和按序归约门控的P4参考协议。服务器无NPU卡，没有运行设备通信；CPU完成声明、CCU Launch/WaitEvent或记录校验均不能替代硬件完成证明，新协议没有自动启用。

## P3：稳定描述符与有界UB

入口：[SIMT producer](../../csrc/npu/experimental/cam_async_pipeline/simt_descriptor_producer.cpp)、[VF](../../csrc/npu/experimental/cam_async_pipeline/simt_descriptor_vf.h)、[共享算法](../../csrc/npu/experimental/cam_async_pipeline/stable_descriptors.h)、[预算](../../csrc/npu/experimental/cam_async_pipeline/simt_descriptor_budget.h)。

1. 原P2 checked cursor先完整预检metadata、固定TP目标、source/window及固定槽容量。GM metadata不变。
2. SIMT每lane拥有完整TP行，原位把UB counts改成inclusive prefixes，不另分配第二份64KiB表。随后不能再使用依赖原counts的cursor。
3. SIMT每lane独占 `tp*R+expert` 固定槽，通过TP前缀恢复原expert→TP source顺序及peer目的前缀，不用atomic ticket；空/非当前chunk槽写零。
4. 每256槽形成8KiB tile，通过MTE3连续写回；最后一次写回完成才发布本轮成功状态。

MTE2_S/S_MTE2供scalar预检，MTE2_V/S_V供VF读取；prefix VF后V_S/S_V保证跨TP前缀可见；每tile用V_MTE3→MTE3_S/S_V保护copy和复用。同步代码已实际编译，尚无硬件执行证据。

| UB项目 | 字节 |
| --- | --- |
| metadata / descriptor tile / status | 65536 / 8192 / 32 |
| native arena总量 | 73760 |
| SIMT保守预留 | 139264：128KiB Data Cache文档上界＋8KiB compiler |
| 完整候选最低physical UB | 213024 |
| 仅失败状态路径最低launch预算 | 139296 |

调用者在launch前检查CanLaunchDescriptorProducer；低于完整预算返回独立ResourceUnavailable=11。必须在启动CCU前整批选择原P2候选/MTE路径。新增预算不提高原host/harness最低要求，不能launch连SIMT cache加status都容不下的kernel。预留是文档上界策略，不是实测默认Cache或实际arena边界。

P3仍保留scalar全量预检，每slot恢复source前缀仍有O(TP)读开销。这版证明稳定layout、slot独占和批量描述符写回，尚不证明CCU传输与下一VF/GEMM实际重叠或加速。P2的注册、stream交接、源消费和远端可见桥仍需实卡资格。

## P4：独立版本、credit与完成边界

入口：[参考协议](../../csrc/npu/experimental/cam_async_pipeline/slot_protocol.h)、[独立ring布局](../../csrc/npu/experimental/cam_async_pipeline/ring_layout.h)、[AIV记录codec](../../csrc/npu/experimental/cam_async_pipeline/slot_record_codec.cpp)。新标识为 `AFDSLOT4` / version1 / fingerprint `0xBF8A4DFD12800101`。双方必须明确一致并使用独立registered allocation；不能给compact-v2旧ready加epoch后混用。

每个单producer/consumer域默认2槽、最多8槽。128B控制记录包含magic/version/fingerprint、domain/slot/epoch/request/chunk、state、boundary flags、payload/metadata长度和slotCount。ring预留128B头区，每槽有分开的producer-ready/receiver-credit记录及payload/metadata空间，128B对齐；总span及offset使用checked UINT64运算。P3的compact-v2目的offset不能直接用于P4，须按新layout重新规划。

```mermaid
flowchart LR
  FREE[Free credit] --> WRITE[Writing lease]
  WRITE --> READY[SourceConsumed和RemoteVisible后Ready]
  READY --> READ[Reading]
  READ --> CAPTURE[独立backing捕获完成]
  CAPTURE --> CREDIT[Consumed和ReceiverCredit]
  CREDIT --> RETIRE[TaskRetired无后续目标写]
  RETIRE --> FREE
```

SourceConsumed、RemoteVisible、ReceiverConsumed、ReceiverCredit和TaskRetired分别使用绑定完整lease的外部完成声明，默认均false；真实后端资格通过后才能签发。planning status、valid record、local完成或P2 WaitEvent不能自动提升成RemoteVisible。credit先到而task仍可能写回时，槽保持Consumed；两者均完成后才Free。源及descriptor不可改写直到SourceConsumed，目标slot不可改写直到credit与task退休。

每次Acquire分配单调UINT64 epoch，拒绝错误domain/slot/request/chunk、陈旧/重复ack及回绕；epoch耗尽后停止发放。Abort不清busy槽、不凭超时返credit；恢复需要source/descriptor reads、target writes、receiver reads及owned backing访问全部停止，且domain/last epoch匹配。恢复不重置epoch；进程/对象重建必须换新domain，或恢复同domain持久lastIssuedEpoch，不能以默认0重开同一代。fallback只在launch前整批选择，已提交后的错误须abort/drain，不能覆写同槽或直接跨协议重发。

OrderedCombineGate将传输槽与逻辑top-k序号分开：先ReserveCapture，实际拷贝至caller独立backing，CaptureComplete后才可归还通信credit。只按逻辑0…K−1启动下一项，ReductionComplete确认算术及backing读取结束后才前进。Gate最多64贡献项，是实验profile限制，不缩小原算子范围。caller须先取得当前transport的BeginRead权限，并验证原routing到logical index的映射；gate不代替这两项校验。

## 参考模型与设备codec的界限

BoundedSlotProtocol与OrderedCombineGate是portable single-coordinator协议代码，调用须串行化；没有实现跨rank GM CAS、网络消息、cache acquire、atomic publication或CCU事件绑定。CPU测试声明来自明确的测试ledger，不是设备完成证据。

AIV codec只检查调用者已保有且不可并发改写的本地staging快照，并可复制至另一块staging。128B MTE copy不是atomic snapshot，不自动发布ready或credit；并发wire访问仍需qualified commit/release/acquire机制。codec支持Ascend/aarch64与x86小端对象表示；offset helpers仅接受成功构建且不可改动的layout。

当前CCU Variable仅能与立即数比较，不能假设可直接比较动态expected epoch；lease核验由AIV/参考协议承担。CCU读取descriptor仍要求同一invocation的producer完成交接，不能仅见上轮status0便消费。旧任务实际停止写入或全新allocation是复用安全条件，epoch数值不能取消旧写。

## 验证与复现

现场：CANN9.0.0、BiSheng clang15.0.5（2026-02-12）、A5宏3101/dav-c310-vec；Python3.11.15、torch2.8.0+cpu、torch-npu2.8.0.post4。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash csrc/npu/experimental/cam_async_pipeline/build.sh \
  /usr/local/Ascend/cann-9.0.0 "$PWD/.cache/final-pipeline-build"

python -m pytest -o addopts='' -q -ra \
  tests/unit/package/test_cam_async_pipeline.py \
  tests/unit/package/test_cam_async_slot_protocol.py \
  tests/unit/package/test_cam_async_slot_wire.py
```

脚本独立重建P2 CCU指令编排类并链接真实libhcomm（ABI0、no-undefined），确认BatchWrite缺失member的负诊断，再编译P3/P4设备对象；不替换当前vendor或启动通信。

| 检查 | 最终结果 |
| --- | --- |
| P3设备对象 | 实际SIMT producer编译通过，包含prefix/descriptor两个真实simt_entry：120B/288B |
| P4设备对象 | AIV不可变记录codec实际编译通过 |
| CCU builder | 真实libhcomm链接、ldd -r及加载通过，未实际注册/Translate/执行任务 |
| A5原包、扩展及A3回归 | A2E/E2A＋四async完整编译及Meta加载通过；A3副本不含未改的GMM |
| 最终测试 | **176 passed、46 subtests passed，无skip** |

新增28项分别为P3描述符11项、P4 ownership/order13项和wire layout/codec4项。P3用真实共享helper及独立raw-routing byte-scatter oracle覆盖partial/last/空peer、1/3/8/256槽tile、291槽跨tile、257 lane、超过4GiB offset、window guard及UB边界。P4用独立lease/payload ledger覆盖credit耗尽、迟producer/reader/retirement、陈旧ack、空chunk、epoch耗尽与abort/drain。双槽被高逻辑贡献占满时，实际捕获至独立backing并归还credit，使低序号能够到达；FP32反例按逻辑顺序得0，按到达顺序得1。wire4项仅检查geometry、ABI codec及schema/lease拒绝，不验证routing或硬件同步。

本地worktree：`C:/D_disk/Github/afd-plugin/.cache/a5-final-worktree`。远端新目录：`/home/zhuguangyu/afd_async_a5_final_20261013_cdw1Ky`。原阶段及已有工程未覆盖；本地产物、日志在忽略目录 `.cache/a5-final-validation/`，不提交Git。仓库内build.sh已在final_repro/独立目录复现通过；Ruff check/format、Markdown和diff whitespace通过。219个csrc/npu及tests/unit/package快照文件与本地输入指纹一致；Shell脚本仅在Linux快照统一LF行尾。

| 产物 | 大小 / SHA256 |
| --- | --- |
| AFD_ascend950.run | 762691B；3aabaf9543e3a0012f2dab1aae1e7ddc63193b881525f61da3aa043e00ca2256 |
| _C_ascend.cpython-311-aarch64-linux-gnu.so | 399352B；8a5958d69ca69cde5b39d665d8ef7a442064dd7fad4fedac098e7eb57d70b123 |
| libafd_cam_async_ccu_experimental.so | 80328B；9af359d6d7c4cc0fc1bfe6a98df2b55cc26e174fd0fd7e18f3090f83ab7d78e6 |
| simt_descriptor_producer.o | 131744B；049c076ff2e453484d2391a448acf87eccd5c5998b93ff700c237448d052c51b |
| slot_record_codec.o | 98160B；c400f3bdad5ace995cfff23a3f3c5ca0b31aa1efe01a15ffc41207795d987597 |
| A3回归包（远端） | bd06d692b96f759477a75839640ca907ffa3520cc266d4116c7a5fc7d6d13768 |

日志为aclnn_a5_final_build.log、aclnn_a3_final_regression.log、binding_a5_final_build.log、final_checks.log/xml、p3_producer_build.log、p4_codec_build.log、final_reproduce.log、final_link_check.log及VF/codec符号日志；artifact_manifest.json保存指纹。临时启动的容器完成后恢复为exited。

## A5实机后续资格

分别用独立进程加载B0/P0/P1/P2/P3/P4匹配vendor，保持相同输入和拓扑。先资格化原四算子，再确认registered spans/tokens、CCU thread/stream producer handoff及SourceConsumed/RemoteVisible/TaskRetired和credit消息的完成域。P4额外验证schema协商、persistent epoch、delay/out-of-order/reuse、credit耗尽、异常drain、backing容量及原top-k映射。测量完整cycle跨rank最大延迟、AIV/UB/cache占用、submit/packing/等待与重叠，不能据CPU模型或编译结果宣称加速。
