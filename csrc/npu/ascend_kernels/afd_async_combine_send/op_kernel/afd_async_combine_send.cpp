/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncCombineSend kernel function definition file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncCombineSend operator definition file
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "afd_async_combine_send.h"
#include "afd_async_combine_send_tiling.h"
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
#include "cam_async_a5_prepare.h"
#endif

using namespace AscendC;
using namespace MoeDistributeCombineSendImpl;
using namespace Cam;
// Patch reason: full-profile calls require lease boundaries around the data kernel.
// Patch functionality: acquire the frame before access and publish completion after drain.
// Signature: unchanged; legacy calls retain the original data path.
extern "C" __global__ __aicore__ void afd_async_combine_send(
    GM_ADDR expandX, GM_ADDR commArgs, GM_ADDR batchInfo,
    GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AfdAsyncCombineSendTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", AfdAsyncCombineSendTilingData);
    TPipe pipe;
    int32_t isCamComm = 1;
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncCombineSendTilingData, tilingData, tilingGM);

    // ### PATCH START: Full-feature slot acquisition precedes all data access
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    const uint64_t fullCall = tilingData.moeDistributeCombineInfo.magic;
    if (!AsyncA5::BeforeCall(pipe, commArgs, fullCall,
            tilingData.moeDistributeCombineInfo.totalUbSize, tilingData.moeDistributeCombineInfo.aivNum)) return;
#endif
    // ### PATCH END: Full-feature slot acquisition precedes all data access
    // ### PATCH START: Device planning and stream-ordered publication are separate launches
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    if (TILING_KEY_IS(AFD_ASYNC_COMBINE_PREPARE_SCALAR_KEY) || TILING_KEY_IS(AFD_ASYNC_COMBINE_PREPARE_SIMT_KEY)) {
        AsyncA5::PrepareCombine(pipe, commArgs, batchInfo, fullCall);
        return;
    }
    if (AsyncA5::IsFullCall(fullCall) && AsyncA5::CallMode(fullCall) == AsyncA5::Mode::PublishCombine) {
        AsyncA5::AfterCall(pipe, commArgs, fullCall, batchInfo);
        return;
    }
#endif
    // ### PATCH END: Device planning and stream-ordered publication are separate launches
    if (TILING_KEY_IS(100)) {
        AfdAsyncCombineSend<bfloat16_t> op;
        op.Init(expandX, workspaceGM, &pipe, &tilingData, commArgs, batchInfo, isCamComm);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        AfdAsyncCombineSend<float16_t> op;
        op.Init(expandX, workspaceGM, &pipe, &tilingData, commArgs, batchInfo, isCamComm);
        op.Process();
    }
    // ### PATCH START: Publish completion only after the original kernel drains
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    AsyncA5::AfterCall(pipe, commArgs, fullCall, batchInfo);
#endif
    // ### PATCH END: Publish completion only after the original kernel drains
}
