/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncDispatchRecv kernel function definition file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncDispatchRecv operator definition file
 */
#include "kernel_operator.h"
#include "afd_async_dispatch_recv.h"
#include "afd_async_dispatch_recv_tiling.h"

using namespace AscendC;
using namespace MoeDistributeDispatchImpl;
using namespace Cam;

// Patch reason: full-profile calls require lease boundaries around the data kernel.
// Patch functionality: acquire the frame before access and publish completion after drain.
// Signature: unchanged; legacy calls retain the original data path.
extern "C" __global__ __aicore__ void afd_async_dispatch_recv(
    GM_ADDR x, GM_ADDR commArgs,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR batchInfoOut, GM_ADDR epRecvCountRoutedOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AfdAsyncDispatchRecvTilingData);
    TPipe pipe;
    int32_t isCamComm = 1;
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncDispatchRecvTilingData, tilingData, tilingGM);

    // ### PATCH START: Full-feature slot acquisition precedes all data access
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    const uint64_t fullCall = tilingData.moeDistributeDispatchInfo.magic;
    if (!AsyncA5::BeforeCall(pipe, commArgs, fullCall,
            tilingData.moeDistributeDispatchInfo.totalUbSize, tilingData.moeDistributeDispatchInfo.aivNum)) return;
#endif
    // ### PATCH END: Full-feature slot acquisition precedes all data access

    int dynamicQuant = tilingData.moeDistributeDispatchInfo.dynamicQuant;

    if (TILING_KEY_IS(100)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchRecv<bfloat16_t, bfloat16_t, false> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchRecv<bfloat16_t, int8_t, true> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    } else if (TILING_KEY_IS(101)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchRecv<float16_t, float16_t, false> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchRecv<float16_t, int8_t, true> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    }
    // ### PATCH START: Publish completion only after the original kernel drains
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    AsyncA5::AfterCall(pipe, commArgs, fullCall, batchInfoOut);
#endif
    // ### PATCH END: Publish completion only after the original kernel drains
}
