/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncCombineRecv kernel function definition file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncCombineRecv operator definition file
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "afd_async_combine_recv.h"
#include "afd_async_combine_recv_tiling.h"

using namespace AscendC;
using namespace MoeDistributeCombineRecvImpl;
using namespace Cam;
// Patch reason: full-profile calls require lease boundaries around the data kernel.
// Patch functionality: acquire the frame before access and publish completion after drain.
// Signature: unchanged; legacy calls retain the original data path.
extern "C" __global__ __aicore__ void afd_async_combine_recv(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR commArgs,
    GM_ADDR xOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AfdAsyncCombineRecvTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", AfdAsyncCombineRecvTilingData);
    TPipe pipe;
    int32_t isCamComm = 1;
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncCombineRecvTilingData, tilingData, tilingGM);

    // ### PATCH START: Full-feature slot acquisition precedes all data access
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    const uint64_t fullCall = tilingData.moeDistributeCombineInfo.magic;
    if (!AsyncA5::BeforeCall(pipe, commArgs, fullCall,
            tilingData.moeDistributeCombineInfo.totalUbSize, tilingData.moeDistributeCombineInfo.aivNum)) return;
#endif
    // ### PATCH END: Full-feature slot acquisition precedes all data access
    if (TILING_KEY_IS(100)) {
        AfdAsyncCombineRecv<bfloat16_t, false> op;
        op.Init(expandX, expertIds, expertScales, xOut, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        AfdAsyncCombineRecv<float16_t, false> op;
        op.Init(expandX, expertIds, expertScales, xOut, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        AfdAsyncCombineRecv<bfloat16_t, true> op;
        op.Init(expandX, expertIds, expertScales, xOut, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
        AfdAsyncCombineRecv<float16_t, true> op;
        op.Init(expandX, expertIds, expertScales, xOut, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
        op.Process();
    }
    // ### PATCH START: Publish completion only after the original kernel drains
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    AsyncA5::AfterCall(pipe, commArgs, fullCall);
#endif
    // ### PATCH END: Publish completion only after the original kernel drains
}
