// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
// Compile-only CANN 9.0 / v310 client probe. This is not a production transport.
// AlltoAllvWrite is a collective API; successful compilation does not establish
// independent-peer participation, compact-v2 window identity or remote visibility.

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"
#include "adv_api/hccl/hccl.h"

#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3101
#error "This probe requires the installed CANN 9.0 dav-c310-vec target (3101)."
#endif

namespace {
constexpr uint32_t CLIENT_CORE_ID = 0;
constexpr uint32_t TILING_STATUS_INDEX = 0;
constexpr uint32_t HANDLE_STATUS_INDEX = 1;
constexpr uint32_t WAIT_STATUS_INDEX = 2;
constexpr uint32_t STATUS_WORD_COUNT = 3;
constexpr AscendC::HcclServerConfig CLIENT_CONFIG = {
    AscendC::CoreType::ON_AIV, CLIENT_CORE_ID};

struct CcuClientProbeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling ccWriteTiling;
    uint64_t remoteWinOffset;
    uint64_t localDataSize;
};
constexpr uint32_t TILING_BYTE_COUNT = sizeof(CcuClientProbeTilingData);
} // namespace

// Caller obligations for a future device execution: provide a matching MC2 CCU
// context/algorithm and complete valid tiling for HCCL_CMD_HALF_ALLTOALLV;
// source and version-matched uint64 offset/size arrays must be visible in GM.
// Only one AIV submits, so this probe does not exercise non-working-core Wait.
// The supplied remoteWinOffset is one common offset, not an array of peer offsets.
extern "C" __global__ __aicore__ void hccl_ccu_client_probe(
    GM_ADDR sourceGM, GM_ADDR sendOffsetsGM, GM_ADDR sendSizesGM,
    GM_ADDR statusGM, GM_ADDR tilingGM)
{
    if (AscendC::GetBlockIdx() != CLIENT_CORE_ID) {
        return;
    }
    // Copy the caller's complete tiling bytes to local memory. A standalone
    // compiler invocation does not supply operator-generator tiling macros.
    CcuClientProbeTilingData tilingData;
    const __gm__ uint8_t *tilingSource =
        reinterpret_cast<const __gm__ uint8_t *>(tilingGM);
    uint8_t *tilingDestination = reinterpret_cast<uint8_t *>(&tilingData);
    for (uint32_t byteIndex = 0; byteIndex < TILING_BYTE_COUNT; ++byteIndex) {
        tilingDestination[byteIndex] = tilingSource[byteIndex];
    }

    AscendC::GlobalTensor<int32_t> status;
    status.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(statusGM), STATUS_WORD_COUNT);
    AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU, CLIENT_CONFIG> client;
    client.InitV2(AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>(), &tilingData);
    const int32_t tilingStatus = client.SetCcTilingV2(
        offsetof(CcuClientProbeTilingData, ccWriteTiling));
    status.SetValue(TILING_STATUS_INDEX, tilingStatus);
    if (tilingStatus != 0) {
        return;
    }

    const AscendC::HcclHandle handle = client.AlltoAllvWrite<false>(
        sourceGM, sendOffsetsGM, sendSizesGM,
        tilingData.remoteWinOffset, tilingData.localDataSize);
    status.SetValue(HANDLE_STATUS_INDEX, static_cast<int32_t>(handle));
    if (handle < 0) {
        return;
    }
    client.Commit(handle);
    status.SetValue(WAIT_STATUS_INDEX, client.Wait(handle));
    client.Finalize();
}
