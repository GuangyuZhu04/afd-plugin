// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
// Expected-negative compile diagnostic for the installed CANN 9.0 v310 SDK.
// Hccl declares BatchWrite publicly, but the complete CCU HcclImpl specialization
// lacks that method. Do not include this file in production package sources.

#include <cstdint>

#include "kernel_operator.h"
#include "adv_api/hccl/hccl.h"

#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3101
#error "This diagnostic must use the installed CANN 9.0 dav-c310-vec target (3101)."
#endif

namespace {
constexpr uint32_t CLIENT_CORE_ID = 0;
constexpr uint32_t PROBE_ITEM_COUNT = 1;
constexpr uint16_t PROBE_QUEUE_ID = 0;
constexpr AscendC::HcclServerConfig CLIENT_CONFIG = {
    AscendC::CoreType::ON_AIV, CLIENT_CORE_ID};
} // namespace

// No execution is intended. The expected failure is specifically the missing
// HcclImpl<HCCL_SERVER_TYPE_CCU>::BatchWrite member, not an include/target error.
extern "C" __global__ __aicore__ void negative_ccu_batch_write_probe(
    GM_ADDR descriptorsGM, GM_ADDR handleGM)
{
    AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU, CLIENT_CONFIG> client;
    const AscendC::HcclHandle handle = client.BatchWrite<true>(
        descriptorsGM, PROBE_ITEM_COUNT, PROBE_QUEUE_ID);
    AscendC::GlobalTensor<int32_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(handleGM));
    result.SetValue(CLIENT_CORE_ID, static_cast<int32_t>(handle));
}
