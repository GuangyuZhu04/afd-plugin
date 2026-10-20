// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_SIMT_DESCRIPTOR_VF_H
#define AFD_CAM_ASYNC_SIMT_DESCRIPTOR_VF_H

#include "kernel_operator.h"
#include "simt_api/cpp/kernel_simt_utils.h"

#define CAM_ASYNC_PIPELINE_INLINE __simt_callee__ inline
#include "cam_async_stable_descriptors.h"
#undef CAM_ASYNC_PIPELINE_INLINE

namespace Cam::AsyncPipeline {
constexpr uint32_t DESCRIPTOR_VF_THREAD_COUNT = 128;
enum DescriptorWord : uint32_t {
    SOURCE_OFFSET_WORD,
    DESTINATION_OFFSET_WORD,
    BYTE_COUNT_WORD,
    DESTINATION_RANK_WORD,
    DESCRIPTOR_WORD_COUNT,
};

struct UbDescriptorBackend {
    static __simt_callee__ inline uint64_t ReadInfo(__ubuf__ const int64_t *metadata, uint64_t index)
    {
        return static_cast<uint64_t>(metadata[index]);
    }

    static __simt_callee__ inline void WriteInfo(__ubuf__ int64_t *metadata,
        uint64_t index, uint64_t value)
    {
        metadata[index] = static_cast<int64_t>(value);
    }

    static __simt_callee__ inline void WriteDescriptor(__ubuf__ uint64_t *output,
        uint64_t localSlot, uint64_t sourceOffset, uint64_t destinationOffset,
        uint64_t bytes, uint64_t destinationRank)
    {
        const uint64_t offset = localSlot * DESCRIPTOR_WORD_COUNT;
        output[offset + SOURCE_OFFSET_WORD] = sourceOffset;
        output[offset + DESTINATION_OFFSET_WORD] = destinationOffset;
        output[offset + BYTE_COUNT_WORD] = bytes;
        output[offset + DESTINATION_RANK_WORD] = destinationRank;
    }
};

// Complete preflight has admitted all counts before these UB-only helpers run.
// Each lane owns TP rows in the first phase and fixed descriptor slots in the
// second. The native caller owns the phase fence and every GM handoff.
__simt_vf__ __launch_bounds__(DESCRIPTOR_VF_THREAD_COUNT) inline void prefix_descriptor_tp_rows_vf(
    __ubuf__ int64_t *metadata, uint64_t tpSize, uint64_t expertsPerRank)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t lanes = AscendC::Simt::GetThreadNum<0>();
    PrefixTpRowsForLane<UbDescriptorBackend>(metadata, tpSize, expertsPerRank, lane, lanes);
}

__simt_vf__ __launch_bounds__(DESCRIPTOR_VF_THREAD_COUNT) inline void describe_fixed_slots_vf(
    __ubuf__ const int64_t *metadata, __ubuf__ uint64_t *output,
    uint64_t tpSize, uint64_t expertsPerRank, uint64_t startExpert, uint64_t endExpert,
    uint64_t rowBytes, uint64_t payloadBase, uint64_t tpGroupRank,
    uint64_t slotBegin, uint64_t slotCount)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t lanes = AscendC::Simt::GetThreadNum<0>();
    DescribeSlotsForLane<UbDescriptorBackend>(metadata, output, tpSize, expertsPerRank,
        startExpert, endExpert, rowBytes, payloadBase, tpGroupRank,
        slotBegin, slotCount, lane, lanes);
}
} // namespace Cam::AsyncPipeline

#endif // AFD_CAM_ASYNC_SIMT_DESCRIPTOR_VF_H
