// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include <cstddef>

#include "kernel_operator.h"
#include "simt_descriptor_vf.h"

#define CAM_ASYNC_PIPELINE_BUDGET_INLINE __aicore__ inline
#include "simt_descriptor_budget.h"
#undef CAM_ASYNC_PIPELINE_BUDGET_INLINE

#define CAM_ASYNC_CCU_INLINE __aicore__ inline
#define CAM_ASYNC_CCU_INFO_POINTER __ubuf__ const int64_t *
#include "../cam_async_ccu/combine_segments.h"
#undef CAM_ASYNC_CCU_INFO_POINTER
#undef CAM_ASYNC_CCU_INLINE

namespace Cam::AsyncPipeline {
constexpr uint64_t DESCRIPTOR_SLOTS_PER_TILE = DESCRIPTOR_TILE_BYTES / sizeof(AsyncCcu::SendSegment);
constexpr uint64_t START_EXPERT_FIELD = 3;
constexpr uint64_t END_EXPERT_FIELD = 4;
static_assert(DESCRIPTOR_WORD_COUNT * sizeof(uint64_t) == sizeof(AsyncCcu::SendSegment),
    "VF descriptors must match the unchanged P2 CCU GM ABI");
static_assert(SOURCE_OFFSET_WORD * sizeof(uint64_t) == offsetof(AsyncCcu::SendSegment, srcByteOffset),
    "VF source offset must retain the P2 field position");
static_assert(DESTINATION_OFFSET_WORD * sizeof(uint64_t) == offsetof(AsyncCcu::SendSegment, dstByteOffset),
    "VF destination offset must retain the P2 field position");
static_assert(BYTE_COUNT_WORD * sizeof(uint64_t) == offsetof(AsyncCcu::SendSegment, byteCount),
    "VF byte count must retain the P2 field position");
static_assert(DESTINATION_RANK_WORD * sizeof(uint64_t) == offsetof(AsyncCcu::SendSegment, dstRank),
    "VF destination rank must retain the P2 field position");
static_assert(DESCRIPTOR_TILE_BYTES % sizeof(AsyncCcu::SendSegment) == 0,
    "descriptor tile must hold complete fixed slots");
static_assert(STATUS_BUFFER_BYTES >= AsyncCcu::ALIGNMENT_BYTES,
    "status MTE storage must hold one aligned block");

template <AscendC::HardEvent Event>
__aicore__ inline void Synchronize(AscendC::TPipe &pipe)
{
    const auto id = pipe.FetchEventID<Event>();
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}

__aicore__ inline void WriteProducerStatus(AscendC::TPipe &pipe,
    AscendC::LocalTensor<uint64_t> &buffer, AscendC::GlobalTensor<uint64_t> &output, uint64_t status)
{
    buffer.SetValue(0, status);
    Synchronize<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::DataCopyPad(output, buffer, {1U, sizeof(uint64_t), 0U, 0U, 0U});
    Synchronize<AscendC::HardEvent::MTE3_S>(pipe);
}
} // namespace Cam::AsyncPipeline

// Standalone A5 experimental producer; exactly one AIV owns this invocation.
// batch_info is immutable until its UB copy is made. The physical UB argument
// must come from the real platform, not from the original operator's minimum.
// The host MUST call CanLaunchDescriptorProducer before launching even a
// no-VF failure path; the compiled unit still contains SIMT code/cache resources.
// Consumers must acquire THIS invocation's producer completion before reading
// status/descriptors; status alone is not an epoch or a completion handshake.
// No ready notification, credit release, or original operator is changed here.
extern "C" __global__ __aicore__ void afd_ccu_prepare_combine_segments_simt(
    GM_ADDR batchInfo, GM_ADDR descriptors, GM_ADDR planStatus, uint64_t metadataEntries,
    uint64_t hiddenSize, uint64_t attnRankNum, uint64_t moeRankNum,
    uint64_t expertsPerRank, uint64_t tpSize, uint64_t moeRankId,
    uint64_t sourceCapacityBytes, uint64_t windowBytes, uint64_t descriptorCapacity,
    uint64_t expectedTpGroupRank, uint64_t physicalUbBytes)
{
    using namespace AscendC;
    using namespace Cam::AsyncCcu;
    using namespace Cam::AsyncPipeline;
    if (GetBlockIdx() != 0) {
        return;
    }
    TPipe pipe;
    TBuf<> statusStorage;
    pipe.InitBuffer(statusStorage, STATUS_BUFFER_BYTES);
    auto statusBuffer = statusStorage.Get<uint64_t>();
    GlobalTensor<uint64_t> statusOutput;
    statusOutput.SetGlobalBuffer((__gm__ uint64_t *)planStatus);
    if (!CanUseDescriptorSimt(physicalUbBytes)) {
        WriteProducerStatus(pipe, statusBuffer, statusOutput,
            static_cast<uint64_t>(ProducerStatus::ResourceUnavailable));
        return;
    }
    // Invalidate a retained success before planning. This does not replace the
    // caller's required same-invocation completion handoff to CCU.
    WriteProducerStatus(pipe, statusBuffer, statusOutput, static_cast<uint64_t>(Status::InvalidMetadata));

    TBuf<> metadataStorage, descriptorStorage;
    pipe.InitBuffer(metadataStorage, METADATA_BUFFER_BYTES);
    pipe.InitBuffer(descriptorStorage, DESCRIPTOR_TILE_BYTES);
    auto metadata = metadataStorage.Get<int64_t>();
    auto tile = descriptorStorage.Get<uint64_t>();
    GlobalTensor<int64_t> input;
    GlobalTensor<uint64_t> output;
    input.SetGlobalBuffer((__gm__ int64_t *)batchInfo);
    output.SetGlobalBuffer((__gm__ uint64_t *)descriptors);

    uint64_t slots, expectedEntries;
    if (!CheckedMultiply(tpSize, expertsPerRank, slots)
        || !CheckedAdd(BATCH_INFO_FIELDS, tpSize, expectedEntries)
        || !CheckedAdd(expectedEntries, slots, expectedEntries)) {
        WriteProducerStatus(pipe, statusBuffer, statusOutput, static_cast<uint64_t>(Status::ArithmeticOverflow));
        return;
    }
    if (metadataEntries != expectedEntries || metadataEntries > METADATA_BUFFER_BYTES / sizeof(int64_t)) {
        WriteProducerStatus(pipe, statusBuffer, statusOutput, static_cast<uint64_t>(Status::InvalidMetadata));
        return;
    }
    if (descriptorCapacity < slots) {
        WriteProducerStatus(pipe, statusBuffer, statusOutput, static_cast<uint64_t>(Status::DescriptorCapacityExceeded));
        return;
    }
    DataCopyPad(metadata, input, {1U, static_cast<uint32_t>(metadataEntries * sizeof(int64_t)), 0U, 0U, 0U},
        {false, 0U, 0U, 0});
    Synchronize<HardEvent::MTE2_S>(pipe);
    Synchronize<HardEvent::S_MTE2>(pipe);

    const CombineShape shape{hiddenSize, attnRankNum, moeRankNum, expertsPerRank, tpSize, moeRankId};
    const PlanBounds bounds{sourceCapacityBytes, windowBytes, descriptorCapacity, expectedTpGroupRank};
    CombineSegmentCursor cursor;
    const Status result = cursor.Init((__ubuf__ const int64_t *)metadata.GetPhyAddr(), metadataEntries, shape, bounds);
    if (result != Status::Success) {
        WriteProducerStatus(pipe, statusBuffer, statusOutput, static_cast<uint64_t>(result));
        return;
    }

    // Init preflights this same full-chunk plan and bounds every emitted span.
    // Keep scalar fields before the count rows become UB-only inclusive prefixes.
    const uint64_t startExpert = static_cast<uint64_t>(metadata.GetValue(START_EXPERT_FIELD));
    const uint64_t endExpert = static_cast<uint64_t>(metadata.GetValue(END_EXPERT_FIELD));
    const uint64_t rowBytes = COMBINE_ELEMENT_BYTES * hiddenSize;
    uint64_t columns, statisticsBytes, dispatchBytes, flagBytes, payloadBase;
    if (!CheckedMultiply(moeRankNum, expertsPerRank, columns)
        || !CheckedAdd(moeRankNum, columns, columns)
        || !CheckedMultiply(sizeof(uint32_t) * STATISTICS_ROWS, columns, statisticsBytes)
        || !CheckedAlign(statisticsBytes, dispatchBytes)
        || !CheckedMultiply(sizeof(uint32_t), moeRankNum, flagBytes)
        || !CheckedAlign(flagBytes, flagBytes)
        || !CheckedAdd(dispatchBytes, flagBytes, payloadBase)) {
        WriteProducerStatus(pipe, statusBuffer, statusOutput, static_cast<uint64_t>(Status::ArithmeticOverflow));
        return;
    }

    // The retained GM->UB copy feeds both scalar preflight and SIMT. Keep an
    // explicit MTE2->V handoff as well as the scalar-preflight->V dependency.
    Synchronize<HardEvent::MTE2_V>(pipe);
    Synchronize<HardEvent::S_V>(pipe);
    AscendC::Simt::VF_CALL<prefix_descriptor_tp_rows_vf>(AscendC::Simt::Dim3(DESCRIPTOR_VF_THREAD_COUNT),
        (__ubuf__ int64_t *)metadata.GetPhyAddr(), tpSize, expertsPerRank);
    // All TP prefix writes complete before any lane reads another TP's row.
    Synchronize<HardEvent::V_S>(pipe);
    Synchronize<HardEvent::S_V>(pipe);

    for (uint64_t slotBegin = 0; slotBegin < slots; slotBegin += DESCRIPTOR_SLOTS_PER_TILE) {
        const uint64_t remaining = slots - slotBegin;
        const uint64_t slotCount = remaining < DESCRIPTOR_SLOTS_PER_TILE ? remaining : DESCRIPTOR_SLOTS_PER_TILE;
        AscendC::Simt::VF_CALL<describe_fixed_slots_vf>(AscendC::Simt::Dim3(DESCRIPTOR_VF_THREAD_COUNT),
            (__ubuf__ const int64_t *)metadata.GetPhyAddr(), (__ubuf__ uint64_t *)tile.GetPhyAddr(),
            tpSize, expertsPerRank, startExpert, endExpert, rowBytes, payloadBase,
            expectedTpGroupRank, slotBegin, slotCount);
        Synchronize<HardEvent::V_MTE3>(pipe);
        DataCopyPad(output[slotBegin * DESCRIPTOR_WORD_COUNT], tile,
            {1U, static_cast<uint32_t>(slotCount * sizeof(SendSegment)), 0U, 0U, 0U});
        // MTE3 must finish reading the tile before another VF overwrites it.
        Synchronize<HardEvent::MTE3_S>(pipe);
        Synchronize<HardEvent::S_V>(pipe);
    }
    WriteProducerStatus(pipe, statusBuffer, statusOutput, static_cast<uint64_t>(Status::Success));
}
