// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include <cstddef>

#include "kernel_operator.h"

#define CAM_ASYNC_CCU_INLINE __aicore__ inline
#define CAM_ASYNC_CCU_INFO_POINTER __ubuf__ const int64_t *
#include "combine_segments.h"
#undef CAM_ASYNC_CCU_INFO_POINTER
#undef CAM_ASYNC_CCU_INLINE

namespace Cam::AsyncCcu {
constexpr uint32_t MAX_METADATA_BYTES = 64 * 1024;
constexpr uint32_t SOURCE_OFFSET_WORD = offsetof(SendSegment, srcByteOffset) / sizeof(uint64_t);
constexpr uint32_t DESTINATION_OFFSET_WORD = offsetof(SendSegment, dstByteOffset) / sizeof(uint64_t);
constexpr uint32_t BYTE_COUNT_WORD = offsetof(SendSegment, byteCount) / sizeof(uint64_t);
constexpr uint32_t DESTINATION_RANK_WORD = offsetof(SendSegment, dstRank) / sizeof(uint64_t);

template <AscendC::HardEvent Event>
__aicore__ inline void Synchronize(AscendC::TPipe &pipe)
{
    const auto id = pipe.FetchEventID<Event>();
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}

__aicore__ inline void WriteStatus(AscendC::TPipe &pipe,
    AscendC::LocalTensor<uint64_t> &buffer, AscendC::GlobalTensor<uint64_t> &output, Status status)
{
    buffer.SetValue(0, static_cast<uint64_t>(status));
    Synchronize<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::DataCopyPad(output, buffer, {1U, sizeof(uint64_t), 0U, 0U, 0U});
    Synchronize<AscendC::HardEvent::MTE3_S>(pipe);
}
} // namespace Cam::AsyncCcu

// Experimental producer, launched with one AIV after batch_info becomes stable.
// The fixed TP group comes from the channel configuration, never from metadata.
// CCU consumers must acquire this producer's completion before reading status.
// This does not publish compact-v2 ready or change the four existing operators.
extern "C" __global__ __aicore__ void afd_ccu_prepare_combine_segments(
    GM_ADDR batchInfo, GM_ADDR descriptors, GM_ADDR planStatus, uint64_t metadataEntries,
    uint64_t hiddenSize, uint64_t attnRankNum, uint64_t moeRankNum,
    uint64_t expertsPerRank, uint64_t tpSize, uint64_t moeRankId,
    uint64_t sourceCapacityBytes, uint64_t windowBytes, uint64_t descriptorCapacity,
    uint64_t expectedTpGroupRank)
{
    using namespace AscendC;
    using namespace Cam::AsyncCcu;
    if (GetBlockIdx() != 0) {
        return;
    }
    TPipe pipe;
    TBuf<> metadataStorage, descriptorStorage, statusStorage;
    pipe.InitBuffer(metadataStorage, MAX_METADATA_BYTES);
    pipe.InitBuffer(descriptorStorage, ALIGNMENT_BYTES);
    pipe.InitBuffer(statusStorage, ALIGNMENT_BYTES);
    auto metadata = metadataStorage.Get<int64_t>();
    auto record = descriptorStorage.Get<uint64_t>();
    auto statusBuffer = statusStorage.Get<uint64_t>();
    GlobalTensor<int64_t> input;
    GlobalTensor<uint64_t> output, statusOutput;
    input.SetGlobalBuffer((__gm__ int64_t *)batchInfo);
    output.SetGlobalBuffer((__gm__ uint64_t *)descriptors);
    statusOutput.SetGlobalBuffer((__gm__ uint64_t *)planStatus);

    uint64_t slots, expectedEntries;
    if (!CheckedMultiply(tpSize, expertsPerRank, slots)
        || !CheckedAdd(BATCH_INFO_FIELDS, tpSize, expectedEntries)
        || !CheckedAdd(expectedEntries, slots, expectedEntries)) {
        WriteStatus(pipe, statusBuffer, statusOutput, Status::ArithmeticOverflow);
        return;
    }
    if (metadataEntries != expectedEntries || metadataEntries > MAX_METADATA_BYTES / sizeof(int64_t)) {
        WriteStatus(pipe, statusBuffer, statusOutput, Status::InvalidMetadata);
        return;
    }
    if (descriptorCapacity < slots) {
        WriteStatus(pipe, statusBuffer, statusOutput, Status::DescriptorCapacityExceeded);
        return;
    }
    DataCopyPad(metadata, input, {1U, static_cast<uint32_t>(metadataEntries * sizeof(int64_t)), 0U, 0U, 0U},
        {false, 0U, 0U, 0});
    Synchronize<HardEvent::MTE2_S>(pipe);
    Synchronize<HardEvent::S_MTE2>(pipe);

    const CombineShape shape{hiddenSize, attnRankNum, moeRankNum, expertsPerRank, tpSize, moeRankId};
    const PlanBounds bounds{sourceCapacityBytes, windowBytes, descriptorCapacity, expectedTpGroupRank};
    CombineSegmentCursor cursor;
    Status result = cursor.Init((__ubuf__ const int64_t *)metadata.GetPhyAddr(), metadataEntries, shape, bounds);
    if (result != Status::Success) {
        // The CCU status guard prevents any previous descriptor from being used.
        WriteStatus(pipe, statusBuffer, statusOutput, result);
        return;
    }

    for (uint64_t field = 0; field < sizeof(SendSegment) / sizeof(uint64_t); ++field) {
        record.SetValue(field, 0);
    }
    Synchronize<HardEvent::S_MTE3>(pipe);
    for (uint64_t slot = 0; slot < cursor.descriptorSlots(); ++slot) {
        DataCopyPad(output[slot * sizeof(SendSegment) / sizeof(uint64_t)], record,
            {1U, sizeof(SendSegment), 0U, 0U, 0U});
    }
    Synchronize<HardEvent::MTE3_S>(pipe);
    SendSegment segment{};
    uint64_t slotIndex;
    while ((result = cursor.Next(segment, slotIndex)) == Status::Success) {
        const uint64_t tpIndex = slotIndex / expertsPerRank;
        if (!MatchesFixedPeer(segment, expectedTpGroupRank + tpIndex)) {
            WriteStatus(pipe, statusBuffer, statusOutput, Status::TargetGroupMismatch);
            return;
        }
        record.SetValue(SOURCE_OFFSET_WORD, segment.srcByteOffset);
        record.SetValue(DESTINATION_OFFSET_WORD, segment.dstByteOffset);
        record.SetValue(BYTE_COUNT_WORD, segment.byteCount);
        record.SetValue(DESTINATION_RANK_WORD, segment.dstRank);
        Synchronize<HardEvent::S_MTE3>(pipe);
        DataCopyPad(output[slotIndex * sizeof(SendSegment) / sizeof(uint64_t)], record,
            {1U, sizeof(SendSegment), 0U, 0U, 0U});
        Synchronize<HardEvent::MTE3_S>(pipe);
    }
    WriteStatus(pipe, statusBuffer, statusOutput, result == Status::Done ? Status::Success : result);
}
