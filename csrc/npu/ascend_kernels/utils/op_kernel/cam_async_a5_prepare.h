// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_A5_PREPARE_H
#define AFD_CAM_ASYNC_A5_PREPARE_H

#include "cam_async_a5_control.h"
#define CAM_ASYNC_CCU_INLINE __aicore__ inline
#define CAM_ASYNC_CCU_INFO_POINTER __ubuf__ const int64_t *
#include "cam_async_combine_segments.h"
#undef CAM_ASYNC_CCU_INFO_POINTER
#undef CAM_ASYNC_CCU_INLINE
#define CAM_ASYNC_PIPELINE_BUDGET_INLINE __aicore__ inline
#include "cam_async_descriptor_budget.h"
#undef CAM_ASYNC_PIPELINE_BUDGET_INLINE

#if defined(TILING_KEY_VAR) && TILING_KEY_VAR == AFD_ASYNC_COMBINE_PREPARE_SIMT_KEY
#define AFD_ASYNC_A5_DESCRIPTOR_SIMT
#include "cam_async_descriptor_simt.h"
#endif

namespace Cam::AsyncA5 {
constexpr uint64_t PREPARE_TILE_BYTES = 8192;
constexpr uint64_t PREPARE_STATUS_BYTES = 32;
constexpr uint64_t PREPARE_METADATA_BYTES = 65536;
constexpr uint64_t SEGMENT_WORDS = sizeof(AsyncCcu::SendSegment) / sizeof(uint64_t);
constexpr uint64_t SEGMENTS_PER_TILE = PREPARE_TILE_BYTES / sizeof(AsyncCcu::SendSegment);

template <AscendC::HardEvent Event>
__aicore__ inline void PrepareFence(AscendC::TPipe &pipe)
{
    const auto id = pipe.FetchEventID<Event>();
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}

__aicore__ inline void StorePlanStatus(AscendC::TPipe &pipe,
    AscendC::LocalTensor<uint64_t> &buffer, GM_ADDR destination, uint64_t status)
{
    buffer.SetValue(0, status);
    AscendC::GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer((__gm__ uint64_t *)destination);
    PrepareFence<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::DataCopyPad(output, buffer, {1U, sizeof(uint64_t), 0U, 0U, 0U});
    PrepareFence<AscendC::HardEvent::MTE3_S>(pipe);
}

// Integrates the P2 checked planner and P3 VF into CS's real ACLNN launch.
// Input metadata remains device-resident. Fixed global-Attention slots allow
// host CCU submission without reading the dynamically selected TP group.
// The status is published last and is invalidated by the caller before launch.
__aicore__ inline void PrepareCombine(AscendC::TPipe &pipe, GM_ADDR arguments,
    GM_ADDR batchInfo, uint64_t call)
{
    using namespace AscendC;
    RuntimeArgs args;
    AsyncPipeline::RingLayout layout;
    {
        Control control;
        control.Init(&pipe, arguments, call,
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
        args = control.Args();
        layout = control.Layout();
    }
    pipe.Reset();
    TBuf<> metadataStorage, tileStorage, statusStorage;
    pipe.InitBuffer(statusStorage, PREPARE_STATUS_BYTES);
    auto statusBuffer = statusStorage.Get<uint64_t>();
    GM_ADDR const statusAddress = reinterpret_cast<GM_ADDR>(args.planStatus);
    StorePlanStatus(pipe, statusBuffer, statusAddress, static_cast<uint64_t>(Error::InvalidMetadata));
    if (args.error != 0) return;
    const uint64_t entries = 5 + args.tpSize + args.expertsPerRank * args.tpSize;
    const uint64_t globalSlots = args.attentionRanks * args.expertsPerRank;
    if (entries > PREPARE_METADATA_BYTES / sizeof(int64_t)
        || args.descriptorCapacity < globalSlots) return;
    pipe.InitBuffer(metadataStorage, PREPARE_METADATA_BYTES);
    pipe.InitBuffer(tileStorage, PREPARE_TILE_BYTES);
    auto metadata = metadataStorage.Get<int64_t>();
    auto tile = tileStorage.Get<uint64_t>();
    GlobalTensor<int64_t> input;
    GlobalTensor<uint64_t> output;
    input.SetGlobalBuffer((__gm__ int64_t *)batchInfo);
    output.SetGlobalBuffer((__gm__ uint64_t *)args.descriptors);
    DataCopyPad(metadata, input, {1U, static_cast<uint32_t>(entries * sizeof(int64_t)), 0U, 0U, 0U},
        {false, 0U, 0U, 0});
    PrepareFence<HardEvent::MTE2_S>(pipe);
    PrepareFence<HardEvent::S_MTE2>(pipe);
    const AsyncCcu::CombineShape shape{args.hiddenSize, args.attentionRanks,
        args.moeRanks, args.expertsPerRank, args.tpSize, args.localRank};
    const AsyncCcu::PlanBounds bounds{args.sourceCapacityBytes, layout.payloadCapacity,
        args.descriptorCapacity, args.attentionGroup};
    AsyncCcu::CombineSegmentCursor cursor;
    const auto status = cursor.Init((__ubuf__ const int64_t *)metadata.GetPhyAddr(), entries, shape, bounds);
    if (status != AsyncCcu::Status::Success) {
        StorePlanStatus(pipe, statusBuffer, statusAddress, static_cast<uint64_t>(status));
        return;
    }
    // All nonparticipating peers get zero descriptors. Each invocation has a
    // complete table; stale descriptors from another DP group cannot be sent.
    for (uint32_t word = 0; word < PREPARE_TILE_BYTES / sizeof(uint64_t); ++word) tile.SetValue(word, 0);
    for (uint64_t begin = 0; begin < globalSlots; begin += SEGMENTS_PER_TILE) {
        const uint64_t count = globalSlots - begin < SEGMENTS_PER_TILE ? globalSlots - begin : SEGMENTS_PER_TILE;
        PrepareFence<HardEvent::S_MTE3>(pipe);
        DataCopyPad(output[begin * SEGMENT_WORDS], tile,
            {1U, static_cast<uint32_t>(count * sizeof(AsyncCcu::SendSegment)), 0U, 0U, 0U});
        PrepareFence<HardEvent::MTE3_S>(pipe);
    }
    const uint64_t frame = FrameOffset(layout, args.slot);
#ifdef AFD_ASYNC_A5_DESCRIPTOR_SIMT
    const uint64_t startExpert = metadata.GetValue(3);
    const uint64_t endExpert = metadata.GetValue(4);
    const uint64_t statistics = (4 * (args.moeRanks + args.moeRanks * args.expertsPerRank) * 48 + 31) / 32 * 32;
    const uint64_t flags = (4 * args.moeRanks + 31) / 32 * 32;
    PrepareFence<HardEvent::MTE2_V>(pipe);
    PrepareFence<HardEvent::S_V>(pipe);
    AscendC::Simt::VF_CALL<AsyncPipeline::prefix_descriptor_tp_rows_vf>(
        AscendC::Simt::Dim3(AsyncPipeline::DESCRIPTOR_VF_THREAD_COUNT),
        (__ubuf__ int64_t *)metadata.GetPhyAddr(), args.tpSize, args.expertsPerRank);
    PrepareFence<HardEvent::V_S>(pipe);
    PrepareFence<HardEvent::S_V>(pipe);
    const uint64_t slots = args.tpSize * args.expertsPerRank;
    for (uint64_t begin = 0; begin < slots; begin += SEGMENTS_PER_TILE) {
        const uint64_t count = slots - begin < SEGMENTS_PER_TILE ? slots - begin : SEGMENTS_PER_TILE;
        AscendC::Simt::VF_CALL<AsyncPipeline::describe_fixed_slots_vf>(
            AscendC::Simt::Dim3(AsyncPipeline::DESCRIPTOR_VF_THREAD_COUNT),
            (__ubuf__ const int64_t *)metadata.GetPhyAddr(), (__ubuf__ uint64_t *)tile.GetPhyAddr(),
            args.tpSize, args.expertsPerRank, startExpert, endExpert, args.hiddenSize * 2,
            frame + statistics + flags, args.attentionGroup, begin, count);
        PrepareFence<HardEvent::V_MTE3>(pipe);
        DataCopyPad(output[(args.attentionGroup * args.expertsPerRank + begin) * SEGMENT_WORDS], tile,
            {1U, static_cast<uint32_t>(count * sizeof(AsyncCcu::SendSegment)), 0U, 0U, 0U});
        PrepareFence<HardEvent::MTE3_S>(pipe);
        PrepareFence<HardEvent::S_V>(pipe);
    }
#else
    AsyncCcu::SendSegment segment;
    uint64_t slot;
    while (cursor.Next(segment, slot) == AsyncCcu::Status::Success) {
        segment.dstByteOffset += frame;
        tile.SetValue(0, segment.srcByteOffset);
        tile.SetValue(1, segment.dstByteOffset);
        tile.SetValue(2, segment.byteCount);
        tile.SetValue(3, segment.dstRank);
        PrepareFence<HardEvent::S_MTE3>(pipe);
        DataCopyPad(output[(args.attentionGroup * args.expertsPerRank + slot) * SEGMENT_WORDS], tile,
            {1U, sizeof(AsyncCcu::SendSegment), 0U, 0U, 0U});
        PrepareFence<HardEvent::MTE3_S>(pipe);
    }
#endif
    StorePlanStatus(pipe, statusBuffer, statusAddress, 0);
}
} // namespace Cam::AsyncA5
#endif // AFD_CAM_ASYNC_A5_PREPARE_H
