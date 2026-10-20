// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_A5_CONTRACT_H
#define AFD_CAM_ASYNC_A5_CONTRACT_H

#include <cstdint>
#include <limits>

#ifndef AFD_CAM_A5_INLINE
#define AFD_CAM_A5_INLINE inline
#define AFD_CAM_A5_DEFAULT_INLINE
#endif
#define CAM_ASYNC_PIPELINE_INLINE AFD_CAM_A5_INLINE
#include "cam_async_ring_layout.h"
#undef CAM_ASYNC_PIPELINE_INLINE

namespace Cam::AsyncA5 {
constexpr uint64_t CALL_MAGIC = 0x4146350000000000ULL;
constexpr uint64_t CALL_MAGIC_MASK = 0xFFFFFF0000000000ULL;
constexpr uint64_t CONTEXT_MAGIC = 0x414644413546554CULL;
constexpr uint64_t CONTEXT_VERSION = 1;
// A distinct wire identity: the runtime adds an epoch commit word and embeds
// compact frames. It must not interoperate with the immutable AFDSLOT4 codec.
constexpr uint64_t FRAME_PROTOCOL_MAGIC = 0x4146444135534C54ULL;
constexpr uint64_t FRAME_PROTOCOL_VERSION = 1;
constexpr uint64_t FRAME_PROTOCOL_FINGERPRINT = 0xA5F0010120080010ULL;
constexpr uint64_t MAX_RANKS = 64;
constexpr uint64_t MODE_MASK = 0xFF;
constexpr uint64_t SLOT_COUNT_SHIFT = 8;
constexpr uint64_t SLOT_INDEX_SHIFT = 16;
constexpr uint64_t ARGUMENT_WORDS = 32;
constexpr uint64_t POLL_LIMIT = 100000000;
constexpr uint64_t MAX_EPOCH = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

enum class Mode : uint64_t { Initialize = 1, Dispatch = 2, Receive = 3,
    PrepareCombine = 4, PublishCombine = 5, Combine = 6 };
enum class Error : uint64_t { Success = 0, InvalidContext = 1, InvalidLayout = 2,
    InvalidRecord = 3, StaleEpoch = 4, Timeout = 5, Aborted = 6,
    InvalidMetadata = 7, ResourceUnavailable = 8, InvalidWindow = 9 };

// The half tensor passed as comm_args is an opaque, session-owned allocation.
// All fields are uint64 words. Per-call tensors own the mutable ticket fields;
// no process-global handle or raw host pointer is transported to the device.
struct RuntimeArgs {
    uint64_t magic{CONTEXT_MAGIC};
    uint64_t version{CONTEXT_VERSION};
    uint64_t domain{0};
    uint64_t windowBytes{0};
    uint64_t slotCount{0};
    uint64_t slot{0};
    uint64_t epoch{0};
    uint64_t attentionGroup{0};
    uint64_t localRank{0};
    uint64_t attentionRanks{0};
    uint64_t moeRanks{0};
    uint64_t expertsPerRank{0};
    uint64_t tpSize{0};
    uint64_t hiddenSize{0};
    uint64_t topk{0};
    uint64_t receiveCapacity{0};
    uint64_t descriptors{0};
    uint64_t descriptorCapacity{0};
    uint64_t planStatus{0};
    uint64_t sourceCapacityBytes{0};
    uint64_t abortAddress{0};
    uint64_t localWindow{0};
    uint64_t physicalUbBytes{0};
    uint64_t activeAivs{0};
    uint64_t error{0};
    uint64_t lastChunk{0};
    uint64_t reserved[6]{};
};
static_assert(sizeof(RuntimeArgs) == ARGUMENT_WORDS * sizeof(uint64_t), "A5 context ABI changed");

AFD_CAM_A5_INLINE bool IsFullCall(uint64_t magic)
{
    return (magic & CALL_MAGIC_MASK) == CALL_MAGIC;
}
AFD_CAM_A5_INLINE uint64_t EncodeCall(Mode mode, uint64_t slots, uint64_t slot = 0)
{
    return CALL_MAGIC | static_cast<uint64_t>(mode)
        | (slots << SLOT_COUNT_SHIFT) | (slot << SLOT_INDEX_SHIFT);
}
AFD_CAM_A5_INLINE Mode CallMode(uint64_t magic) { return static_cast<Mode>(magic & MODE_MASK); }
AFD_CAM_A5_INLINE uint64_t CallSlots(uint64_t magic) { return (magic >> SLOT_COUNT_SHIFT) & MODE_MASK; }
AFD_CAM_A5_INLINE uint64_t CallSlot(uint64_t magic) { return (magic >> SLOT_INDEX_SHIFT) & MODE_MASK; }

// AFDSLOT4 is the outer protocol. A slot's payload contains the unchanged
// compact routed layout; metadata contains separate per-writer ready/credit
// records. The dedicated full-feature HCCL domain must never host legacy calls.
AFD_CAM_A5_INLINE bool PlanLayout(uint64_t bytes, uint64_t ranks, uint64_t slots,
    uint64_t domain, AsyncPipeline::RingLayout &layout)
{
    using namespace AsyncPipeline;
    if (ranks == 0 || ranks > MAX_RANKS || slots < DEFAULT_SLOT_COUNT || slots > MAX_SLOT_COUNT)
        return false;
    const uint64_t fixed = RING_HEADER_BYTES + slots * CONTROL_RECORDS_PER_SLOT * sizeof(SlotControlRecord);
    const uint64_t metadata = ranks * CONTROL_RECORDS_PER_SLOT * sizeof(SlotControlRecord);
    if (bytes <= fixed || (bytes - fixed) / slots <= metadata) return false;
    const uint64_t payload = ((bytes - fixed) / slots - metadata) / RING_ALIGNMENT * RING_ALIGNMENT;
    if (payload == 0) return false;
    return BuildRingLayout({domain, slots, payload, metadata}, bytes, layout) == WireStatus::Success;
}
AFD_CAM_A5_INLINE uint64_t FrameOffset(const AsyncPipeline::RingLayout &layout, uint64_t slot)
{
    return layout.payloadOffset + slot * layout.slotStride;
}
AFD_CAM_A5_INLINE uint64_t WriterRecordOffset(const AsyncPipeline::RingLayout &layout,
    uint64_t slot, uint64_t writer, AsyncPipeline::RecordRole role)
{
    return FrameOffset(layout, slot) + layout.metadataOffsetInSlot
        + (writer * AsyncPipeline::CONTROL_RECORDS_PER_SLOT
        + (role == AsyncPipeline::RecordRole::ReceiverCredit ? 1 : 0)) * sizeof(AsyncPipeline::SlotControlRecord);
}
AFD_CAM_A5_INLINE bool ValidateArgs(const RuntimeArgs &args)
{
    return args.magic == CONTEXT_MAGIC && args.version == CONTEXT_VERSION && args.domain != 0
        && args.attentionRanks > 0 && args.moeRanks > 0
        && args.attentionRanks + args.moeRanks <= MAX_RANKS
        && args.localRank < args.attentionRanks + args.moeRanks
        && args.tpSize != 0 && args.attentionRanks % args.tpSize == 0
        && args.slotCount >= AsyncPipeline::DEFAULT_SLOT_COUNT
        && args.slotCount <= AsyncPipeline::MAX_SLOT_COUNT && args.slot < args.slotCount;
}

AFD_CAM_A5_INLINE bool CommitSampleMatches(uint64_t before,
    const AsyncPipeline::SlotControlRecord &record, uint64_t after)
{
    return before != 0 && before == after && record.epoch == before && record.reserved2 == before;
}

// Validates an acquired immutable snapshot, not the physical completion itself.
// Runtime publication/consumption events must establish those boundaries first.
AFD_CAM_A5_INLINE bool FrameRecordMatches(const AsyncPipeline::SlotControlRecord &record,
    const RuntimeArgs &args, const AsyncPipeline::RingLayout &layout,
    uint64_t epoch, uint64_t slot)
{
    if (record.magic != FRAME_PROTOCOL_MAGIC || record.version != FRAME_PROTOCOL_VERSION
        || record.fingerprint != FRAME_PROTOCOL_FINGERPRINT || epoch > MAX_EPOCH
        || record.reserved2 != epoch || record.payloadBytes != layout.payloadCapacity
        || record.metadataBytes != 0) return false;
    auto normalized = record;
    normalized.magic = AsyncPipeline::PROTOCOL_MAGIC;
    normalized.version = AsyncPipeline::PROTOCOL_VERSION;
    normalized.fingerprint = AsyncPipeline::PROTOCOL_FINGERPRINT;
    normalized.reserved2 = 0;
    return AsyncPipeline::ValidateControlRecord(normalized, args.domain, args.slotCount)
            == AsyncPipeline::Status::Success
        && record.slotIndex == slot && record.epoch == epoch && epoch != 0
        && record.requestId == epoch && record.chunkId == 0;
}

AFD_CAM_A5_INLINE AsyncPipeline::SlotControlRecord FrameRecord(const RuntimeArgs &args,
    const AsyncPipeline::RingLayout &layout, AsyncPipeline::SlotState state, uint64_t boundaries)
{
    return {FRAME_PROTOCOL_MAGIC, FRAME_PROTOCOL_VERSION, FRAME_PROTOCOL_FINGERPRINT,
        args.domain, args.slot, args.epoch, args.epoch, 0, static_cast<uint64_t>(state),
        boundaries, layout.payloadCapacity, 0, args.slotCount, 0, 0, 0};
}
} // namespace Cam::AsyncA5

#ifdef AFD_CAM_A5_DEFAULT_INLINE
#undef AFD_CAM_A5_INLINE
#undef AFD_CAM_A5_DEFAULT_INLINE
#endif
#endif // AFD_CAM_ASYNC_A5_CONTRACT_H
