// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_RING_LAYOUT_H
#define AFD_CAM_ASYNC_RING_LAYOUT_H
#include <climits>
#include <cstdint>
#ifndef CAM_ASYNC_PIPELINE_INLINE
#define CAM_ASYNC_PIPELINE_INLINE inline
#define CAM_ASYNC_RING_DEFAULT_INLINE
#endif
#include "cam_async_slot_protocol.h"

#define CAM_ASYNC_CHECKED_INLINE CAM_ASYNC_PIPELINE_INLINE
#include "cam_async_checked_arithmetic.h"
#undef CAM_ASYNC_CHECKED_INLINE

namespace Cam::AsyncPipeline {
constexpr uint64_t RING_ALIGNMENT = 128;
constexpr uint64_t CONTROL_RECORDS_PER_SLOT = 2;
constexpr uint64_t RING_HEADER_BYTES = RING_ALIGNMENT;
constexpr uint64_t MAX_WIRE_VALUE = ~uint64_t{0};
enum class RecordRole : uint64_t { ProducerReady = 1, ReceiverCredit = 2 };
enum class WireStatus : uint64_t { Success, InvalidLayout, Overflow, InvalidSchema, WrongLease, MissingBoundary };
struct RingLayout {
    uint64_t domainId{0};
    uint64_t slotCount{0};
    uint64_t payloadCapacity{0};
    uint64_t metadataCapacity{0};
    uint64_t controlsOffset{0};
    uint64_t payloadOffset{0};
    uint64_t metadataOffsetInSlot{0};
    uint64_t slotStride{0};
    uint64_t totalBytes{0};
};

CAM_ASYNC_PIPELINE_INLINE bool WireAdd(uint64_t a, uint64_t b, uint64_t &result)
{
    if (b > MAX_WIRE_VALUE - a) return false;
    result = a + b;
    return true;
}
CAM_ASYNC_PIPELINE_INLINE bool WireMultiply(uint64_t a, uint64_t b, uint64_t &result)
{
    return AsyncArithmetic::CheckedMultiply64(a, b, result);
}
CAM_ASYNC_PIPELINE_INLINE bool WireAlign(uint64_t bytes, uint64_t &result)
{
    uint64_t rounded;
    if (!WireAdd(bytes, RING_ALIGNMENT - 1, rounded)) return false;
    result = rounded / RING_ALIGNMENT * RING_ALIGNMENT;
    return true;
}

// A dedicated registered allocation; no offset into compact-v2 is reused.
// Each slot has separate producer-ready and receiver-credit records, followed
// by its own payload and metadata storage. No shared writer cacheline is used.
CAM_ASYNC_PIPELINE_INLINE WireStatus BuildRingLayout(const ProtocolConfig &config,
    uint64_t registeredBytes, RingLayout &layout)
{
    layout = {};
    if (config.domainId == 0 || config.slotCount < DEFAULT_SLOT_COUNT
        || config.slotCount > MAX_SLOT_COUNT) return WireStatus::InvalidLayout;
    uint64_t controlBytes, payloadStorage, metadataStorage, stride, allSlotBytes, payloadOffset, total;
    if (!WireMultiply(config.slotCount, CONTROL_RECORDS_PER_SLOT * sizeof(SlotControlRecord), controlBytes)
        || !WireAdd(RING_HEADER_BYTES, controlBytes, payloadOffset)
        || !WireAlign(config.payloadCapacityBytes, payloadStorage)
        || !WireAlign(config.metadataCapacityBytes, metadataStorage)
        || !WireAdd(payloadStorage, metadataStorage, stride)
        || !WireMultiply(config.slotCount, stride, allSlotBytes)
        || !WireAdd(payloadOffset, allSlotBytes, total)) return WireStatus::Overflow;
    if (total > registeredBytes) return WireStatus::InvalidLayout;
    layout = {config.domainId, config.slotCount, config.payloadCapacityBytes,
        config.metadataCapacityBytes, RING_HEADER_BYTES, payloadOffset, payloadStorage, stride, total};
    return WireStatus::Success;
}

// Offset helpers require a successful BuildRingLayout result that remains
// immutable; arbitrary caller-constructed fields are not bounds authority.
CAM_ASYNC_PIPELINE_INLINE WireStatus ControlOffset(const RingLayout &layout, uint64_t slot,
    RecordRole role, uint64_t &offset)
{
    offset = 0;
    if (slot >= layout.slotCount || (role != RecordRole::ProducerReady
        && role != RecordRole::ReceiverCredit)) return WireStatus::InvalidLayout;
    uint64_t cell;
    if (!WireMultiply(slot, CONTROL_RECORDS_PER_SLOT, cell)
        || !WireAdd(cell, role == RecordRole::ReceiverCredit ? 1 : 0, cell)
        || !WireMultiply(cell, sizeof(SlotControlRecord), cell)
        || !WireAdd(layout.controlsOffset, cell, offset)) return WireStatus::Overflow;
    return WireStatus::Success;
}
CAM_ASYNC_PIPELINE_INLINE WireStatus PayloadOffset(const RingLayout &layout, uint64_t slot,
    uint64_t byteOffset, uint64_t byteCount, uint64_t &offset)
{
    offset = 0;
    if (slot >= layout.slotCount || byteOffset > layout.payloadCapacity
        || byteCount > layout.payloadCapacity - byteOffset) return WireStatus::InvalidLayout;
    uint64_t slotOffset;
    if (!WireMultiply(slot, layout.slotStride, slotOffset)
        || !WireAdd(layout.payloadOffset, slotOffset, offset)
        || !WireAdd(offset, byteOffset, offset)) return WireStatus::Overflow;
    return WireStatus::Success;
}

// Schema and expected-lease checks over an immutable snapshot only. A valid
// record is not a remote completion proof, atomic snapshot, or ownership CAS.
CAM_ASYNC_PIPELINE_INLINE WireStatus ValidateWireRecord(const SlotControlRecord &record,
    const RingLayout &layout, const Lease &expected, RecordRole role)
{
    if (ValidateControlRecord(record, layout.domainId, layout.slotCount) != Status::Success
        || record.payloadBytes > layout.payloadCapacity || record.metadataBytes > layout.metadataCapacity)
        return WireStatus::InvalidSchema;
    const Lease actual{record.domainId, record.slotIndex, record.epoch, record.requestId, record.chunkId};
    if (expected.epoch == 0 || !SameLease(actual, expected)) return WireStatus::WrongLease;
    if (role == RecordRole::ProducerReady) {
        if (record.state != static_cast<uint64_t>(SlotState::Ready)
            || (record.boundaryFlags & (SOURCE_CONSUMED | REMOTE_VISIBLE))
                != (SOURCE_CONSUMED | REMOTE_VISIBLE)) return WireStatus::MissingBoundary;
    } else if (role == RecordRole::ReceiverCredit) {
        if ((record.state != static_cast<uint64_t>(SlotState::Consumed)
                && record.state != static_cast<uint64_t>(SlotState::Free))
            || (record.boundaryFlags & (RECEIVER_CONSUMED | RECEIVER_CREDIT))
                != (RECEIVER_CONSUMED | RECEIVER_CREDIT)) return WireStatus::MissingBoundary;
    } else return WireStatus::InvalidLayout;
    return WireStatus::Success;
}

// Ascend/aarch64 and x86 little-endian object representation is required.
// This is not a general cross-endian conversion. Character access avoids aliasing a struct as
// an array of uint64 objects. Backend reads/writes aligned staging words only.
template <typename Backend, typename Pointer>
CAM_ASYNC_PIPELINE_INLINE void DecodeControlWords(Pointer input, SlotControlRecord &record)
{
    auto *bytes = reinterpret_cast<unsigned char *>(&record);
    for (uint64_t word = 0; word < sizeof(record) / sizeof(uint64_t); ++word) {
        const uint64_t value = Backend::ReadWord(input, word);
        for (uint64_t byte = 0; byte < sizeof(uint64_t); ++byte)
            bytes[word * sizeof(uint64_t) + byte] = static_cast<unsigned char>(value >> (byte * CHAR_BIT));
    }
}
template <typename Backend, typename Pointer>
CAM_ASYNC_PIPELINE_INLINE void EncodeControlWords(const SlotControlRecord &record, Pointer output)
{
    const auto *bytes = reinterpret_cast<const unsigned char *>(&record);
    for (uint64_t word = 0; word < sizeof(record) / sizeof(uint64_t); ++word) {
        uint64_t value = 0;
        for (uint64_t byte = 0; byte < sizeof(uint64_t); ++byte)
            value |= static_cast<uint64_t>(bytes[word * sizeof(uint64_t) + byte]) << (byte * CHAR_BIT);
        Backend::WriteWord(output, word, value);
    }
}
} // namespace Cam::AsyncPipeline
#ifdef CAM_ASYNC_RING_DEFAULT_INLINE
#undef CAM_ASYNC_RING_DEFAULT_INLINE
#undef CAM_ASYNC_PIPELINE_INLINE
#endif
#endif // AFD_CAM_ASYNC_RING_LAYOUT_H
