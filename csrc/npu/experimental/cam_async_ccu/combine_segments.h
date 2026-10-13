// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_CCU_COMBINE_SEGMENTS_H
#define AFD_CAM_ASYNC_CCU_COMBINE_SEGMENTS_H

#include <cstdint>
#include <type_traits>

// A native AIV caller may supply __aicore__ inline and an explicitly qualified
// UB metadata pointer. The standalone CPU/CCU host prototype needs no CANN types.
#ifndef CAM_ASYNC_CCU_INLINE
#define CAM_ASYNC_CCU_INLINE inline
#define CAM_ASYNC_CCU_DEFAULT_INLINE
#endif
#ifndef CAM_ASYNC_CCU_INFO_POINTER
#define CAM_ASYNC_CCU_INFO_POINTER const int64_t *
#define CAM_ASYNC_CCU_DEFAULT_INFO_POINTER
#endif

namespace Cam::AsyncCcu {
constexpr uint64_t ALIGNMENT_BYTES = 32;
constexpr uint64_t STATISTICS_ROWS = 48;
constexpr uint64_t MAX_WORLD_RANKS = 64;
constexpr uint64_t BATCH_INFO_FIELDS = 5;
constexpr uint64_t COMBINE_ELEMENT_BYTES = sizeof(uint16_t);
constexpr uint64_t MAX_SOURCE_EXPERT_ROWS = 65535;
constexpr uint64_t MAX_CHUNK_ROWS = 262144;
constexpr uint64_t MAX_U32_VALUE = 0xFFFFFFFFULL;
constexpr uint64_t MAX_U64_VALUE = ~uint64_t{0};

// Fixed GM ABI: slot tp * localExperts + expert. Unused slots are all zero.
struct alignas(32) SendSegment {
    uint64_t srcByteOffset;
    uint64_t dstByteOffset;
    uint64_t byteCount;
    uint64_t dstRank;
};
static_assert(sizeof(SendSegment) == ALIGNMENT_BYTES, "CCU descriptor must be 32 bytes");
static_assert(alignof(SendSegment) == ALIGNMENT_BYTES, "CCU descriptor must be aligned");
static_assert(std::is_standard_layout<SendSegment>::value, "CCU descriptor must have a stable layout");

struct CombineShape {
    uint64_t hiddenSize;
    uint64_t attnRankNum;
    uint64_t moeRankNum;
    uint64_t routeExpertNumPerMoe;
    uint64_t tpSize;
    uint64_t moeRankId;
};

struct PlanBounds {
    uint64_t sourceCapacityBytes;
    uint64_t windowBytes;
    uint64_t descriptorCapacity;
    // This comes from the fixed-channel launch configuration, NEVER from the
    // batchInfo being checked. A device producer rejects another ready TP group.
    uint64_t expectedTpGroupRank;
};

struct RowInterval {
    uint64_t begin;
    uint64_t end;
};

struct ReadyNotification {
    uint64_t dstRank;
    uint64_t flagByteOffset;
    uint32_t value;
};

enum class Status : uint32_t {
    Success,
    Done,
    InvalidShape,
    InvalidMetadata,
    ArithmeticOverflow,
    SourceCapacityExceeded,
    WindowCapacityExceeded,
    DescriptorCapacityExceeded,
    TargetGroupMismatch,
    NoFinalReady,
};

CAM_ASYNC_CCU_INLINE bool CheckedAdd(uint64_t left, uint64_t right, uint64_t &result)
{
    if (right > MAX_U64_VALUE - left) {
        return false;
    }
    result = left + right;
    return true;
}

CAM_ASYNC_CCU_INLINE bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t &result)
{
    if (left != 0 && right > MAX_U64_VALUE / left) {
        return false;
    }
    result = left * right;
    return true;
}

CAM_ASYNC_CCU_INLINE bool CheckedAlign(uint64_t bytes, uint64_t &result)
{
    uint64_t rounded;
    if (!CheckedAdd(bytes, ALIGNMENT_BYTES - 1, rounded)) {
        return false;
    }
    result = rounded / ALIGNMENT_BYTES * ALIGNMENT_BYTES;
    return true;
}

CAM_ASYNC_CCU_INLINE Status PartitionRows(uint64_t rows, uint64_t aivId,
    uint64_t aivCount, RowInterval &interval)
{
    interval = {};
    if (aivCount == 0 || aivCount > STATISTICS_ROWS || aivId >= aivCount) {
        return Status::InvalidShape;
    }
    const uint64_t quotient = rows / aivCount;
    const uint64_t remainder = rows % aivCount;
    uint64_t start;
    if (!CheckedMultiply(quotient, aivId, start)
        || !CheckedAdd(start, aivId < remainder ? aivId : remainder, interval.begin)
        || !CheckedAdd(interval.begin, quotient + (aivId < remainder ? 1 : 0), interval.end)) {
        return Status::ArithmeticOverflow;
    }
    return Status::Success;
}

// No transfer is allowed through a mismatched fixed channel. Empty GM records
// carry no destination and are skipped by the CCU program before data access.
CAM_ASYNC_CCU_INLINE bool MatchesFixedPeer(const SendSegment &segment, uint64_t peerRank)
{
    return segment.byteCount == 0 || segment.dstRank == peerRank;
}

class CombineSegmentCursor {
public:
    using InfoPointer = CAM_ASYNC_CCU_INFO_POINTER;

    // Validates the entire plan before emitting any descriptor. Metadata must
    // remain immutable until iteration ends, just like CS's retained UB copy.
    // The optional interval describes the original native-AIV source-row slice;
    // omitted means scalar AIV0 plans the complete chunk for CCU offload.
    CAM_ASYNC_CCU_INLINE Status Init(InfoPointer batchInfo, uint64_t entries,
        const CombineShape &shape, const PlanBounds &bounds, const RowInterval *interval = nullptr)
    {
        shape_ = shape;
        bounds_ = bounds;
        info_ = batchInfo;
        status_ = Status::InvalidShape;
        chunkRows_ = 0;
        segmentCount_ = 0;
        emittedCount_ = 0;
        slots_ = 0;
        sourceRow_ = 0;
        for (uint64_t tp = 0; tp < MAX_WORLD_RANKS; ++tp) {
            tpPrefix_[tp] = 0;
        }
        uint64_t worldSize;
        uint64_t experts;
        uint64_t columns;
        if (shape.hiddenSize == 0 || shape.hiddenSize > MAX_U32_VALUE
            || shape.hiddenSize % ALIGNMENT_BYTES != 0 || shape.attnRankNum == 0
            || shape.moeRankNum == 0 || shape.attnRankNum > MAX_WORLD_RANKS
            || shape.moeRankNum > MAX_WORLD_RANKS || shape.tpSize == 0
            || shape.tpSize > shape.attnRankNum || shape.attnRankNum % shape.tpSize != 0
            || shape.routeExpertNumPerMoe == 0 || shape.routeExpertNumPerMoe > MAX_U32_VALUE
            || !CheckedAdd(shape.attnRankNum, shape.moeRankNum, worldSize)
            || worldSize > MAX_WORLD_RANKS || shape.moeRankId < shape.attnRankNum
            || shape.moeRankId >= worldSize
            || !CheckedMultiply(shape.moeRankNum, shape.routeExpertNumPerMoe, experts)
            || !CheckedAdd(shape.moeRankNum, experts, columns) || columns > MAX_U32_VALUE
            || bounds.expectedTpGroupRank >= shape.attnRankNum
            || bounds.expectedTpGroupRank % shape.tpSize != 0
            || shape.tpSize > shape.attnRankNum - bounds.expectedTpGroupRank) {
            return status_;
        }
        uint64_t statisticsBytes;
        uint64_t flagBytes;
        uint64_t expectedEntries;
        if (!CheckedMultiply(COMBINE_ELEMENT_BYTES, shape.hiddenSize, rowBytes_)
            || !CheckedMultiply(sizeof(uint32_t), columns, statisticsBytes)
            || !CheckedMultiply(statisticsBytes, STATISTICS_ROWS, statisticsBytes)
            || !CheckedAlign(statisticsBytes, dispatchBytes_)
            || !CheckedMultiply(sizeof(uint32_t), shape.moeRankNum, flagBytes)
            || !CheckedAlign(flagBytes, flagBytes)
            || !CheckedAdd(dispatchBytes_, flagBytes, payloadBase_)
            || !CheckedMultiply(shape.tpSize, shape.routeExpertNumPerMoe, slots_)
            || !CheckedAdd(BATCH_INFO_FIELDS, shape.tpSize, expectedEntries)
            || !CheckedAdd(expectedEntries, slots_, expectedEntries)) {
            return Fail(Status::ArithmeticOverflow);
        }
        if (bounds.descriptorCapacity < slots_) {
            return Fail(Status::DescriptorCapacityExceeded);
        }
        if (bounds.windowBytes < payloadBase_) {
            return Fail(Status::WindowCapacityExceeded);
        }
        if (batchInfo == nullptr || entries != expectedEntries) {
            return Fail(Status::InvalidMetadata);
        }
        for (uint64_t field = 0; field < BATCH_INFO_FIELDS; ++field) {
            if (batchInfo[field] < 0) {
                return Fail(Status::InvalidMetadata);
            }
        }
        startExpert_ = static_cast<uint64_t>(batchInfo[3]);
        endExpert_ = static_cast<uint64_t>(batchInfo[4]);
        tpGroupRank_ = static_cast<uint64_t>(batchInfo[1]);
        if (startExpert_ > endExpert_ || endExpert_ >= shape.routeExpertNumPerMoe
            || static_cast<uint64_t>(batchInfo[2]) > MAX_U32_VALUE) {
            return Fail(Status::InvalidMetadata);
        }
        if (tpGroupRank_ != bounds.expectedTpGroupRank) {
            return Fail(Status::TargetGroupMismatch);
        }
        uint64_t wholeRows = 0;
        for (uint64_t tp = 0; tp < shape.tpSize; ++tp) {
            if (batchInfo[BATCH_INFO_FIELDS + tp] < 0
                || static_cast<uint64_t>(batchInfo[BATCH_INFO_FIELDS + tp]) >= MAX_U32_VALUE) {
                return Fail(Status::InvalidMetadata);
            }
            for (uint64_t expert = 0; expert < shape.routeExpertNumPerMoe; ++expert) {
                const int64_t signedCount = ReadCount(tp, expert);
                if (signedCount < 0 || static_cast<uint64_t>(signedCount) > MAX_SOURCE_EXPERT_ROWS) {
                    return Fail(Status::InvalidMetadata);
                }
                const uint64_t count = static_cast<uint64_t>(signedCount);
                if (!CheckedAdd(wholeRows, count, wholeRows)) {
                    return Fail(Status::ArithmeticOverflow);
                }
                if (expert < startExpert_ && !CheckedAdd(tpPrefix_[tp], count, tpPrefix_[tp])) {
                    return Fail(Status::ArithmeticOverflow);
                }
                if (expert >= startExpert_ && expert <= endExpert_
                    && !CheckedAdd(chunkRows_, count, chunkRows_)) {
                    return Fail(Status::ArithmeticOverflow);
                }
            }
        }
        if (wholeRows > MAX_U32_VALUE || wholeRows != static_cast<uint64_t>(batchInfo[0])
            || chunkRows_ > MAX_CHUNK_ROWS) {
            return Fail(Status::InvalidMetadata);
        }
        uint64_t chunkBytes;
        if (!CheckedMultiply(chunkRows_, rowBytes_, chunkBytes)) {
            return Fail(Status::ArithmeticOverflow);
        }
        if (chunkBytes > bounds.sourceCapacityBytes) {
            return Fail(Status::SourceCapacityExceeded);
        }
        interval_ = interval == nullptr ? RowInterval{0, chunkRows_} : *interval;
        if (interval_.begin > interval_.end || interval_.end > chunkRows_) {
            return Fail(Status::InvalidShape);
        }
        // Preflight every emitted span before a producer writes/launches anything.
        uint64_t sourceRow = 0;
        for (uint64_t expert = startExpert_; expert <= endExpert_; ++expert) {
            for (uint64_t tp = 0; tp < shape.tpSize; ++tp) {
                SendSegment segment{};
                const uint64_t count = static_cast<uint64_t>(ReadCount(tp, expert));
                const Status spanStatus = MakeSegment(sourceRow, tpPrefix_[tp], tp, count, segment);
                if (spanStatus != Status::Success) {
                    return Fail(spanStatus);
                }
                if (segment.byteCount != 0) {
                    ++segmentCount_;
                }
                if (!CheckedAdd(sourceRow, count, sourceRow)
                    || !CheckedAdd(tpPrefix_[tp], count, tpPrefix_[tp])) {
                    return Fail(Status::ArithmeticOverflow);
                }
            }
        }
        ResetIteration();
        status_ = Status::Success;
        return status_;
    }

    // Success returns one nonzero span and its fixed TP-major GM slot. Done means
    // all selected expert/TP cells were visited. No arithmetic errors wrap fields.
    CAM_ASYNC_CCU_INLINE Status Next(SendSegment &segment, uint64_t &slotIndex)
    {
        segment = {};
        slotIndex = 0;
        if (status_ != Status::Success) {
            return status_;
        }
        while (nextExpert_ <= endExpert_) {
            const uint64_t expert = nextExpert_;
            const uint64_t tp = nextTp_;
            const int64_t signedCount = ReadCount(tp, expert);
            if (signedCount < 0 || static_cast<uint64_t>(signedCount) > MAX_SOURCE_EXPERT_ROWS) {
                return Fail(Status::InvalidMetadata);
            }
            const uint64_t count = static_cast<uint64_t>(signedCount);
            const Status spanStatus = MakeSegment(sourceRow_, tpPrefix_[tp], tp, count, segment);
            if (spanStatus != Status::Success) {
                return Fail(spanStatus);
            }
            if (!CheckedAdd(sourceRow_, count, sourceRow_)
                || !CheckedAdd(tpPrefix_[tp], count, tpPrefix_[tp])) {
                segment = {};
                return Fail(Status::ArithmeticOverflow);
            }
            ++nextTp_;
            if (nextTp_ == shape_.tpSize) {
                nextTp_ = 0;
                ++nextExpert_;
            }
            if (segment.byteCount != 0) {
                slotIndex = tp * shape_.routeExpertNumPerMoe + expert;
                ++emittedCount_;
                return Status::Success;
            }
        }
        if (sourceRow_ != chunkRows_ || emittedCount_ != segmentCount_) {
            return Fail(Status::InvalidMetadata);
        }
        return Status::Done;
    }

    // Describes the old prefix+1 notification ONLY. This helper neither publishes
    // ready nor proves CCU completion. The caller must wait for remote visibility.
    CAM_ASYNC_CCU_INLINE Status ReadyForTp(uint64_t tp, ReadyNotification &notification) const
    {
        notification = {};
        if (status_ != Status::Success) {
            return status_;
        }
        if (tp >= shape_.tpSize) {
            return Status::InvalidShape;
        }
        if (endExpert_ != shape_.routeExpertNumPerMoe - 1) {
            return Status::NoFinalReady;
        }
        const int64_t prefix = info_[BATCH_INFO_FIELDS + tp];
        if (prefix < 0 || static_cast<uint64_t>(prefix) >= MAX_U32_VALUE) {
            return Status::InvalidMetadata;
        }
        notification.dstRank = tpGroupRank_ + tp;
        notification.flagByteOffset = dispatchBytes_
            + sizeof(uint32_t) * (shape_.moeRankId - shape_.attnRankNum);
        notification.value = static_cast<uint32_t>(static_cast<uint64_t>(prefix) + 1);
        return Status::Success;
    }

    CAM_ASYNC_CCU_INLINE uint64_t descriptorSlots() const { return slots_; }
    CAM_ASYNC_CCU_INLINE uint64_t chunkRows() const { return chunkRows_; }
    CAM_ASYNC_CCU_INLINE uint64_t segmentCount() const { return segmentCount_; }
    CAM_ASYNC_CCU_INLINE Status status() const { return status_; }

private:
    CAM_ASYNC_CCU_INLINE Status Fail(Status status)
    {
        status_ = status;
        return status;
    }

    CAM_ASYNC_CCU_INLINE int64_t ReadCount(uint64_t tp, uint64_t expert) const
    {
        return info_[BATCH_INFO_FIELDS + shape_.tpSize
            + tp * shape_.routeExpertNumPerMoe + expert];
    }

    CAM_ASYNC_CCU_INLINE void ResetIteration()
    {
        sourceRow_ = 0;
        emittedCount_ = 0;
        nextExpert_ = startExpert_;
        nextTp_ = 0;
        for (uint64_t tp = 0; tp < shape_.tpSize; ++tp) {
            tpPrefix_[tp] = 0;
            for (uint64_t expert = 0; expert < startExpert_; ++expert) {
                tpPrefix_[tp] += static_cast<uint64_t>(ReadCount(tp, expert));
            }
        }
    }

    CAM_ASYNC_CCU_INLINE Status MakeSegment(uint64_t sourceRow, uint64_t expertPrefix,
        uint64_t tp, uint64_t count, SendSegment &segment) const
    {
        segment = {};
        uint64_t sourceEnd;
        if (!CheckedAdd(sourceRow, count, sourceEnd)) {
            return Status::ArithmeticOverflow;
        }
        const uint64_t begin = sourceRow > interval_.begin ? sourceRow : interval_.begin;
        const uint64_t end = sourceEnd < interval_.end ? sourceEnd : interval_.end;
        if (begin >= end) {
            return Status::Success;
        }
        const int64_t signedPrefix = info_[BATCH_INFO_FIELDS + tp];
        if (signedPrefix < 0 || static_cast<uint64_t>(signedPrefix) >= MAX_U32_VALUE) {
            return Status::InvalidMetadata;
        }
        uint64_t destinationRow;
        uint64_t destinationPayloadOffset;
        uint64_t sourceSpanEnd;
        uint64_t destinationSpanEnd;
        if (!CheckedAdd(static_cast<uint64_t>(signedPrefix), expertPrefix, destinationRow)
            || !CheckedAdd(destinationRow, begin - sourceRow, destinationRow)
            || !CheckedMultiply(begin, rowBytes_, segment.srcByteOffset)
            || !CheckedMultiply(end - begin, rowBytes_, segment.byteCount)
            || !CheckedMultiply(destinationRow, rowBytes_, destinationPayloadOffset)
            || !CheckedAdd(payloadBase_, destinationPayloadOffset, segment.dstByteOffset)
            || !CheckedAdd(segment.srcByteOffset, segment.byteCount, sourceSpanEnd)
            || !CheckedAdd(segment.dstByteOffset, segment.byteCount, destinationSpanEnd)) {
            segment = {};
            return Status::ArithmeticOverflow;
        }
        if (sourceSpanEnd > bounds_.sourceCapacityBytes) {
            segment = {};
            return Status::SourceCapacityExceeded;
        }
        if (destinationSpanEnd > bounds_.windowBytes) {
            segment = {};
            return Status::WindowCapacityExceeded;
        }
        segment.dstRank = tpGroupRank_ + tp;
        return Status::Success;
    }

    InfoPointer info_{nullptr};
    CombineShape shape_{};
    PlanBounds bounds_{};
    RowInterval interval_{};
    Status status_{Status::InvalidMetadata};
    uint64_t rowBytes_{0};
    uint64_t dispatchBytes_{0};
    uint64_t payloadBase_{0};
    uint64_t slots_{0};
    uint64_t chunkRows_{0};
    uint64_t segmentCount_{0};
    uint64_t emittedCount_{0};
    uint64_t sourceRow_{0};
    uint64_t startExpert_{0};
    uint64_t endExpert_{0};
    uint64_t nextExpert_{0};
    uint64_t nextTp_{0};
    uint64_t tpGroupRank_{0};
    uint64_t tpPrefix_[MAX_WORLD_RANKS]{};
};
} // namespace Cam::AsyncCcu

#ifdef CAM_ASYNC_CCU_DEFAULT_INFO_POINTER
#undef CAM_ASYNC_CCU_DEFAULT_INFO_POINTER
#undef CAM_ASYNC_CCU_INFO_POINTER
#endif
#ifdef CAM_ASYNC_CCU_DEFAULT_INLINE
#undef CAM_ASYNC_CCU_DEFAULT_INLINE
#undef CAM_ASYNC_CCU_INLINE
#endif
#endif // AFD_CAM_ASYNC_CCU_COMBINE_SEGMENTS_H
