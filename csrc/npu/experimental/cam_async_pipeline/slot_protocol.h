// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_SLOT_PROTOCOL_H
#define AFD_CAM_ASYNC_SLOT_PROTOCOL_H
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef CAM_ASYNC_PIPELINE_INLINE
#define CAM_ASYNC_PIPELINE_INLINE inline
#define CAM_ASYNC_PIPELINE_DEFAULT_INLINE
#endif
namespace Cam::AsyncPipeline {
constexpr uint64_t PROTOCOL_MAGIC = 0x414644534C4F5434ULL;
constexpr uint64_t PROTOCOL_VERSION = 1;
// Format identifier for this 16-word schema, not a payload hash or proof.
constexpr uint64_t PROTOCOL_FINGERPRINT = 0xBF8A4DFD12800101ULL;
constexpr uint64_t DEFAULT_SLOT_COUNT = 2;
constexpr uint64_t MAX_SLOT_COUNT = 8;
constexpr uint64_t MAX_COMBINE_SLOTS = 64;
constexpr uint64_t MAX_EPOCH = ~uint64_t{0};

enum class SlotState : uint64_t { Free, Writing, Ready, Reading, Consumed };
enum class Status : uint32_t {
    Success, NoCredit, InvalidConfig, InvalidLease, StaleEpoch, InvalidState,
    MissingProof, Duplicate, Aborted, EpochExhausted, Done, NotReady,
    WrongRequest, WrongIndex, InvalidRecord
};
constexpr uint64_t SOURCE_CONSUMED = uint64_t{1} << 0;
constexpr uint64_t REMOTE_VISIBLE = uint64_t{1} << 1;
constexpr uint64_t RECEIVER_CONSUMED = uint64_t{1} << 2;
constexpr uint64_t TASK_RETIRED = uint64_t{1} << 3;
constexpr uint64_t RECEIVER_CREDIT = uint64_t{1} << 4;
constexpr uint64_t ALL_BOUNDARY_FLAGS = SOURCE_CONSUMED | REMOTE_VISIBLE
    | RECEIVER_CONSUMED | TASK_RETIRED | RECEIVER_CREDIT;

struct Lease {
    uint64_t domainId{0};
    uint64_t slotIndex{0};
    uint64_t epoch{0};
    uint64_t requestId{0};
    uint64_t chunkId{0};
};
struct alignas(64) SlotControlRecord {
    uint64_t magic;
    uint64_t version;
    uint64_t fingerprint;
    uint64_t domainId;
    uint64_t slotIndex;
    uint64_t epoch;
    uint64_t requestId;
    uint64_t chunkId;
    uint64_t state;
    uint64_t boundaryFlags;
    uint64_t payloadBytes;
    uint64_t metadataBytes;
    uint64_t slotCount;
    uint64_t reserved0;
    uint64_t reserved1;
    uint64_t reserved2;
};
static_assert(sizeof(SlotControlRecord) == 128 && alignof(SlotControlRecord) == 64,
    "P4 control record is two aligned 64-byte lines");
static_assert(std::is_standard_layout<SlotControlRecord>::value, "P4 control ABI must be stable");
static_assert(offsetof(SlotControlRecord, state) == 64
    && offsetof(SlotControlRecord, boundaryFlags) == 72
    && offsetof(SlotControlRecord, slotCount) == 96, "P4 control offsets changed");

struct ProtocolConfig {
    uint64_t domainId{0};
    uint64_t slotCount{DEFAULT_SLOT_COUNT};
    uint64_t payloadCapacityBytes{0};
    uint64_t metadataCapacityBytes{0};
};
// These are qualified external boundary claims, never inferred from Launch,
// a P2 WaitEvent, a planning status, or a control record that merely validates.
struct LocalCompletionProof { Lease lease{}; bool localDone{false}; };
struct SourceConsumedProof {
    Lease lease{};
    bool payloadReadsFinished{false};
    bool descriptorReadsFinished{false};
};
struct RemoteVisibilityProof {
    Lease lease{};
    bool payloadRemoteVisible{false};
    bool metadataRemoteVisible{false};
};
struct ReceiverConsumedProof {
    Lease lease{};
    bool wireReadsFinished{false};
    bool ownedBackingCaptured{false};
};
struct TaskRetiredProof {
    Lease lease{};
    bool taskEnded{false};
    bool noFutureTargetWrites{false};
};
struct ReceiverCreditProof { Lease lease{}; bool creditAcknowledged{false}; };
struct QuiescenceProof {
    uint64_t domainId{0};
    uint64_t lastIssuedEpoch{0};
    bool sourceReadsStopped{false};
    bool descriptorReadsStopped{false};
    bool targetWritesStopped{false};
    bool receiverReadsStopped{false};
    bool ownedBackingAccessStopped{false};
};

CAM_ASYNC_PIPELINE_INLINE bool SameLease(const Lease &a, const Lease &b)
{
    return a.domainId == b.domainId && a.slotIndex == b.slotIndex && a.epoch == b.epoch
        && a.requestId == b.requestId && a.chunkId == b.chunkId;
}

// Syntax/schema validation only. It is not an atomic snapshot, release/acquire
// operation, a proof of data visibility, or an inter-rank ownership transition.
CAM_ASYNC_PIPELINE_INLINE Status ValidateControlRecord(const SlotControlRecord &record,
    uint64_t domainId, uint64_t slotCount)
{
    if (domainId == 0 || slotCount < DEFAULT_SLOT_COUNT || slotCount > MAX_SLOT_COUNT
        || record.magic != PROTOCOL_MAGIC || record.version != PROTOCOL_VERSION
        || record.fingerprint != PROTOCOL_FINGERPRINT || record.domainId != domainId
        || record.slotCount != slotCount || record.slotIndex >= slotCount
        || record.state > static_cast<uint64_t>(SlotState::Consumed)
        || (record.boundaryFlags & ~ALL_BOUNDARY_FLAGS) != 0
        || record.reserved0 != 0 || record.reserved1 != 0 || record.reserved2 != 0
        || (record.state != static_cast<uint64_t>(SlotState::Free) && record.epoch == 0)) {
        return Status::InvalidRecord;
    }
    const auto state = static_cast<SlotState>(record.state);
    if ((state == SlotState::Ready || state == SlotState::Reading || state == SlotState::Consumed)
        && (record.boundaryFlags & (SOURCE_CONSUMED | REMOTE_VISIBLE))
            != (SOURCE_CONSUMED | REMOTE_VISIBLE)) return Status::InvalidRecord;
    if (state == SlotState::Consumed && (record.boundaryFlags & RECEIVER_CONSUMED) == 0)
        return Status::InvalidRecord;
    if ((record.boundaryFlags & RECEIVER_CONSUMED) != 0
        && state != SlotState::Consumed && state != SlotState::Free) return Status::InvalidRecord;
    if ((record.boundaryFlags & RECEIVER_CREDIT) != 0
        && state != SlotState::Consumed && state != SlotState::Free) return Status::InvalidRecord;
    return Status::Success;
}

// Portable single-coordinator reference protocol. Calls must be serialized.
// It does not implement GM CAS, network messages, fences or SDK event bindings.
// The caller keeps descriptors immutable until SourceConsumed and target slot
// data immutable until both ReceiverCredit and TaskRetired have been qualified.
class BoundedSlotProtocol {
public:
    CAM_ASYNC_PIPELINE_INLINE Status Init(const ProtocolConfig &config, uint64_t lastIssuedEpoch = 0)
    {
        if (initialized_) return Status::InvalidState;
        if (config.domainId == 0 || config.slotCount < DEFAULT_SLOT_COUNT
            || config.slotCount > MAX_SLOT_COUNT
            || config.payloadCapacityBytes > MAX_EPOCH - config.metadataCapacityBytes
            || config.payloadCapacityBytes + config.metadataCapacityBytes
                > MAX_EPOCH - sizeof(SlotControlRecord)
            || config.payloadCapacityBytes + config.metadataCapacityBytes + sizeof(SlotControlRecord)
                > MAX_EPOCH / config.slotCount) return Status::InvalidConfig;
        config_ = config;
        lastEpoch_ = lastIssuedEpoch;
        for (uint64_t slot = 0; slot < config.slotCount; ++slot) {
            records_[slot] = {PROTOCOL_MAGIC, PROTOCOL_VERSION, PROTOCOL_FINGERPRINT,
                config.domainId, slot, 0, 0, 0, static_cast<uint64_t>(SlotState::Free),
                0, 0, 0, config.slotCount, 0, 0, 0};
        }
        initialized_ = true;
        return Status::Success;
    }

    CAM_ASYNC_PIPELINE_INLINE Status Acquire(uint64_t requestId, uint64_t chunkId,
        uint64_t payloadBytes, uint64_t metadataBytes, Lease &lease)
    {
        lease = {};
        if (!initialized_) return Status::InvalidState;
        if (aborted_) return Status::Aborted;
        if (lastEpoch_ == MAX_EPOCH) return Status::EpochExhausted;
        if (payloadBytes > config_.payloadCapacityBytes
            || metadataBytes > config_.metadataCapacityBytes) return Status::InvalidConfig;
        for (uint64_t slot = 0; slot < config_.slotCount; ++slot) {
            const auto &record = records_[slot];
            if (record.state != static_cast<uint64_t>(SlotState::Free)
                && record.requestId == requestId && record.chunkId == chunkId) return Status::Duplicate;
        }
        for (uint64_t slot = 0; slot < config_.slotCount; ++slot) {
            auto &record = records_[slot];
            if (record.state != static_cast<uint64_t>(SlotState::Free)) continue;
            ++lastEpoch_;
            record.epoch = lastEpoch_;
            record.requestId = requestId;
            record.chunkId = chunkId;
            record.state = static_cast<uint64_t>(SlotState::Writing);
            record.boundaryFlags = 0;
            record.payloadBytes = payloadBytes;
            record.metadataBytes = metadataBytes;
            lease = {config_.domainId, slot, lastEpoch_, requestId, chunkId};
            return Status::Success;
        }
        return Status::NoCredit;
    }

    CAM_ASYNC_PIPELINE_INLINE Status RecordSourceConsumed(const SourceConsumedProof &proof)
    {
        const Status match = CheckActive(proof.lease);
        if (match != Status::Success) return match;
        if (!proof.payloadReadsFinished || !proof.descriptorReadsFinished) return Status::MissingProof;
        return SetBoundary(proof.lease.slotIndex, SOURCE_CONSUMED);
    }
    CAM_ASYNC_PIPELINE_INLINE Status RecordRemoteVisibility(const RemoteVisibilityProof &proof)
    {
        const Status match = CheckActive(proof.lease);
        if (match != Status::Success) return match;
        if (!proof.payloadRemoteVisible || !proof.metadataRemoteVisible) return Status::MissingProof;
        return SetBoundary(proof.lease.slotIndex, REMOTE_VISIBLE);
    }
    CAM_ASYNC_PIPELINE_INLINE Status PublishReady(const Lease &lease)
    {
        if (aborted_) return Status::Aborted;
        const Status match = CheckActive(lease);
        if (match != Status::Success) return match;
        auto &record = records_[lease.slotIndex];
        if (record.state != static_cast<uint64_t>(SlotState::Writing)) return Status::InvalidState;
        if ((record.boundaryFlags & (SOURCE_CONSUMED | REMOTE_VISIBLE))
            != (SOURCE_CONSUMED | REMOTE_VISIBLE)) return Status::MissingProof;
        record.state = static_cast<uint64_t>(SlotState::Ready);
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status BeginRead(const Lease &lease)
    {
        if (aborted_) return Status::Aborted;
        const Status match = CheckActive(lease);
        if (match != Status::Success) return match;
        auto &record = records_[lease.slotIndex];
        if (record.state != static_cast<uint64_t>(SlotState::Ready)) return Status::InvalidState;
        record.state = static_cast<uint64_t>(SlotState::Reading);
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status CompleteRead(const ReceiverConsumedProof &proof)
    {
        const Status match = CheckActive(proof.lease);
        if (match != Status::Success) return match;
        auto &record = records_[proof.lease.slotIndex];
        if (record.state != static_cast<uint64_t>(SlotState::Reading)) return Status::InvalidState;
        if (!proof.wireReadsFinished || !proof.ownedBackingCaptured) return Status::MissingProof;
        record.boundaryFlags |= RECEIVER_CONSUMED;
        record.state = static_cast<uint64_t>(SlotState::Consumed);
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status ReturnCredit(const ReceiverCreditProof &proof)
    {
        const Status match = CheckActive(proof.lease);
        if (match != Status::Success) return match;
        auto &record = records_[proof.lease.slotIndex];
        if (record.state != static_cast<uint64_t>(SlotState::Consumed)) return Status::InvalidState;
        if (!proof.creditAcknowledged) return Status::MissingProof;
        const Status result = SetBoundary(proof.lease.slotIndex, RECEIVER_CREDIT);
        MaybeFree(proof.lease.slotIndex);
        return result;
    }
    CAM_ASYNC_PIPELINE_INLINE Status RecordTaskRetired(const TaskRetiredProof &proof)
    {
        const Status match = CheckActive(proof.lease);
        if (match != Status::Success) return match;
        if (!proof.taskEnded || !proof.noFutureTargetWrites) return Status::MissingProof;
        const Status result = SetBoundary(proof.lease.slotIndex, TASK_RETIRED);
        MaybeFree(proof.lease.slotIndex);
        return result;
    }
    CAM_ASYNC_PIPELINE_INLINE Status CanReclaimSource(const Lease &lease, bool &reclaimable) const
    {
        reclaimable = false;
        const Status match = CheckLease(lease);
        if (match != Status::Success) return match;
        reclaimable = (records_[lease.slotIndex].boundaryFlags & SOURCE_CONSUMED) != 0;
        return Status::Success;
    }

    CAM_ASYNC_PIPELINE_INLINE Status Abort()
    {
        if (!initialized_) return Status::InvalidState;
        aborted_ = true;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status RecoverAfterQuiescence(const QuiescenceProof &proof)
    {
        if (!initialized_ || !aborted_) return Status::InvalidState;
        if (proof.domainId != config_.domainId || proof.lastIssuedEpoch != lastEpoch_)
            return Status::InvalidLease;
        if (!proof.sourceReadsStopped || !proof.descriptorReadsStopped || !proof.targetWritesStopped
            || !proof.receiverReadsStopped || !proof.ownedBackingAccessStopped) return Status::MissingProof;
        for (uint64_t slot = 0; slot < config_.slotCount; ++slot) {
            records_[slot].state = static_cast<uint64_t>(SlotState::Free);
            records_[slot].boundaryFlags = 0;
            records_[slot].payloadBytes = 0;
            records_[slot].metadataBytes = 0;
        }
        // Preserve every epoch and the monotone issuer; recovery never rewinds.
        aborted_ = false;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE uint64_t availableCredits() const
    {
        if (!initialized_ || aborted_) return 0;
        uint64_t count = 0;
        for (uint64_t slot = 0; slot < config_.slotCount; ++slot)
            count += records_[slot].state == static_cast<uint64_t>(SlotState::Free) ? 1 : 0;
        return count;
    }
    CAM_ASYNC_PIPELINE_INLINE uint64_t lastIssuedEpoch() const { return lastEpoch_; }
    CAM_ASYNC_PIPELINE_INLINE bool aborted() const { return aborted_; }
    CAM_ASYNC_PIPELINE_INLINE Status Snapshot(uint64_t slot, SlotControlRecord &record) const
    {
        record = {};
        if (!initialized_ || slot >= config_.slotCount) return Status::InvalidLease;
        record = records_[slot];
        return Status::Success;
    }

private:
    CAM_ASYNC_PIPELINE_INLINE Status CheckLease(const Lease &lease) const
    {
        if (!initialized_ || lease.domainId != config_.domainId || lease.slotIndex >= config_.slotCount
            || lease.epoch == 0) return Status::InvalidLease;
        const auto &record = records_[lease.slotIndex];
        if (lease.epoch != record.epoch) return Status::StaleEpoch;
        if (lease.requestId != record.requestId || lease.chunkId != record.chunkId) return Status::InvalidLease;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status CheckActive(const Lease &lease) const
    {
        const Status status = CheckLease(lease);
        if (status != Status::Success) return status;
        return records_[lease.slotIndex].state == static_cast<uint64_t>(SlotState::Free)
            ? Status::InvalidState : Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status SetBoundary(uint64_t slot, uint64_t flag)
    {
        if ((records_[slot].boundaryFlags & flag) != 0) return Status::Duplicate;
        records_[slot].boundaryFlags |= flag;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE void MaybeFree(uint64_t slot)
    {
        auto &record = records_[slot];
        if (!aborted_ && record.state == static_cast<uint64_t>(SlotState::Consumed)
            && (record.boundaryFlags & (RECEIVER_CREDIT | TASK_RETIRED))
                == (RECEIVER_CREDIT | TASK_RETIRED)) record.state = static_cast<uint64_t>(SlotState::Free);
    }
    ProtocolConfig config_{};
    SlotControlRecord records_[MAX_SLOT_COUNT]{};
    uint64_t lastEpoch_{0};
    bool initialized_{false};
    bool aborted_{false};
};

struct CaptureTicket { Lease lease{}; uint64_t logicalIndex{0}; };
struct OwnedCaptureProof { CaptureTicket ticket{}; bool ownedBytesReady{false}; bool wireReadsFinished{false}; };
struct ReductionCompleteProof {
    CaptureTicket ticket{};
    bool arithmeticComplete{false};
    bool noFutureBackingReads{false};
};
struct OwnedBackingQuiescenceProof {
    uint64_t domainId{0};
    uint64_t requestId{0};
    bool capturesStopped{false};
    bool reductionsStopped{false};
    bool backingReadsStopped{false};
};
enum class ContributionState : uint32_t { Empty, Capturing, Captured, Reducing, Reduced };

// The backing itself is caller-owned and independent of transport slots. Capture
// completes before receiver credit; this avoids high-index arrivals occupying all
// wire slots while logical index zero waits. Only a proven completed reduction
// advances the logical top-k cursor, never arrival order or issue order alone.
// This gate is not a transport authority: before ReserveCapture the caller must
// successfully BeginRead for the current active transport lease, retain that
// lease until wire reads complete, and validate logicalIndex against routing.
// Receiver credit is returned only after both capture and transport CompleteRead
// succeed. The gate cannot establish GM ownership or validate routing itself.
class OrderedCombineGate {
public:
    CAM_ASYNC_PIPELINE_INLINE Status Init(uint64_t domainId, uint64_t requestId, uint64_t topk)
    {
        if (domainId == 0 || topk == 0 || topk > MAX_COMBINE_SLOTS) return Status::InvalidConfig;
        if (initialized_) {
            if (domainId != domain_ || requestId <= request_) return Status::WrongRequest;
            if ((aborted_ && !quiesced_) || (!aborted_ && next_ != topk_)) return Status::InvalidState;
        }
        domain_ = domainId;
        request_ = requestId;
        topk_ = topk;
        next_ = 0;
        reducing_ = false;
        aborted_ = false;
        quiesced_ = false;
        for (uint64_t index = 0; index < MAX_COMBINE_SLOTS; ++index) {
            states_[index] = ContributionState::Empty;
            leases_[index] = {};
        }
        initialized_ = true;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status ReserveCapture(const Lease &lease, uint64_t logicalIndex,
        CaptureTicket &ticket)
    {
        ticket = {};
        if (!initialized_) return Status::InvalidState;
        if (aborted_) return Status::Aborted;
        if (lease.domainId != domain_ || lease.requestId != request_) return Status::WrongRequest;
        if (lease.epoch == 0 || lease.slotIndex >= MAX_SLOT_COUNT) return Status::InvalidLease;
        if (logicalIndex >= topk_) return Status::WrongIndex;
        if (states_[logicalIndex] != ContributionState::Empty) return Status::Duplicate;
        leases_[logicalIndex] = lease;
        states_[logicalIndex] = ContributionState::Capturing;
        ticket = {lease, logicalIndex};
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status CaptureComplete(const OwnedCaptureProof &proof)
    {
        const Status match = CheckTicket(proof.ticket);
        if (match != Status::Success) return match;
        const uint64_t index = proof.ticket.logicalIndex;
        if (states_[index] != ContributionState::Capturing) return Status::InvalidState;
        if (!proof.ownedBytesReady || !proof.wireReadsFinished) return Status::MissingProof;
        states_[index] = ContributionState::Captured;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status BeginNextReduction(uint64_t &logicalIndex)
    {
        logicalIndex = 0;
        if (!initialized_) return Status::InvalidState;
        if (aborted_) return Status::Aborted;
        if (reducing_) return Status::InvalidState;
        if (next_ == topk_) return Status::Done;
        if (states_[next_] != ContributionState::Captured) return Status::NotReady;
        states_[next_] = ContributionState::Reducing;
        reducing_ = true;
        logicalIndex = next_;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status CompleteReduction(const ReductionCompleteProof &proof)
    {
        const Status match = CheckTicket(proof.ticket);
        if (match != Status::Success) return match;
        const uint64_t index = proof.ticket.logicalIndex;
        if (!reducing_ || index != next_ || states_[index] != ContributionState::Reducing)
            return Status::InvalidState;
        if (!proof.arithmeticComplete || !proof.noFutureBackingReads) return Status::MissingProof;
        states_[index] = ContributionState::Reduced;
        reducing_ = false;
        ++next_;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status Abort()
    {
        if (!initialized_) return Status::InvalidState;
        aborted_ = true;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE Status QuiesceOwnedBacking(const OwnedBackingQuiescenceProof &proof)
    {
        if (!initialized_ || !aborted_) return Status::InvalidState;
        if (proof.domainId != domain_ || proof.requestId != request_) return Status::WrongRequest;
        if (!proof.capturesStopped || !proof.reductionsStopped || !proof.backingReadsStopped)
            return Status::MissingProof;
        quiesced_ = true;
        return Status::Success;
    }
    CAM_ASYNC_PIPELINE_INLINE uint64_t nextLogicalIndex() const { return next_; }
    CAM_ASYNC_PIPELINE_INLINE bool allReduced() const { return initialized_ && next_ == topk_; }
private:
    CAM_ASYNC_PIPELINE_INLINE Status CheckTicket(const CaptureTicket &ticket) const
    {
        if (!initialized_ || ticket.lease.domainId != domain_ || ticket.lease.requestId != request_)
            return Status::WrongRequest;
        if (ticket.logicalIndex >= topk_) return Status::WrongIndex;
        if (!SameLease(ticket.lease, leases_[ticket.logicalIndex])) return Status::StaleEpoch;
        return Status::Success;
    }
    Lease leases_[MAX_COMBINE_SLOTS]{};
    ContributionState states_[MAX_COMBINE_SLOTS]{};
    uint64_t domain_{0};
    uint64_t request_{0};
    uint64_t topk_{0};
    uint64_t next_{0};
    bool initialized_{false};
    bool reducing_{false};
    bool aborted_{false};
    bool quiesced_{false};
};
} // namespace Cam::AsyncPipeline
#ifdef CAM_ASYNC_PIPELINE_DEFAULT_INLINE
#undef CAM_ASYNC_PIPELINE_DEFAULT_INLINE
#undef CAM_ASYNC_PIPELINE_INLINE
#endif
#endif // AFD_CAM_ASYNC_SLOT_PROTOCOL_H
