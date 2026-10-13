// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// This serialized CPU probe checks caller ownership and fixed reduction order.
// It does not implement or qualify SDK events, GM atomic publication, CCU
// remote visibility, task retirement, cache coherency, or NPU performance.
#include "slot_protocol.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>

using Cam::AsyncPipeline::BoundedSlotProtocol;
using Cam::AsyncPipeline::CaptureTicket;
using Cam::AsyncPipeline::ContributionState;
using Cam::AsyncPipeline::Lease;
using Cam::AsyncPipeline::OrderedCombineGate;
using Cam::AsyncPipeline::ProtocolConfig;
using Cam::AsyncPipeline::SlotControlRecord;
using Cam::AsyncPipeline::SlotState;
using Cam::AsyncPipeline::Status;
namespace Pipeline = Cam::AsyncPipeline;
namespace {
constexpr uint64_t TEST_DOMAIN = 41;
constexpr uint64_t TEST_REQUEST = 17;
constexpr uint64_t PAYLOAD_CAPACITY = 32;
constexpr uint64_t METADATA_CAPACITY = 8;
constexpr uint64_t THREE_CONTRIBUTIONS = 3;
constexpr uint8_t FIRST_PATTERN = 0x35;
constexpr uint8_t SECOND_PATTERN = 0xA6;
constexpr uint8_t REUSE_PATTERN = 0xE9;
constexpr float LARGE_FLOAT = 16777216.0F;

void Require(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
void Expect(Status actual, Status expected, const char *message)
{
    if (actual != expected) {
        std::cerr << message << ": status=" << static_cast<uint32_t>(actual)
            << ", expected=" << static_cast<uint32_t>(expected) << '\n';
        std::exit(EXIT_FAILURE);
    }
}
ProtocolConfig Config(uint64_t slots = Pipeline::DEFAULT_SLOT_COUNT)
{
    return {TEST_DOMAIN, slots, PAYLOAD_CAPACITY, METADATA_CAPACITY};
}
SlotControlRecord Snapshot(const BoundedSlotProtocol &protocol, const Lease &lease)
{
    SlotControlRecord record{};
    Expect(protocol.Snapshot(lease.slotIndex, record), Status::Success, "snapshot");
    return record;
}
void QualifyReady(BoundedSlotProtocol &protocol, const Lease &lease)
{
    Expect(protocol.RecordSourceConsumed({lease, true, true}), Status::Success, "source consumed");
    Expect(protocol.RecordRemoteVisibility({lease, true, true}), Status::Success, "remote visible");
    Expect(protocol.PublishReady(lease), Status::Success, "publish ready");
}
void Consume(BoundedSlotProtocol &protocol, const Lease &lease)
{
    Expect(protocol.BeginRead(lease), Status::Success, "begin read");
    Expect(protocol.CompleteRead({lease, true, true}), Status::Success, "receiver consumed");
}
void Release(BoundedSlotProtocol &protocol, const Lease &lease)
{
    Expect(protocol.RecordTaskRetired({lease, true, true}), Status::Success, "task retired");
    Expect(protocol.ReturnCredit({lease, true}), Status::Success, "return credit");
}
void ConfigSchema()
{
    BoundedSlotProtocol uninitialized;
    Lease lease{TEST_DOMAIN, 0, 1, TEST_REQUEST, 0};
    Expect(uninitialized.Acquire(TEST_REQUEST, 0, 0, 0, lease), Status::InvalidState, "uninitialized acquire");
    Require(lease.epoch == 0 && uninitialized.availableCredits() == 0, "uninitialized outputs");
    Expect(uninitialized.Abort(), Status::InvalidState, "uninitialized abort");
    for (const uint64_t slots : {uint64_t{0}, uint64_t{1}, Pipeline::MAX_SLOT_COUNT + 1}) {
        BoundedSlotProtocol protocol;
        Expect(protocol.Init(Config(slots)), Status::InvalidConfig, "invalid slot count");
    }
    BoundedSlotProtocol invalidDomain;
    auto config = Config();
    config.domainId = 0;
    Expect(invalidDomain.Init(config), Status::InvalidConfig, "invalid domain");
    BoundedSlotProtocol capacityOverflow;
    config = Config();
    config.payloadCapacityBytes = Pipeline::MAX_EPOCH;
    Expect(capacityOverflow.Init(config), Status::InvalidConfig, "capacity sum overflow");
    BoundedSlotProtocol ringOverflow;
    config = Config();
    config.payloadCapacityBytes = Pipeline::MAX_EPOCH / Pipeline::DEFAULT_SLOT_COUNT;
    config.metadataCapacityBytes = 0;
    Expect(ringOverflow.Init(config), Status::InvalidConfig, "ring capacity overflow");
    BoundedSlotProtocol maximum;
    Expect(maximum.Init(Config(Pipeline::MAX_SLOT_COUNT)), Status::Success, "maximum slots");
    Require(maximum.availableCredits() == Pipeline::MAX_SLOT_COUNT, "maximum credits");
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "default slots");
    Require(protocol.availableCredits() == Pipeline::DEFAULT_SLOT_COUNT, "default credits");
    Expect(protocol.Init(Config()), Status::InvalidState, "must not reinitialize epochs");
    Expect(protocol.Acquire(TEST_REQUEST, 0, PAYLOAD_CAPACITY + 1, 0, lease), Status::InvalidConfig, "payload bound");
    Expect(protocol.Acquire(TEST_REQUEST, 0, 0, METADATA_CAPACITY + 1, lease), Status::InvalidConfig, "metadata bound");
    Expect(protocol.Acquire(TEST_REQUEST, 0, PAYLOAD_CAPACITY, METADATA_CAPACITY, lease), Status::Success, "acquire full capacities");
    const auto original = Snapshot(protocol, lease);
    Expect(Pipeline::ValidateControlRecord(original, TEST_DOMAIN, Pipeline::DEFAULT_SLOT_COUNT), Status::Success, "valid record");
    const auto invalid = [](const SlotControlRecord &record) {
        Expect(Pipeline::ValidateControlRecord(record, TEST_DOMAIN, Pipeline::DEFAULT_SLOT_COUNT), Status::InvalidRecord, "corrupt record accepted");
    };
    auto record = original; ++record.magic; invalid(record);
    record = original; ++record.version; invalid(record);
    record = original; ++record.fingerprint; invalid(record);
    record = original; ++record.domainId; invalid(record);
    record = original; record.slotIndex = Pipeline::DEFAULT_SLOT_COUNT; invalid(record);
    record = original; record.epoch = 0; invalid(record);
    record = original; record.state = static_cast<uint64_t>(SlotState::Consumed) + 1; invalid(record);
    record = original; record.boundaryFlags = Pipeline::ALL_BOUNDARY_FLAGS + 1; invalid(record);
    record = original; ++record.slotCount; invalid(record);
    record = original; record.reserved0 = 1; invalid(record);
    record = original; record.reserved1 = 1; invalid(record);
    record = original; record.reserved2 = 1; invalid(record);
    for (const SlotState state : {SlotState::Ready, SlotState::Reading, SlotState::Consumed}) {
        record = original; record.state = static_cast<uint64_t>(state); invalid(record);
    }
    for (const SlotState state : {SlotState::Writing, SlotState::Ready, SlotState::Reading}) {
        record = original;
        record.state = static_cast<uint64_t>(state);
        record.boundaryFlags = Pipeline::SOURCE_CONSUMED | Pipeline::REMOTE_VISIBLE | Pipeline::RECEIVER_CONSUMED;
        invalid(record);
    }
    record = original; record.boundaryFlags = Pipeline::RECEIVER_CREDIT; invalid(record);
    Expect(Pipeline::ValidateControlRecord(original, 0, Pipeline::DEFAULT_SLOT_COUNT), Status::InvalidRecord, "invalid expected domain");
    Expect(Pipeline::ValidateControlRecord(original, TEST_DOMAIN, 1), Status::InvalidRecord, "invalid expected slots");
    QualifyReady(protocol, lease);
    auto ready = Snapshot(protocol, lease);
    Expect(Pipeline::ValidateControlRecord(ready, TEST_DOMAIN, Pipeline::DEFAULT_SLOT_COUNT), Status::Success, "ready schema");
    Consume(protocol, lease);
    Release(protocol, lease);
    auto free = Snapshot(protocol, lease);
    Expect(Pipeline::ValidateControlRecord(free, TEST_DOMAIN, Pipeline::DEFAULT_SLOT_COUNT), Status::Success, "free retained epoch schema");
}
void ProofBoundaries()
{
    static_assert(!std::is_convertible<Pipeline::LocalCompletionProof,
        Pipeline::RemoteVisibilityProof>::value, "local completion cannot qualify visibility");
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "init");
    Lease lease{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, PAYLOAD_CAPACITY, METADATA_CAPACITY, lease), Status::Success, "acquire");
    const Pipeline::LocalCompletionProof local{lease, true};
    Require(local.localDone, "local proof fixture");
    Require(Snapshot(protocol, lease).boundaryFlags == 0, "local completion did not change flags");
    Expect(protocol.PublishReady(lease), Status::MissingProof, "local done cannot publish");
    Expect(protocol.RecordSourceConsumed({lease}), Status::MissingProof, "default source proof");
    Expect(protocol.RecordSourceConsumed({lease, true, false}), Status::MissingProof, "descriptor still in use");
    Expect(protocol.RecordSourceConsumed({lease, false, true}), Status::MissingProof, "payload still in use");
    Expect(protocol.RecordSourceConsumed({lease, true, true}), Status::Success, "full source proof");
    Expect(protocol.RecordSourceConsumed({lease, true, true}), Status::Duplicate, "duplicate source proof");
    Expect(protocol.PublishReady(lease), Status::MissingProof, "source reclaim is not remote visibility");
    Expect(protocol.RecordRemoteVisibility({lease}), Status::MissingProof, "default remote proof");
    Expect(protocol.RecordRemoteVisibility({lease, true, false}), Status::MissingProof, "metadata not visible");
    Expect(protocol.RecordRemoteVisibility({lease, false, true}), Status::MissingProof, "payload not visible");
    Expect(protocol.BeginRead(lease), Status::InvalidState, "reader cannot enter writing");
    Expect(protocol.RecordRemoteVisibility({lease, true, true}), Status::Success, "full remote proof");
    Expect(protocol.RecordRemoteVisibility({lease, true, true}), Status::Duplicate, "duplicate visibility");
    Expect(protocol.PublishReady(lease), Status::Success, "ready");
    Expect(protocol.PublishReady(lease), Status::InvalidState, "duplicate ready");
    Expect(protocol.ReturnCredit({lease, true}), Status::InvalidState, "ready is not consumed");
    Expect(protocol.BeginRead(lease), Status::Success, "begin read");
    Expect(protocol.CompleteRead({lease}), Status::MissingProof, "default receiver proof");
    Expect(protocol.CompleteRead({lease, true, false}), Status::MissingProof, "must capture owned backing");
    Expect(protocol.CompleteRead({lease, false, true}), Status::MissingProof, "wire reads still pending");
    Expect(protocol.CompleteRead({lease, true, true}), Status::Success, "receiver proof");
    Expect(protocol.ReturnCredit({lease}), Status::MissingProof, "default credit proof");
    Expect(protocol.RecordTaskRetired({lease}), Status::MissingProof, "default retirement proof");
    Expect(protocol.RecordTaskRetired({lease, true, false}), Status::MissingProof, "late target writes possible");
    Expect(protocol.RecordTaskRetired({lease, false, true}), Status::MissingProof, "task has not ended");
    Expect(protocol.ReturnCredit({lease, true}), Status::Success, "credit acknowledged");
    Expect(protocol.ReturnCredit({lease, true}), Status::Duplicate, "duplicate credit");
    Require(Snapshot(protocol, lease).state == static_cast<uint64_t>(SlotState::Consumed), "credit alone did not free");
    Require(protocol.availableCredits() == 1, "one slot remains occupied");
    Expect(protocol.RecordTaskRetired({lease, true, true}), Status::Success, "retired after credit");
    Require(protocol.availableCredits() == Pipeline::DEFAULT_SLOT_COUNT, "qualified return frees");
    Expect(protocol.ReturnCredit({lease, true}), Status::InvalidState, "ACK cannot replay on free");
}
void SourceOwnership()
{
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "init");
    Lease lease{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, PAYLOAD_CAPACITY, METADATA_CAPACITY, lease), Status::Success, "acquire");
    std::array<uint8_t, PAYLOAD_CAPACITY> source{};
    std::array<uint8_t, PAYLOAD_CAPACITY> wire{};
    source.fill(FIRST_PATTERN);
    std::array<uint64_t, 1> descriptor{PAYLOAD_CAPACITY};
    std::memcpy(wire.data(), source.data(), source.size());
    bool reclaimable = true;
    Expect(protocol.CanReclaimSource(lease, reclaimable), Status::Success, "reclaim query");
    Require(!reclaimable, "payload copy alone cannot release descriptor");
    Expect(protocol.RecordSourceConsumed({lease, true, false}), Status::MissingProof, "descriptor read delayed");
    Expect(protocol.CanReclaimSource(lease, reclaimable), Status::Success, "partial query");
    Require(!reclaimable && descriptor[0] == source.size(), "caller retained source and descriptor");
    const uint64_t capturedBytes = descriptor[0];
    Expect(protocol.RecordSourceConsumed({lease, true, true}), Status::Success, "both source reads ended");
    Expect(protocol.CanReclaimSource(lease, reclaimable), Status::Success, "full query");
    Require(reclaimable, "source released independently of target visibility");
    source.fill(REUSE_PATTERN);
    descriptor[0] = 0;
    Require(capturedBytes == wire.size(), "captured descriptor survives source reuse");
    for (const auto value : wire) Require(value == FIRST_PATTERN, "wire bytes survive source reuse");
    Expect(protocol.PublishReady(lease), Status::MissingProof, "source reclaim still cannot publish");
}
void DelayedCredit()
{
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "init");
    Lease first{}, delayedSender{}, next{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, PAYLOAD_CAPACITY, 0, first), Status::Success, "first");
    Expect(protocol.Acquire(TEST_REQUEST, 1, PAYLOAD_CAPACITY, 0, delayedSender), Status::Success, "second");
    Expect(protocol.Acquire(TEST_REQUEST, 2, PAYLOAD_CAPACITY, 0, next), Status::NoCredit, "bounded backpressure");
    QualifyReady(protocol, first);
    std::array<uint8_t, PAYLOAD_CAPACITY> wire{};
    wire.fill(FIRST_PATTERN);
    Expect(protocol.RecordTaskRetired({first, true, true}), Status::Success, "retired before receiver");
    Require(protocol.availableCredits() == 0, "retirement alone cannot release unread bytes");
    Expect(protocol.BeginRead(first), Status::Success, "receiver starts");
    std::array<uint8_t, PAYLOAD_CAPACITY> backing{};
    std::memcpy(backing.data(), wire.data(), wire.size());
    Expect(protocol.CompleteRead({first, true, true}), Status::Success, "owned capture");
    Expect(protocol.Acquire(TEST_REQUEST, 2, PAYLOAD_CAPACITY, 0, next), Status::NoCredit, "delayed ACK still owns wire");
    Require(Snapshot(protocol, delayedSender).state == static_cast<uint64_t>(SlotState::Writing), "sender delay independent");
    Expect(protocol.ReturnCredit({first, true}), Status::Success, "delayed ACK arrives");
    Expect(protocol.Acquire(TEST_REQUEST, 2, PAYLOAD_CAPACITY, 0, next), Status::Success, "reuse released slot");
    Require(next.slotIndex == first.slotIndex && next.epoch > delayedSender.epoch, "new incarnation");
    wire.fill(REUSE_PATTERN);
    for (const auto value : backing) Require(value == FIRST_PATTERN, "independent backing survives transport reuse");
    Expect(protocol.ReturnCredit({first, true}), Status::StaleEpoch, "late ACK cannot release reused slot");
    Require(Snapshot(protocol, next).state == static_cast<uint64_t>(SlotState::Writing), "late ACK preserved new owner");
}
void RetirementBarrier()
{
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "init");
    Lease old{}, blocker{}, next{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, PAYLOAD_CAPACITY, 0, old), Status::Success, "old request");
    Expect(protocol.Acquire(TEST_REQUEST, 1, 0, 0, blocker), Status::Success, "occupy other slot");
    QualifyReady(protocol, old);
    std::array<uint8_t, PAYLOAD_CAPACITY> wire{};
    wire.fill(FIRST_PATTERN);
    Expect(protocol.BeginRead(old), Status::Success, "begin read");
    const auto backing = wire;
    Expect(protocol.CompleteRead({old, true, true}), Status::Success, "capture");
    Expect(protocol.ReturnCredit({old, true}), Status::Success, "ACK before task retirement");
    Expect(protocol.Acquire(TEST_REQUEST + 1, 0, PAYLOAD_CAPACITY, 0, next), Status::NoCredit, "old writer still owns target");
    // A delayed old transfer may still modify its old slot until retirement.
    wire.fill(SECOND_PATTERN);
    for (const auto value : backing) Require(value == FIRST_PATTERN, "late write cannot change owned capture");
    Expect(protocol.RecordTaskRetired({old, true, false}), Status::MissingProof, "local end insufficient");
    Expect(protocol.Acquire(TEST_REQUEST + 1, 0, PAYLOAD_CAPACITY, 0, next), Status::NoCredit, "incomplete retirement cannot reuse");
    Expect(protocol.RecordTaskRetired({old, true, true}), Status::Success, "no future writes qualified");
    Expect(protocol.Acquire(TEST_REQUEST + 1, 0, PAYLOAD_CAPACITY, 0, next), Status::Success, "reuse after retirement");
    wire.fill(REUSE_PATTERN);
    Expect(protocol.RecordTaskRetired({old, true, true}), Status::StaleEpoch, "old retirement cannot affect new epoch");
    for (const auto value : wire) Require(value == REUSE_PATTERN, "new bytes preserved");
}
void LeaseIsolation()
{
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "init");
    Lease lease{}, duplicate{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, 0, 0, lease), Status::Success, "acquire");
    Expect(protocol.Acquire(TEST_REQUEST, 0, 0, 0, duplicate), Status::Duplicate, "active request chunk unique");
    Require(duplicate.epoch == 0, "failed acquire clears output");
    auto wrong = lease; ++wrong.domainId;
    Expect(protocol.RecordSourceConsumed({wrong, true, true}), Status::InvalidLease, "wrong domain");
    wrong = lease; wrong.slotIndex = Pipeline::MAX_SLOT_COUNT;
    Expect(protocol.RecordRemoteVisibility({wrong, true, true}), Status::InvalidLease, "wrong slot");
    wrong = lease; ++wrong.requestId;
    Expect(protocol.RecordTaskRetired({wrong, true, true}), Status::InvalidLease, "wrong request");
    wrong = lease; ++wrong.chunkId;
    Expect(protocol.RecordSourceConsumed({wrong, true, true}), Status::InvalidLease, "wrong chunk");
    wrong = lease; wrong.epoch = 0;
    Expect(protocol.RecordSourceConsumed({wrong, true, true}), Status::InvalidLease, "zero epoch");
    wrong = lease; ++wrong.epoch;
    Expect(protocol.RecordSourceConsumed({wrong, true, true}), Status::StaleEpoch, "future epoch");
    Require(Snapshot(protocol, lease).boundaryFlags == 0, "invalid proofs made no progress");
    QualifyReady(protocol, lease);
    Consume(protocol, lease);
    Release(protocol, lease);
    Expect(protocol.Acquire(TEST_REQUEST, 0, 0, 0, duplicate), Status::Success, "same chunk new incarnation");
    Require(duplicate.slotIndex == lease.slotIndex && duplicate.epoch > lease.epoch, "epoch prevents ABA");
    Expect(protocol.RecordRemoteVisibility({lease, true, true}), Status::StaleEpoch, "old visibility");
    Expect(protocol.CompleteRead({lease, true, true}), Status::StaleEpoch, "old completion");
    Expect(protocol.ReturnCredit({lease, true}), Status::StaleEpoch, "old credit");
    Require(Snapshot(protocol, duplicate).boundaryFlags == 0, "old proofs cannot authorize new owner");
}
void EmptyChunk()
{
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "init");
    Lease empty{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, 0, METADATA_CAPACITY, empty), Status::Success, "empty payload acquire");
    const uint64_t metadataMarker = TEST_REQUEST;
    uint64_t wireMetadata = 0;
    Expect(protocol.RecordSourceConsumed({empty, true, false}), Status::MissingProof, "empty payload descriptor still in use");
    wireMetadata = metadataMarker;
    Expect(protocol.RecordSourceConsumed({empty, true, true}), Status::Success, "empty source consumed");
    Expect(protocol.RecordRemoteVisibility({empty, true, false}), Status::MissingProof, "empty chunk still requires metadata");
    Expect(protocol.PublishReady(empty), Status::MissingProof, "no empty shortcut");
    Expect(protocol.RecordRemoteVisibility({empty, true, true}), Status::Success, "empty metadata visible");
    Expect(protocol.PublishReady(empty), Status::Success, "publish empty");
    Expect(protocol.BeginRead(empty), Status::Success, "read empty");
    const uint64_t ownedMetadata = wireMetadata;
    Expect(protocol.CompleteRead({empty, true, true}), Status::Success, "capture empty metadata");
    Release(protocol, empty);
    Require(ownedMetadata == metadataMarker, "empty message remains observable");
    Require(protocol.availableCredits() == Pipeline::DEFAULT_SLOT_COUNT, "empty chunk credits restored");
}
float AddFloat(float left, float right)
{
    volatile float rounded = left + right;
    return rounded;
}
void OrderedCapture()
{
    BoundedSlotProtocol protocol;
    OrderedCombineGate gate;
    Expect(protocol.Init(Config()), Status::Success, "init protocol");
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST, THREE_CONTRIBUTIONS), Status::Success, "init ordered combine");
    const std::array<float, THREE_CONTRIBUTIONS> logical{LARGE_FLOAT, 1.0F, -LARGE_FLOAT};
    const std::array<uint64_t, THREE_CONTRIBUTIONS> arrival{1, 2, 0};
    std::array<float, Pipeline::MAX_SLOT_COUNT> wire{};
    std::array<float, THREE_CONTRIBUTIONS> backing{};
    std::array<CaptureTicket, THREE_CONTRIBUTIONS> tickets{};
    std::array<Lease, THREE_CONTRIBUTIONS> leases{};
    // Both bounded transport slots are occupied by higher-K arrivals first.
    for (uint64_t index = 1; index < THREE_CONTRIBUTIONS; ++index) {
        Expect(protocol.Acquire(TEST_REQUEST, index, sizeof(float), 0, leases[index]), Status::Success, "fill high-K wire slot");
        wire[leases[index].slotIndex] = logical[index];
        QualifyReady(protocol, leases[index]);
    }
    Require(leases[1].slotIndex != leases[2].slotIndex && protocol.availableCredits() == 0, "both slots occupied");
    Expect(protocol.Acquire(TEST_REQUEST, 0, sizeof(float), 0, leases[0]), Status::NoCredit, "logical zero initially blocked");
    uint64_t pendingIndex = Pipeline::MAX_COMBINE_SLOTS;
    Expect(gate.BeginNextReduction(pendingIndex), Status::NotReady, "no high-K arithmetic shortcut");
    float arrivalSum = 0.0F;
    for (const auto logicalIndex : arrival) {
        if (logicalIndex == 0) {
            Expect(protocol.Acquire(TEST_REQUEST, 0, sizeof(float), 0, leases[0]), Status::Success, "missing logical zero can arrive");
            Require(leases[0].slotIndex == leases[1].slotIndex
                && leases[0].epoch > leases[2].epoch, "logical zero reuses returned credit with new epoch");
            wire[leases[0].slotIndex] = logical[0];
            QualifyReady(protocol, leases[0]);
        }
        const auto &lease = leases[logicalIndex];
        Expect(protocol.BeginRead(lease), Status::Success, "capture begins");
        Expect(gate.ReserveCapture(lease, logicalIndex, tickets[logicalIndex]), Status::Success, "reserve owned backing");
        backing[logicalIndex] = wire[lease.slotIndex];
        arrivalSum = AddFloat(arrivalSum, backing[logicalIndex]);
        Expect(gate.CaptureComplete({tickets[logicalIndex], true, true}), Status::Success, "owned capture done");
        Expect(protocol.CompleteRead({lease, true, true}), Status::Success, "transport receiver complete");
        Release(protocol, lease);
        const uint64_t expectedCredits = logicalIndex == 1 ? 1 : Pipeline::DEFAULT_SLOT_COUNT;
        Require(protocol.availableCredits() == expectedCredits, "waiting for lower K does not pin completed transport");
        wire[lease.slotIndex] = 0.0F;
        Require(backing[logicalIndex] == logical[logicalIndex], "owned backing independent of wire reuse");
        if (logicalIndex != 0) {
            uint64_t next = Pipeline::MAX_COMBINE_SLOTS;
            Expect(gate.BeginNextReduction(next), Status::NotReady, "higher arrivals cannot bypass logical zero");
            Require(gate.nextLogicalIndex() == 0, "cursor waits for logical zero");
        }
    }
    float orderedSum = 0.0F;
    for (uint64_t expected = 0; expected < THREE_CONTRIBUTIONS; ++expected) {
        uint64_t index = Pipeline::MAX_COMBINE_SLOTS;
        Expect(gate.BeginNextReduction(index), Status::Success, "start logical reduction");
        Require(index == expected && gate.nextLogicalIndex() == expected, "reduction cursor not issue advanced");
        orderedSum = AddFloat(orderedSum, backing[index]);
        Expect(gate.CompleteReduction({tickets[index], true, true}), Status::Success, "completed arithmetic");
    }
    Require(orderedSum == 0.0F && arrivalSum == 1.0F, "FP32 logical order differs from arrival order");
    Require(gate.allReduced(), "all logical contributions reduced");
    uint64_t next = 0;
    Expect(gate.BeginNextReduction(next), Status::Done, "no duplicate reduction");
}
void ReductionCompletion()
{
    OrderedCombineGate gate;
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST, 2), Status::Success, "init");
    const Lease first{TEST_DOMAIN, 0, 1, TEST_REQUEST, 0};
    const Lease second{TEST_DOMAIN, 1, 2, TEST_REQUEST, 1};
    CaptureTicket a{}, b{};
    Expect(gate.ReserveCapture(first, 0, a), Status::Success, "reserve first");
    Expect(gate.ReserveCapture(second, 1, b), Status::Success, "reserve second");
    Expect(gate.CaptureComplete({a}), Status::MissingProof, "default capture proof");
    Expect(gate.CaptureComplete({a, true, false}), Status::MissingProof, "wire reader unfinished");
    Expect(gate.CaptureComplete({a, false, true}), Status::MissingProof, "backing not ready");
    uint64_t index = 0;
    Expect(gate.BeginNextReduction(index), Status::NotReady, "cannot reduce incomplete capture");
    Expect(gate.CaptureComplete({a, true, true}), Status::Success, "first captured");
    Expect(gate.CaptureComplete({b, true, true}), Status::Success, "second captured");
    Expect(gate.BeginNextReduction(index), Status::Success, "begin first");
    Require(index == 0, "first index");
    Expect(gate.BeginNextReduction(index), Status::InvalidState, "issue cannot advance cursor");
    Expect(gate.CompleteReduction({b, true, true}), Status::InvalidState, "later K cannot finish first");
    Expect(gate.CompleteReduction({a}), Status::MissingProof, "default reduction proof");
    Expect(gate.CompleteReduction({a, true, false}), Status::MissingProof, "future backing reads possible");
    Expect(gate.CompleteReduction({a, false, true}), Status::MissingProof, "arithmetic incomplete");
    Require(gate.nextLogicalIndex() == 0 && !gate.allReduced(), "completion required to advance");
    Expect(gate.CompleteReduction({a, true, true}), Status::Success, "first finishes");
    Expect(gate.CompleteReduction({a, true, true}), Status::InvalidState, "duplicate reduction");
    Expect(gate.BeginNextReduction(index), Status::Success, "begin second");
    Require(index == 1, "second index");
    Expect(gate.CompleteReduction({b, true, true}), Status::Success, "second finishes");
    Require(gate.allReduced(), "finished all");
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST, 2), Status::WrongRequest, "cannot rewind request");
    Expect(gate.Init(TEST_DOMAIN + 1, TEST_REQUEST + 1, 2), Status::WrongRequest, "cannot change domain");
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST + 1, 2), Status::Success, "reuse backing after completion");
    Expect(gate.CaptureComplete({a, true, true}), Status::WrongRequest, "old request capture cannot recur");
}
void CombineTickets()
{
    OrderedCombineGate uninitialized;
    CaptureTicket ticket{};
    Lease lease{TEST_DOMAIN, 0, 1, TEST_REQUEST, 0};
    Expect(uninitialized.ReserveCapture(lease, 0, ticket), Status::InvalidState, "uninitialized capture");
    for (const uint64_t topk : {uint64_t{0}, Pipeline::MAX_COMBINE_SLOTS + 1}) {
        OrderedCombineGate invalid;
        Expect(invalid.Init(TEST_DOMAIN, TEST_REQUEST, topk), Status::InvalidConfig, "invalid K");
    }
    OrderedCombineGate gate;
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST, Pipeline::MAX_COMBINE_SLOTS), Status::Success, "maximum K");
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST + 1, 2), Status::InvalidState, "active backing cannot reinit");
    auto wrong = lease; ++wrong.domainId;
    Expect(gate.ReserveCapture(wrong, 0, ticket), Status::WrongRequest, "wrong capture domain");
    wrong = lease; ++wrong.requestId;
    Expect(gate.ReserveCapture(wrong, 0, ticket), Status::WrongRequest, "wrong capture request");
    wrong = lease; wrong.epoch = 0;
    Expect(gate.ReserveCapture(wrong, 0, ticket), Status::InvalidLease, "zero capture epoch");
    wrong = lease; wrong.slotIndex = Pipeline::MAX_SLOT_COUNT;
    Expect(gate.ReserveCapture(wrong, 0, ticket), Status::InvalidLease, "invalid transport slot");
    Expect(gate.ReserveCapture(lease, Pipeline::MAX_COMBINE_SLOTS, ticket), Status::WrongIndex, "capture beyond K");
    Expect(gate.ReserveCapture(lease, 0, ticket), Status::Success, "valid capture");
    CaptureTicket duplicate{};
    Expect(gate.ReserveCapture(lease, 0, duplicate), Status::Duplicate, "duplicate logical capture");
    Require(duplicate.lease.epoch == 0, "failed capture clears ticket");
    auto stale = ticket; ++stale.lease.epoch;
    Expect(gate.CaptureComplete({stale, true, true}), Status::StaleEpoch, "future capture epoch");
    stale = ticket; ++stale.lease.chunkId;
    Expect(gate.CaptureComplete({stale, true, true}), Status::StaleEpoch, "wrong capture chunk");
    stale = ticket; ++stale.logicalIndex;
    Expect(gate.CaptureComplete({stale, true, true}), Status::StaleEpoch, "ticket cannot authorize different K");
    stale = ticket; stale.logicalIndex = Pipeline::MAX_COMBINE_SLOTS;
    Expect(gate.CaptureComplete({stale, true, true}), Status::WrongIndex, "invalid completion K");
    Expect(gate.CaptureComplete({ticket, true, true}), Status::Success, "capture qualified");
    Expect(gate.CaptureComplete({ticket, true, true}), Status::InvalidState, "duplicate capture completion");
}
void AbortRecovery()
{
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config()), Status::Success, "init");
    Lease writing{}, reading{}, rejected{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, PAYLOAD_CAPACITY, METADATA_CAPACITY, writing), Status::Success, "writing lease");
    Expect(protocol.Acquire(TEST_REQUEST, 1, PAYLOAD_CAPACITY, 0, reading), Status::Success, "reading lease");
    QualifyReady(protocol, reading);
    Expect(protocol.BeginRead(reading), Status::Success, "reader in flight");
    std::array<uint8_t, PAYLOAD_CAPACITY> busyWire{};
    busyWire.fill(FIRST_PATTERN);
    const uint64_t beforeAbort = protocol.lastIssuedEpoch();
    Expect(protocol.Abort(), Status::Success, "abort");
    Require(protocol.aborted() && protocol.availableCredits() == 0, "abort withdraws credits");
    Expect(protocol.Acquire(TEST_REQUEST + 1, 0, 0, 0, rejected), Status::Aborted, "abort stops issue");
    Expect(protocol.PublishReady(writing), Status::Aborted, "abort stops publication");
    Expect(protocol.BeginRead(reading), Status::Aborted, "abort stops new readers");
    Require(Snapshot(protocol, writing).state == static_cast<uint64_t>(SlotState::Writing), "abort did not clear writer");
    Require(Snapshot(protocol, reading).state == static_cast<uint64_t>(SlotState::Reading), "abort did not clear reader");
    auto proof = Pipeline::QuiescenceProof{TEST_DOMAIN, beforeAbort, true, true, true, true, true};
    auto incomplete = proof; incomplete.descriptorReadsStopped = false;
    Expect(protocol.RecoverAfterQuiescence(incomplete), Status::MissingProof, "descriptor quiescence required");
    incomplete = proof; incomplete.sourceReadsStopped = false;
    Expect(protocol.RecoverAfterQuiescence(incomplete), Status::MissingProof, "source quiescence required");
    incomplete = proof; incomplete.targetWritesStopped = false;
    Expect(protocol.RecoverAfterQuiescence(incomplete), Status::MissingProof, "writer quiescence required");
    incomplete = proof; incomplete.receiverReadsStopped = false;
    Expect(protocol.RecoverAfterQuiescence(incomplete), Status::MissingProof, "receiver quiescence required");
    incomplete = proof; incomplete.ownedBackingAccessStopped = false;
    Expect(protocol.RecoverAfterQuiescence(incomplete), Status::MissingProof, "owned backing quiescence required");
    incomplete = proof; --incomplete.lastIssuedEpoch;
    Expect(protocol.RecoverAfterQuiescence(incomplete), Status::InvalidLease, "stale quiescence epoch");
    incomplete = proof; ++incomplete.domainId;
    Expect(protocol.RecoverAfterQuiescence(incomplete), Status::InvalidLease, "wrong quiescence domain");
    for (const auto value : busyWire) Require(value == FIRST_PATTERN, "incomplete recovery preserves busy data");
    // Existing work may drain, but even full completion cannot free while aborted.
    Expect(protocol.RecordSourceConsumed({writing, true, true}), Status::Success, "source drain during abort");
    Expect(protocol.RecordTaskRetired({writing, true, true}), Status::Success, "writer drain during abort");
    Expect(protocol.CompleteRead({reading, true, true}), Status::Success, "receiver drain during abort");
    Expect(protocol.RecordTaskRetired({reading, true, true}), Status::Success, "reader task drain");
    Expect(protocol.ReturnCredit({reading, true}), Status::Success, "ACK drain");
    Require(Snapshot(protocol, reading).state == static_cast<uint64_t>(SlotState::Consumed), "abort prevents ordinary free");
    Expect(protocol.RecoverAfterQuiescence(proof), Status::Success, "qualified global quiescence");
    Require(!protocol.aborted() && protocol.lastIssuedEpoch() == beforeAbort, "recovery preserves issuer");
    Expect(protocol.RecoverAfterQuiescence(proof), Status::InvalidState, "recovery cannot replay");
    Expect(protocol.Acquire(TEST_REQUEST + 1, 0, 0, 0, rejected), Status::Success, "new request after quiescence");
    Require(rejected.epoch == beforeAbort + 1, "recovery never rewinds epoch");
    busyWire.fill(REUSE_PATTERN);
    Expect(protocol.RecordTaskRetired({writing, true, true}), Status::StaleEpoch, "old retirement after recovery");
    Require(Snapshot(protocol, rejected).boundaryFlags == 0, "old proof cannot release new request");
}
void GateAbort()
{
    OrderedCombineGate gate;
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST, 2), Status::Success, "init");
    CaptureTicket first{}, second{};
    const Lease a{TEST_DOMAIN, 0, 1, TEST_REQUEST, 0};
    const Lease b{TEST_DOMAIN, 1, 2, TEST_REQUEST, 1};
    Expect(gate.ReserveCapture(a, 0, first), Status::Success, "reserve first");
    Expect(gate.ReserveCapture(b, 1, second), Status::Success, "reserve second");
    Expect(gate.CaptureComplete({first, true, true}), Status::Success, "capture first");
    uint64_t index = 0;
    Expect(gate.BeginNextReduction(index), Status::Success, "arithmetic in flight");
    Expect(gate.Abort(), Status::Success, "abort gate");
    Expect(gate.BeginNextReduction(index), Status::Aborted, "abort stops new arithmetic");
    CaptureTicket rejected{};
    Expect(gate.ReserveCapture(a, 0, rejected), Status::Aborted, "abort stops new capture");
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST + 1, 2), Status::InvalidState, "abort cannot free backing");
    const auto proof = Pipeline::OwnedBackingQuiescenceProof{TEST_DOMAIN, TEST_REQUEST, true, true, true};
    auto incomplete = proof; incomplete.capturesStopped = false;
    Expect(gate.QuiesceOwnedBacking(incomplete), Status::MissingProof, "capture quiescence required");
    incomplete = proof; incomplete.reductionsStopped = false;
    Expect(gate.QuiesceOwnedBacking(incomplete), Status::MissingProof, "arithmetic quiescence required");
    incomplete = proof; incomplete.backingReadsStopped = false;
    Expect(gate.QuiesceOwnedBacking(incomplete), Status::MissingProof, "backing reader quiescence required");
    incomplete = proof; ++incomplete.requestId;
    Expect(gate.QuiesceOwnedBacking(incomplete), Status::WrongRequest, "wrong backing request");
    incomplete = proof; ++incomplete.domainId;
    Expect(gate.QuiesceOwnedBacking(incomplete), Status::WrongRequest, "wrong backing domain");
    Expect(gate.CaptureComplete({second, true, true}), Status::Success, "drain existing capture");
    Expect(gate.CompleteReduction({first, true, true}), Status::Success, "drain existing arithmetic");
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST + 1, 2), Status::InvalidState, "draining alone is not quiescence");
    Expect(gate.QuiesceOwnedBacking(proof), Status::Success, "all backing access stopped");
    Expect(gate.Init(TEST_DOMAIN, TEST_REQUEST + 1, 2), Status::Success, "backing reuse after proof");
    Expect(gate.CompleteReduction({first, true, true}), Status::WrongRequest, "late arithmetic proof isolated");
}
void EpochExhaustion()
{
    BoundedSlotProtocol protocol;
    Expect(protocol.Init(Config(), Pipeline::MAX_EPOCH - 1), Status::Success, "near exhaustion init");
    Lease final{}, forbidden{};
    Expect(protocol.Acquire(TEST_REQUEST, 0, 0, 0, final), Status::Success, "maximum epoch issued once");
    Require(final.epoch == Pipeline::MAX_EPOCH, "maximum epoch no wrap");
    Expect(protocol.Acquire(TEST_REQUEST, 1, 0, 0, forbidden), Status::EpochExhausted, "cannot wrap despite free slot");
    Require(forbidden.epoch == 0, "failed max acquire clears lease");
    QualifyReady(protocol, final);
    Consume(protocol, final);
    Release(protocol, final);
    Require(protocol.availableCredits() == Pipeline::DEFAULT_SLOT_COUNT, "exhaustion separate from credit capacity");
    Expect(protocol.Acquire(TEST_REQUEST + 1, 0, 0, 0, forbidden), Status::EpochExhausted, "free slot cannot rewind epoch");
    Expect(protocol.Abort(), Status::Success, "abort exhausted protocol");
    Expect(protocol.RecoverAfterQuiescence({TEST_DOMAIN, Pipeline::MAX_EPOCH, true, true, true, true, true}), Status::Success, "recover exhausted protocol");
    Require(protocol.lastIssuedEpoch() == Pipeline::MAX_EPOCH, "recovery retains maximum epoch");
    Expect(protocol.Acquire(TEST_REQUEST + 1, 0, 0, 0, forbidden), Status::EpochExhausted, "recovery cannot reopen issuer");
}
} // namespace

int main(int argc, char **argv)
{
    Require(argc == 2, "supply one scenario");
    const std::string scenario = argv[1];
    if (scenario == "config_schema") ConfigSchema();
    else if (scenario == "proof_boundaries") ProofBoundaries();
    else if (scenario == "source_ownership") SourceOwnership();
    else if (scenario == "delayed_credit") DelayedCredit();
    else if (scenario == "retirement_barrier") RetirementBarrier();
    else if (scenario == "lease_isolation") LeaseIsolation();
    else if (scenario == "empty_chunk") EmptyChunk();
    else if (scenario == "ordered_capture") OrderedCapture();
    else if (scenario == "reduction_completion") ReductionCompletion();
    else if (scenario == "combine_tickets") CombineTickets();
    else if (scenario == "abort_recovery") AbortRecovery();
    else if (scenario == "gate_abort") GateAbort();
    else if (scenario == "epoch_exhaustion") EpochExhaustion();
    else Require(false, "unknown scenario");
    std::cout << scenario << " passed\n";
    return EXIT_SUCCESS;
}
