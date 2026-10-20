// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cam_async_a5_contract.h"
#include "cam_async_combine_segments.h"

namespace {
using namespace Cam::AsyncA5;
using namespace Cam::AsyncPipeline;
constexpr uint64_t WINDOW_BYTES = 1ULL << 20;

void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
RuntimeArgs Args()
{
    RuntimeArgs args;
    args.domain = 17;
    args.windowBytes = WINDOW_BYTES;
    args.slotCount = 2;
    args.localRank = 0;
    args.attentionRanks = 4;
    args.moeRanks = 2;
    args.tpSize = 2;
    args.expertsPerRank = 2;
    args.hiddenSize = 32;
    args.topk = 2;
    args.epoch = 1;
    return args;
}
RingLayout Layout(const RuntimeArgs &args)
{
    RingLayout layout;
    Require(PlanLayout(args.windowBytes, args.attentionRanks + args.moeRanks,
        args.slotCount, args.domain, layout), "layout rejected");
    return layout;
}

void Geometry()
{
    for (uint64_t ranks : {uint64_t{2}, uint64_t{6}, uint64_t{64}}) {
        for (uint64_t slots : {uint64_t{2}, uint64_t{3}, uint64_t{8}}) {
            RingLayout layout;
            Require(PlanLayout(WINDOW_BYTES, ranks, slots, 19, layout), "valid geometry");
            std::vector<uint8_t> ownership(WINDOW_BYTES, 0);
            for (uint64_t slot = 0; slot < slots; ++slot) {
                const auto frame = FrameOffset(layout, slot);
                Require(frame % 128 == 0 && layout.payloadCapacity % 128 == 0, "frame alignment");
                Require(frame + layout.payloadCapacity <= WINDOW_BYTES, "frame bounds");
                for (uint64_t byte = frame; byte < frame + layout.payloadCapacity; ++byte) {
                    Require(ownership[byte] == 0, "overlapping frames");
                    ownership[byte] = 1;
                }
                for (uint64_t rank = 0; rank < ranks; ++rank) {
                    for (auto role : {RecordRole::ProducerReady, RecordRole::ReceiverCredit}) {
                        const auto offset = WriterRecordOffset(layout, slot, rank, role);
                        Require(offset % 128 == 0 && offset + 128 <= WINDOW_BYTES, "control bounds");
                        for (uint64_t byte = offset; byte < offset + 128; ++byte) {
                            Require(ownership[byte] == 0, "overlapping control or payload");
                            ownership[byte] = 2;
                        }
                    }
                }
            }
        }
    }
}

void Bounds()
{
    RingLayout layout;
    for (uint64_t ranks : {uint64_t{0}, uint64_t{65}})
        Require(!PlanLayout(WINDOW_BYTES, ranks, 2, 1, layout), "rank bound");
    for (uint64_t slots : {uint64_t{0}, uint64_t{1}, uint64_t{9}, UINT64_MAX})
        Require(!PlanLayout(WINDOW_BYTES, 4, slots, 1, layout), "slot bound");
    Require(!PlanLayout(1024, 64, 8, 1, layout), "undersized window");
    Require(!PlanLayout(WINDOW_BYTES, 4, 2, 0, layout), "zero domain");
    Require(PlanLayout(UINT64_MAX, 64, 8, 1, layout), "large valid geometry");
    Require(UINT64_MAX - layout.totalBytes < layout.slotCount * 128, "checked full span");
    auto args = Args();
    Require(ValidateArgs(args), "valid context");
    args.tpSize = 3;
    Require(!ValidateArgs(args), "invalid TP");
}

void Schema()
{
    const auto args = Args();
    const auto layout = Layout(args);
    auto record = FrameRecord(args, layout, SlotState::Ready, SOURCE_CONSUMED | REMOTE_VISIBLE | TASK_RETIRED);
    record.reserved2 = args.epoch;
    Require(FrameRecordMatches(record, args, layout, args.epoch, args.slot), "ready schema");
    auto bad = record;
    bad.magic = PROTOCOL_MAGIC;
    Require(!FrameRecordMatches(bad, args, layout, args.epoch, args.slot), "reference wire must not mix");
    bad = record;
    bad.fingerprint ^= 1;
    Require(!FrameRecordMatches(bad, args, layout, args.epoch, args.slot), "fingerprint");
    bad = record;
    bad.boundaryFlags &= ~REMOTE_VISIBLE;
    Require(!FrameRecordMatches(bad, args, layout, args.epoch, args.slot), "missing remote visibility");
    bad = record;
    ++bad.payloadBytes;
    Require(!FrameRecordMatches(bad, args, layout, args.epoch, args.slot), "payload extent");
    Require(!FrameRecordMatches(record, args, layout, args.epoch + 1, args.slot), "stale epoch");
    Require(!FrameRecordMatches(record, args, layout, args.epoch, args.slot + 1), "wrong slot");
}

void Publication()
{
    const auto args = Args();
    auto record = FrameRecord(args, Layout(args), SlotState::Ready, SOURCE_CONSUMED | REMOTE_VISIBLE);
    Require(!CommitSampleMatches(0, record, 0), "unpublished body");
    record.reserved2 = args.epoch;
    Require(CommitSampleMatches(args.epoch, record, args.epoch), "stable commit");
    Require(!CommitSampleMatches(args.epoch, record, 0), "writer invalidated during snapshot");
    Require(!CommitSampleMatches(args.epoch, record, args.epoch + 1), "new epoch during snapshot");
    ++record.epoch;
    Require(!CommitSampleMatches(args.epoch, record, args.epoch), "torn body");
}

void Credit()
{
    auto args = Args();
    const auto layout = Layout(args);
    BoundedSlotProtocol reference;
    Require(reference.Init({args.domain, args.slotCount, layout.payloadCapacity, 0}) == Status::Success, "reference init");
    Lease lease;
    Require(reference.Acquire(1, 0, layout.payloadCapacity, 0, lease) == Status::Success, "acquire");
    Require(reference.RecordSourceConsumed({lease, true, true}) == Status::Success, "source boundary");
    Require(reference.RecordRemoteVisibility({lease, true, true}) == Status::Success, "remote boundary");
    Require(reference.PublishReady(lease) == Status::Success, "publish");
    Require(reference.BeginRead(lease) == Status::Success, "read");
    Require(reference.CompleteRead({lease, true, true}) == Status::Success, "consume");
    Require(reference.ReturnCredit({lease, true}) == Status::Success, "credit");
    Require(reference.availableCredits() == 1, "credit cannot retire a still-writing task");
    Require(reference.RecordTaskRetired({lease, true, true}) == Status::Success, "retire");
    Require(reference.availableCredits() == 2, "credit and retirement free slot");
    auto wire = FrameRecord(args, layout, SlotState::Consumed, ALL_BOUNDARY_FLAGS);
    wire.reserved2 = args.epoch;
    Require(FrameRecordMatches(wire, args, layout, lease.epoch, lease.slotIndex), "runtime credit identity");
    ++wire.requestId;
    Require(!FrameRecordMatches(wire, args, layout, lease.epoch, lease.slotIndex), "wrong request credit");
    Require(reference.Abort() == Status::Success, "abort");
    Require(reference.Acquire(2, 0, 0, 0, lease) == Status::Aborted, "no post-abort reuse");
}

void Descriptors()
{
    auto args = Args();
    args.slot = 1;
    args.localRank = 4;
    args.attentionGroup = 2;
    const auto layout = Layout(args);
    // Expert-major source: e0 has TP counts [2,1], e1 has [0,2].
    // Per-source earlier-MoE prefixes [3,5] produce independent destinations.
    const std::vector<int64_t> info{5, 2, 0, 0, 1, 3, 5, 2, 0, 1, 2};
    Cam::AsyncCcu::CombineSegmentCursor cursor;
    Require(cursor.Init(info.data(), info.size(), {32, 4, 2, 2, 2, 4},
        {5 * 64, layout.payloadCapacity, 8, 2}) == Cam::AsyncCcu::Status::Success, "plan");
    std::vector<Cam::AsyncCcu::SendSegment> descriptors(8);
    Cam::AsyncCcu::SendSegment segment;
    uint64_t index = 0;
    while (cursor.Next(segment, index) == Cam::AsyncCcu::Status::Success) {
        segment.dstByteOffset += FrameOffset(layout, args.slot);
        descriptors[args.attentionGroup * args.expertsPerRank + index] = segment;
    }
    const uint64_t payload = FrameOffset(layout, 1) + 4 * (2 + 4) * 48 + 32;
    Require(descriptors[4].srcByteOffset == 0 && descriptors[4].dstByteOffset == payload + 3 * 64
        && descriptors[4].byteCount == 2 * 64 && descriptors[4].dstRank == 2, "first peer segment");
    Require(descriptors[6].srcByteOffset == 2 * 64 && descriptors[6].dstByteOffset == payload + 5 * 64
        && descriptors[6].byteCount == 64 && descriptors[6].dstRank == 3, "second peer segment");
    Require(descriptors[7].srcByteOffset == 3 * 64 && descriptors[7].dstByteOffset == payload + 6 * 64
        && descriptors[7].byteCount == 2 * 64, "second expert placement");
    for (uint64_t slot : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{5}})
        Require(descriptors[slot].byteCount == 0, "unselected peer/expert is empty");
}

void Tags()
{
    Require(!IsFullCall(0), "legacy mode");
    for (uint64_t slots = 2; slots <= 8; ++slots)
        for (uint64_t slot = 0; slot < slots; ++slot) {
            const auto tag = EncodeCall(Mode::Dispatch, slots, slot);
            Require(IsFullCall(tag) && CallMode(tag) == Mode::Dispatch
                && CallSlots(tag) == slots && CallSlot(tag) == slot, "call-tag roundtrip");
        }
}

void Multiplication()
{
    std::mt19937_64 random(42391);
    const std::vector<uint64_t> edge{0, 1, 2, UINT32_MAX, uint64_t{1} << 32,
        (uint64_t{1} << 32) + 1, uint64_t{1} << 63, UINT64_MAX - 1, UINT64_MAX};
    const auto check = [](uint64_t left, uint64_t right) {
        const __uint128_t expected = static_cast<__uint128_t>(left) * right;
        uint64_t actual = 0;
        const bool fits = Cam::AsyncArithmetic::CheckedMultiply64(left, right, actual);
        Require(fits == (expected <= UINT64_MAX), "multiply overflow decision");
        if (fits) Require(actual == static_cast<uint64_t>(expected), "multiply value");
    };
    for (const auto left : edge) for (const auto right : edge) check(left, right);
    for (uint32_t trial = 0; trial < 10000; ++trial) {
        check(random(), random());
        check(random(), random() & UINT32_MAX);
        check(random() & UINT32_MAX, random() & UINT32_MAX);
    }
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "scenario required");
        const std::string scenario = argv[1];
        if (scenario == "geometry") Geometry();
        else if (scenario == "bounds") Bounds();
        else if (scenario == "schema") Schema();
        else if (scenario == "publication") Publication();
        else if (scenario == "credit") Credit();
        else if (scenario == "descriptors") Descriptors();
        else if (scenario == "tags") Tags();
        else if (scenario == "multiply") Multiplication();
        else throw std::runtime_error("unknown scenario");
        std::cout << scenario << " passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
