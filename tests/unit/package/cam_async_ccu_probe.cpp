// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "combine_segments.h"

namespace {
using namespace Cam::AsyncCcu;
constexpr uint8_t UNTOUCHED = 0xA5;

void Require(bool condition, const char *reason)
{
    if (!condition) throw std::runtime_error(reason);
}

struct Tag {
    uint32_t expert;
    uint32_t tp;
    uint32_t token;
    uint32_t assignment;
};

struct Fixture {
    CombineShape shape{32, 4, 2, 4, 2, 5};
    PlanBounds bounds{0, 4096, 8, 2};
    std::vector<std::vector<std::vector<uint32_t>>> routes{
        {{0,1,4},{1,2,5},{0,2,7},{2,3,4},{0,3,6}},
        {{3,4,5},{0,1,7},{1,2,6}}};
    std::vector<int64_t> metadata;
    std::vector<uint8_t> source;
    std::vector<std::vector<Tag>> allReturned;
    uint64_t start = 0;
    uint64_t end = 3;

    uint64_t PayloadBase() const
    {
        // Fixed wire contract, independent of the cursor's offset implementation.
        const uint64_t statistics = 4 * (shape.moeRankNum + shape.moeRankNum
            * shape.routeExpertNumPerMoe) * 48;
        return ((statistics + 31) / 32) * 32
            + ((4 * shape.moeRankNum + 31) / 32) * 32;
    }

    std::vector<uint8_t> Row(const Tag &tag) const
    {
        std::vector<uint8_t> bytes(shape.hiddenSize * 2);
        for (uint64_t column = 0; column < shape.hiddenSize; ++column) {
            const uint16_t value = static_cast<uint16_t>(0x1000 + tag.expert * 97
                + tag.tp * 31 + tag.token * 11 + tag.assignment * 3 + column);
            std::memcpy(bytes.data() + column * 2, &value, sizeof(value));
        }
        return bytes;
    }

    void Rebuild()
    {
        const uint64_t owner = shape.moeRankId - shape.attnRankNum;
        metadata.assign(5 + shape.tpSize + shape.tpSize * shape.routeExpertNumPerMoe, 0);
        metadata[1] = static_cast<int64_t>(bounds.expectedTpGroupRank);
        metadata[2] = 7;
        metadata[3] = static_cast<int64_t>(start);
        metadata[4] = static_cast<int64_t>(end);
        // A token-first independent oracle derives compact metadata from raw IDs.
        for (uint32_t tp = 0; tp < routes.size(); ++tp) {
            for (uint32_t token = 0; token < routes[tp].size(); ++token) {
                for (const uint32_t expert : routes[tp][token]) {
                    const uint64_t moe = expert / shape.routeExpertNumPerMoe;
                    if (moe < owner) ++metadata[5 + tp];
                    if (moe == owner) {
                        ++metadata[5 + shape.tpSize + tp * shape.routeExpertNumPerMoe
                            + expert % shape.routeExpertNumPerMoe];
                        ++metadata[0];
                    }
                }
            }
        }
        source.clear();
        allReturned.assign(shape.tpSize, {});
        // Expert order comes from the mathematical MoE return layout. Each list
        // is rebuilt from raw routes, never from counts or a produced descriptor.
        for (uint32_t expert = 0;
             expert < shape.moeRankNum * shape.routeExpertNumPerMoe; ++expert) {
            for (uint32_t tp = 0; tp < routes.size(); ++tp) {
                for (uint32_t token = 0; token < routes[tp].size(); ++token) {
                    for (uint32_t slot = 0; slot < routes[tp][token].size(); ++slot) {
                        if (routes[tp][token][slot] != expert) continue;
                        Tag tag{expert, tp, token, slot};
                        allReturned[tp].push_back(tag);
                        if (expert / shape.routeExpertNumPerMoe == owner
                            && expert % shape.routeExpertNumPerMoe >= start
                            && expert % shape.routeExpertNumPerMoe <= end) {
                            const auto bytes = Row(tag);
                            source.insert(source.end(), bytes.begin(), bytes.end());
                        }
                    }
                }
            }
        }
        bounds.sourceCapacityBytes = source.size();
    }

    std::vector<std::vector<uint8_t>> Expected() const
    {
        std::vector<std::vector<uint8_t>> expected(shape.tpSize,
            std::vector<uint8_t>(bounds.windowBytes, UNTOUCHED));
        const uint64_t owner = shape.moeRankId - shape.attnRankNum;
        for (uint32_t tp = 0; tp < allReturned.size(); ++tp) {
            for (uint64_t row = 0; row < allReturned[tp].size(); ++row) {
                const auto &tag = allReturned[tp][row];
                if (tag.expert / shape.routeExpertNumPerMoe != owner
                    || tag.expert % shape.routeExpertNumPerMoe < start
                    || tag.expert % shape.routeExpertNumPerMoe > end) continue;
                const auto bytes = Row(tag);
                std::copy(bytes.begin(), bytes.end(),
                    expected[tp].begin() + PayloadBase() + row * bytes.size());
            }
        }
        return expected;
    }
};

std::vector<SendSegment> Plan(Fixture &fixture, const RowInterval *interval = nullptr)
{
    CombineSegmentCursor plan;
    Require(plan.Init(fixture.metadata.data(), fixture.metadata.size(), fixture.shape,
        fixture.bounds, interval) == Status::Success, "valid compact plan rejected");
    Require(plan.descriptorSlots() == fixture.shape.tpSize * fixture.shape.routeExpertNumPerMoe,
        "fixed launch slot count changed");
    std::vector<SendSegment> slots(plan.descriptorSlots(), SendSegment{});
    uint64_t count = 0;
    for (;;) {
        SendSegment segment{};
        uint64_t slot = 0;
        const Status status = plan.Next(segment, slot);
        if (status == Status::Done) break;
        Require(status == Status::Success, "validated plan failed during iteration");
        Require(slot < slots.size() && slots[slot].byteCount == 0, "duplicate/out-of-range slot");
        Require(MatchesFixedPeer(segment,
            fixture.bounds.expectedTpGroupRank + slot / fixture.shape.routeExpertNumPerMoe),
            "nonzero descriptor targets a different fixed channel");
        slots[slot] = segment;
        ++count;
    }
    Require(count == plan.segmentCount(), "preflight segment count mismatch");
    for (uint64_t tp = 0; tp < fixture.shape.tpSize; ++tp) {
        ReadyNotification ready{};
        const Status status = plan.ReadyForTp(tp, ready);
        if (fixture.end + 1 == fixture.shape.routeExpertNumPerMoe) {
            Require(status == Status::Success, "empty/final participant lost ready description");
            Require(ready.dstRank == fixture.bounds.expectedTpGroupRank + tp
                && ready.value == static_cast<uint64_t>(fixture.metadata[5 + tp]) + 1,
                "prefix+1 ready contract changed");
            const uint64_t expectedFlag = fixture.PayloadBase()
                - ((4 * fixture.shape.moeRankNum + 31) / 32) * 32
                + 4 * (fixture.shape.moeRankId - fixture.shape.attnRankNum);
            Require(ready.flagByteOffset == expectedFlag, "ready flag index changed");
        } else {
            Require(status == Status::NoFinalReady, "non-final chunk publishes completion");
        }
    }
    return slots;
}

void Scatter(const Fixture &fixture, const std::vector<SendSegment> &slots,
    std::vector<std::vector<uint8_t>> &windows)
{
    for (const auto &segment : slots) {
        if (segment.byteCount == 0) {
            Require(segment.srcByteOffset == 0 && segment.dstByteOffset == 0 && segment.dstRank == 0,
                "unused fixed slot is not fully zero");
            continue;
        }
        const uint64_t tp = segment.dstRank - fixture.bounds.expectedTpGroupRank;
        Require(tp < windows.size(), "bad target TP rank");
        Require(segment.srcByteOffset + segment.byteCount <= fixture.source.size()
            && segment.dstByteOffset + segment.byteCount <= windows[tp].size(), "bad byte span");
        std::copy(fixture.source.begin() + segment.srcByteOffset,
            fixture.source.begin() + segment.srcByteOffset + segment.byteCount,
            windows[tp].begin() + segment.dstByteOffset);
    }
}

void CheckOracle(const std::string &scenario)
{
    Fixture fixture;
    if (scenario == "partial_chunk") { fixture.start = 1; fixture.end = 2; }
    if (scenario == "last_chunk") fixture.start = 2;
    if (scenario == "empty_final" || scenario == "empty_nonfinal") {
        for (auto &rank : fixture.routes) {
            for (uint32_t token = 0; token < rank.size(); ++token) {
                for (uint32_t slot = 0; slot < rank[token].size(); ++slot)
                    rank[token][slot] = (token + slot) % 4;
            }
        }
        if (scenario == "empty_nonfinal") fixture.end = 1;
    }
    fixture.Rebuild();
    std::vector<std::vector<uint8_t>> windows(fixture.shape.tpSize,
        std::vector<uint8_t>(fixture.bounds.windowBytes, UNTOUCHED));
    if (scenario == "split_aivs") {
        const uint64_t rows = fixture.source.size() / (fixture.shape.hiddenSize * 2);
        for (const uint64_t cores : {3ULL, 48ULL}) {
            windows.assign(fixture.shape.tpSize,
                std::vector<uint8_t>(fixture.bounds.windowBytes, UNTOUCHED));
            for (uint64_t aiv = cores; aiv > 0; --aiv) {
                RowInterval interval{};
                Require(PartitionRows(rows, aiv - 1, cores, interval) == Status::Success,
                    "original AIV partition rejected");
                Scatter(fixture, Plan(fixture, &interval), windows);
            }
            Require(windows == fixture.Expected(), "AIV intersections lost/duplicated routed rows");
        }
    } else {
        const auto slots = Plan(fixture);
        Scatter(fixture, slots, windows);
        Require(windows == fixture.Expected(), "CCU spans changed expert/TP/token byte layout");
        if (scenario == "empty_final" || scenario == "empty_nonfinal")
            Require(fixture.source.empty(), "empty scenario has a payload");
    }
}

Status Initialize(Fixture &fixture)
{
    CombineSegmentCursor plan;
    const Status status = plan.Init(fixture.metadata.data(), fixture.metadata.size(),
        fixture.shape, fixture.bounds);
    if (status != Status::Success) {
        SendSegment output{1, 2, 3, 4};
        uint64_t slot = 99;
        Require(plan.Next(output, slot) == status && output.byteCount == 0
            && output.srcByteOffset == 0 && output.dstByteOffset == 0 && output.dstRank == 0,
            "invalid plan leaked wrapped/stale descriptor fields");
    }
    return status;
}

void CheckRejections()
{
    Fixture original;
    original.Rebuild();
    auto value = original;
    value.bounds.sourceCapacityBytes -= 1;
    Require(Initialize(value) == Status::SourceCapacityExceeded, "source bounds not enforced");
    value = original;
    value.bounds.descriptorCapacity -= 1;
    Require(Initialize(value) == Status::DescriptorCapacityExceeded,
        "fixed slots must fit even when several descriptors are empty");
    value = original;
    value.bounds.windowBytes = original.PayloadBase() - 1;
    Require(Initialize(value) == Status::WindowCapacityExceeded, "ready region can exceed window");
    value = original;
    value.bounds.windowBytes = original.PayloadBase() + 1;
    Require(Initialize(value) == Status::WindowCapacityExceeded, "payload can exceed window");
    value = original;
    value.metadata[1] = 0;
    Require(Initialize(value) == Status::TargetGroupMismatch, "fixed target group guard is absent");
    value = original;
    value.metadata.back() = static_cast<int64_t>(MAX_SOURCE_EXPERT_ROWS + 1);
    Require(Initialize(value) == Status::InvalidMetadata, "uint16 source/expert counter overflow admitted");
    value = original;
    value.metadata.back() = -1;
    Require(Initialize(value) == Status::InvalidMetadata, "negative device count admitted");
    value = original;
    ++value.metadata[0];
    Require(Initialize(value) == Status::InvalidMetadata, "whole-group count mismatch admitted");
    value = original;
    value.metadata[3] = 3; value.metadata[4] = 2;
    Require(Initialize(value) == Status::InvalidMetadata, "inverted expert interval admitted");
    value = original;
    value.metadata[5] = static_cast<int64_t>(MAX_U32_VALUE);
    Require(Initialize(value) == Status::InvalidMetadata, "ready+1 wraps uint32");
    value = original;
    value.shape.hiddenSize = 33;
    Require(Initialize(value) == Status::InvalidShape, "unaligned hidden admitted");
    value = original;
    value.shape.tpSize = 3;
    Require(Initialize(value) == Status::InvalidShape, "nondividing TP admitted");
    value = original;
    value.shape.moeRankId = 0;
    Require(Initialize(value) == Status::InvalidShape, "Attention actor admitted as MoE sender");
    value = original;
    value.metadata.pop_back();
    Require(Initialize(value) == Status::InvalidMetadata, "noncompact metadata length admitted");
    value = original;
    RowInterval invalid{1, 0};
    CombineSegmentCursor plan;
    Require(plan.Init(value.metadata.data(), value.metadata.size(), value.shape, value.bounds,
        &invalid) == Status::InvalidShape, "inverted AIV slice admitted");
    invalid = {0, value.source.size() / (value.shape.hiddenSize * 2) + 1};
    Require(plan.Init(value.metadata.data(), value.metadata.size(), value.shape, value.bounds,
        &invalid) == Status::InvalidShape, "AIV slice escapes the selected chunk");
    RowInterval interval{};
    Require(PartitionRows(5, 0, 0, interval) == Status::InvalidShape,
        "zero-core division admitted");
    Require(PartitionRows(5, 48, 48, interval) == Status::InvalidShape,
        "bad AIV index admitted");
    Require(!MatchesFixedPeer(SendSegment{0, 32, 64, 2}, 3), "wrong peer accepted");
}

void CheckWideAndOverflow()
{
    const CombineShape shape{64, 2, 2, 1, 2, 3};
    std::vector<int64_t> info{2, 0, 0, 0, 0, 40000000, 50000000, 1, 1};
    PlanBounds bounds{256, 8000000000ULL, 2, 0};
    CombineSegmentCursor plan;
    Require(plan.Init(info.data(), info.size(), shape, bounds) == Status::Success,
        "valid 64-bit target offset rejected");
    SendSegment first{};
    uint64_t slot;
    Require(plan.Next(first, slot) == Status::Success && first.dstByteOffset > MAX_U32_VALUE,
        "target byte offset silently narrowed");
    const CombineShape huge{MAX_U32_VALUE - 31, 2, 2, 1, 2, 3};
    info[5] = static_cast<int64_t>(MAX_U32_VALUE - 1);
    bounds.sourceCapacityBytes = MAX_U64_VALUE;
    bounds.windowBytes = MAX_U64_VALUE;
    Require(plan.Init(info.data(), info.size(), huge, bounds) == Status::ArithmeticOverflow,
        "destination multiplication overflow admitted");
    uint64_t output;
    Require(!CheckedAdd(MAX_U64_VALUE, 1, output)
        && !CheckedMultiply(MAX_U64_VALUE, 2, output)
        && !CheckedAlign(MAX_U64_VALUE, output), "checked arithmetic silently wraps");
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "one scenario is required");
        const std::string scenario = argv[1];
        if (scenario == "rejections") CheckRejections();
        else if (scenario == "wide_offsets") CheckWideAndOverflow();
        else CheckOracle(scenario);
        std::cout << scenario << " passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
