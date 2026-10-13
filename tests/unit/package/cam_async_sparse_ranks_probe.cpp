// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// Execute the production rank tracker with bounded list/counter storage. The
// token-first oracle derives wire rows and expert address slots from raw IDs.
// Wire bytes are opaque fixtures; this is not a quantization or NPU simulation.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cam_async_sparse_ranks.h"

namespace {
using Cam::AsyncSparseRanks::CanUseSparseRanks;
using Cam::AsyncSparseRanks::TokenRankTracker;
using Cam::AsyncSparseRanks::Visit;
constexpr uint8_t UNTOUCHED = 0xA5;
constexpr uint8_t STALE_RANK = 0xFF;
constexpr uint32_t PAYLOAD_BYTES = 64;
constexpr uint32_t SCALE_PADDING_BYTES = 32;
constexpr uint32_t SCALE_BITS = 0x7FC12345;

void Require(bool condition, const char *reason)
{
    if (!condition) throw std::runtime_error(reason);
}

struct RankList {
    std::vector<uint8_t> entries;
    uint32_t writes{0};
    uint32_t reads{0};
};
struct Offsets {
    std::vector<uint16_t> values;
    uint32_t updates{0};
};
struct Backend {
    static void WriteRank(RankList &list, uint32_t index, uint32_t rank)
    {
        Require(index < list.entries.size() && rank <= std::numeric_limits<uint8_t>::max(),
            "visited list write exceeds existing uint8 rank arena");
        list.entries[index] = static_cast<uint8_t>(rank);
        ++list.writes;
    }
    static uint32_t ReadRank(RankList &list, uint32_t index)
    {
        Require(index < list.entries.size(), "visited list read exceeds existing arena");
        ++list.reads;
        return list.entries[index];
    }
    static void IncrementOffset(Offsets &offsets, uint32_t rank)
    {
        Require(rank < offsets.values.size(), "stale or invalid visited owner was read");
        Require(offsets.values[rank] < std::numeric_limits<uint16_t>::max(), "fixture offset overflows uint16");
        ++offsets.values[rank];
        ++offsets.updates;
    }
};

struct Payload {
    uint32_t rank;
    uint32_t token;
    uint16_t wireIndex;
    std::vector<uint8_t> bytes;
    bool operator==(const Payload &other) const
    {
        return rank == other.rank && token == other.token && wireIndex == other.wireIndex && bytes == other.bytes;
    }
};
struct Assignment {
    uint32_t token;
    uint32_t index;
    uint32_t expert;
    uint32_t rank;
    uint32_t expertSlot;
    uint16_t wireIndex;
    bool operator==(const Assignment &other) const
    {
        return token == other.token && index == other.index && expert == other.expert
            && rank == other.rank && expertSlot == other.expertSlot && wireIndex == other.wireIndex;
    }
};
struct Trace {
    std::vector<Payload> payloads;
    std::vector<Assignment> assignments;
    std::vector<std::vector<uint8_t>> addressBytes;
    std::vector<uint16_t> rankOffsets;
    std::vector<uint16_t> expertOffsets;
    uint32_t materializations{0};
};
struct Fixture {
    uint32_t ranks{3};
    uint32_t expertsPerRank{4};
    std::vector<std::vector<uint32_t>> routes{
        {0, 1, 8, 4, 9, 2, 5, 10},
        {4, 4, 5, 5, 6, 6, 7, 7},
        {8, 0, 9, 1, 10, 2, 11, 3},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {11, 7, 3, 10, 6, 2, 9, 5}};
    uint32_t begin{0};
    uint32_t end{5};
    bool quantized{false};

    std::vector<uint8_t> WireRow(uint32_t token) const
    {
        std::vector<uint8_t> row(PAYLOAD_BYTES + (quantized ? SCALE_PADDING_BYTES : 0));
        for (uint32_t byte = 0; byte < row.size(); ++byte)
            row[byte] = static_cast<uint8_t>(token * 131 + byte * 17);
        if (quantized) {
            const uint32_t scale = SCALE_BITS + token;
            std::memcpy(row.data() + PAYLOAD_BYTES, &scale, sizeof(scale));
        }
        return row;
    }
};

struct RawOracle {
    std::vector<std::vector<uint32_t>> rankTokens;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> expertAssignments;
    std::vector<uint32_t> expertPrefixes;
    std::vector<uint16_t> rankSeeds;
    std::vector<uint16_t> expertSeeds;
    Trace expected;
};

RawOracle BuildOracle(const Fixture &fixture)
{
    Require(CanUseSparseRanks(fixture.ranks + 1, 1, fixture.ranks), "fixture outside existing A5 rank domain");
    Require(fixture.expertsPerRank != 0 && fixture.begin <= fixture.end && fixture.end <= fixture.routes.size(),
        "invalid owned slice or expert shape");
    const uint32_t experts = fixture.ranks * fixture.expertsPerRank;
    RawOracle oracle;
    oracle.rankTokens.resize(fixture.ranks);
    oracle.expertAssignments.resize(experts);
    oracle.expertPrefixes.resize(experts);
    oracle.rankSeeds.assign(fixture.ranks, 0);
    oracle.expertSeeds.assign(experts, 0);
    for (uint32_t token = 0; token < fixture.routes.size(); ++token) {
        std::vector<bool> seen(fixture.ranks, false);
        for (uint32_t index = 0; index < fixture.routes[token].size(); ++index) {
            const uint32_t expert = fixture.routes[token][index];
            Require(expert < experts, "raw route outside expert domain");
            const uint32_t rank = expert / fixture.expertsPerRank;
            oracle.expertAssignments[expert].push_back({token, index});
            if (token < fixture.begin) ++oracle.expertSeeds[expert];
            if (!seen[rank]) {
                seen[rank] = true;
                const uint16_t wire = static_cast<uint16_t>(oracle.rankTokens[rank].size());
                oracle.rankTokens[rank].push_back(token);
                if (token < fixture.begin) ++oracle.rankSeeds[rank];
                if (token >= fixture.begin && token < fixture.end)
                    oracle.expected.payloads.push_back({rank, token, wire, fixture.WireRow(token)});
            }
        }
    }
    oracle.expected.addressBytes.resize(fixture.ranks);
    for (uint32_t rank = 0; rank < fixture.ranks; ++rank) {
        uint32_t prefix = 0;
        for (uint32_t local = 0; local < fixture.expertsPerRank; ++local) {
            const uint32_t expert = rank * fixture.expertsPerRank + local;
            oracle.expertPrefixes[expert] = prefix;
            prefix += static_cast<uint32_t>(oracle.expertAssignments[expert].size());
        }
        oracle.expected.addressBytes[rank].assign(prefix * sizeof(uint16_t) + SCALE_PADDING_BYTES, UNTOUCHED);
    }
    // Locate each address by its independent position in raw expert assignment
    // lists and each wire row by its position in raw unique-owner token lists.
    for (uint32_t token = fixture.begin; token < fixture.end; ++token) {
        for (uint32_t index = 0; index < fixture.routes[token].size(); ++index) {
            const uint32_t expert = fixture.routes[token][index];
            const uint32_t rank = expert / fixture.expertsPerRank;
            const auto &assignments = oracle.expertAssignments[expert];
            const auto found = std::find(assignments.begin(), assignments.end(), std::make_pair(token, index));
            const uint32_t slot = oracle.expertPrefixes[expert] + static_cast<uint32_t>(found - assignments.begin());
            const auto &tokens = oracle.rankTokens[rank];
            const uint16_t wire = static_cast<uint16_t>(std::find(tokens.begin(), tokens.end(), token) - tokens.begin());
            oracle.expected.assignments.push_back({token, index, expert, rank, slot, wire});
            std::memcpy(oracle.expected.addressBytes[rank].data() + slot * sizeof(wire), &wire, sizeof(wire));
        }
    }
    oracle.expected.rankOffsets.resize(fixture.ranks);
    oracle.expected.expertOffsets.resize(experts);
    for (uint32_t rank = 0; rank < fixture.ranks; ++rank)
        oracle.expected.rankOffsets[rank] = static_cast<uint16_t>(std::count_if(
            oracle.rankTokens[rank].begin(), oracle.rankTokens[rank].end(),
            [&](uint32_t token) { return token < fixture.end; }));
    for (uint32_t expert = 0; expert < experts; ++expert)
        oracle.expected.expertOffsets[expert] = static_cast<uint16_t>(std::count_if(
            oracle.expertAssignments[expert].begin(), oracle.expertAssignments[expert].end(),
            [&](const auto &assignment) { return assignment.first < fixture.end; }));
    oracle.expected.materializations = fixture.end - fixture.begin;
    return oracle;
}

enum class Mutation { None, EarlyOffset, MissingReset };

Trace Execute(const Fixture &fixture, const RawOracle &oracle, Mutation mutation)
{
    RankList list{std::vector<uint8_t>(fixture.ranks, STALE_RANK)};
    Offsets offsets{oracle.rankSeeds};
    auto expertOffsets = oracle.expertSeeds;
    TokenRankTracker tracker(fixture.ranks);
    Trace trace;
    trace.addressBytes = oracle.expected.addressBytes;
    for (auto &bytes : trace.addressBytes) std::fill(bytes.begin(), bytes.end(), UNTOUCHED);
    for (uint32_t token = fixture.begin; token < fixture.end; ++token) {
        if (mutation != Mutation::MissingReset || token == fixture.begin) tracker.BeginToken();
        const auto row = fixture.WireRow(token);
        ++trace.materializations;
        for (uint32_t index = 0; index < fixture.routes[token].size(); ++index) {
            const uint32_t expert = fixture.routes[token][index];
            const uint32_t rank = expert / fixture.expertsPerRank;
            const uint16_t wire = offsets.values[rank];
            const auto visit = tracker.RecordRank<Backend>(list, rank);
            Require(visit != Visit::InvalidRank, "admitted raw routing rejected");
            if (visit == Visit::First) {
                trace.payloads.push_back({rank, token, wire, row});
                if (mutation == Mutation::EarlyOffset) Backend::IncrementOffset(offsets, rank);
            }
            const uint32_t slot = oracle.expertPrefixes[expert] + expertOffsets[expert];
            Require((static_cast<uint64_t>(slot) + 1) * sizeof(wire) <= trace.addressBytes[rank].size(),
                "expert assignment exceeds metadata capacity");
            trace.assignments.push_back({token, index, expert, rank, slot, wire});
            std::memcpy(trace.addressBytes[rank].data() + slot * sizeof(wire), &wire, sizeof(wire));
            ++expertOffsets[expert];
        }
        if (mutation != Mutation::EarlyOffset) tracker.AdvanceOffsets<Backend>(list, offsets);
    }
    trace.rankOffsets = offsets.values;
    trace.expertOffsets = expertOffsets;
    if (mutation == Mutation::None) {
        Require(list.writes == trace.payloads.size() && list.reads == trace.payloads.size()
            && offsets.updates == trace.payloads.size(), "work is not bounded by actual visited owners");
    }
    return trace;
}

bool SameTrace(const Trace &actual, const Trace &expected)
{
    return actual.payloads == expected.payloads && actual.assignments == expected.assignments
        && actual.addressBytes == expected.addressBytes && actual.rankOffsets == expected.rankOffsets
        && actual.expertOffsets == expected.expertOffsets && actual.materializations == expected.materializations;
}

void Run(const std::string &scenario)
{
    Fixture fixture;
    if (scenario == "quantized_wire") fixture.quantized = true;
    if (scenario == "owned_slice") { fixture.begin = 2; fixture.end = 4; }
    if (scenario == "empty_slice") fixture.begin = fixture.end;
    if (scenario == "rank_one") {
        fixture.ranks = 1;
        fixture.expertsPerRank = 8;
        fixture.routes.assign(3, {0, 1, 1, 2, 2, 3, 3, 4});
        fixture.end = 3;
    }
    if (scenario == "k_more_ranks") {
        fixture.ranks = 2;
        fixture.expertsPerRank = 8;
        fixture.routes.assign(3, {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13});
        fixture.end = 3;
    }
    if (scenario == "stale_tail") {
        fixture.ranks = 7;
        fixture.expertsPerRank = 2;
        fixture.routes = {{0, 2, 4, 6, 8, 10, 12}, {12, 13, 12, 13, 12, 13, 12}, {4, 5, 4, 5, 4, 5, 4}};
        fixture.end = 3;
    }
    if (scenario == "rank_62" || scenario == "full_rank_set") {
        fixture.ranks = 63;
        fixture.expertsPerRank = 2;
        fixture.routes.assign(3, {});
        fixture.end = 3;
        for (auto &routes : fixture.routes) {
            if (scenario == "rank_62") routes = {124, 0, 125, 1, 124, 2, 125, 3};
            else {
                for (uint32_t rank = 0; rank < fixture.ranks; ++rank) routes.push_back(rank * fixture.expertsPerRank);
                for (uint32_t rank = 0; rank < 7; ++rank) routes.push_back(rank * fixture.expertsPerRank + 1);
            }
        }
    }
    if (scenario == "mask_rank_63") {
        RankList list{std::vector<uint8_t>(64, STALE_RANK)};
        Offsets offsets{std::vector<uint16_t>(64, 0)};
        TokenRankTracker tracker(64);
        tracker.BeginToken();
        for (uint32_t rank = 0; rank < 64; ++rank)
            Require(tracker.RecordRank<Backend>(list, rank) == Visit::First, "64-bit mask boundary rejected");
        Require(tracker.RecordRank<Backend>(list, 63) == Visit::Repeated && tracker.visitedCount() == 64,
            "high bit shift or duplicate list capacity changed");
        tracker.AdvanceOffsets<Backend>(list, offsets);
        Require(list.writes == 64 && list.reads == 64 && offsets.updates == 64,
            "full mask or invalid rank changed list work");
        for (const uint16_t offset : offsets.values) Require(offset == 1, "full-mask owner advanced incorrectly");
        Require(!CanUseSparseRanks(65, 1, 64), "portable bit 63 adds an A5 shape");
        return;
    }
    if (scenario == "invalid_rank") {
        RankList list{std::vector<uint8_t>(64, STALE_RANK)};
        TokenRankTracker tracker(64);
        tracker.BeginToken();
        Require(tracker.RecordRank<Backend>(list, 64) == Visit::InvalidRank
            && tracker.RecordRank<Backend>(list, std::numeric_limits<uint32_t>::max()) == Visit::InvalidRank
            && tracker.visitedCount() == 0 && list.writes == 0,
            "invalid rank reaches a shift or list write");
        Require(tracker.RecordRank<Backend>(list, 63) == Visit::First && tracker.visitedCount() == 1,
            "invalid visit changed a later valid mask operation");
        for (const uint32_t count : {0U, 65U}) {
            TokenRankTracker invalid(count);
            Require(invalid.RecordRank<Backend>(list, 0) == Visit::InvalidRank && invalid.visitedCount() == 0,
                "invalid rank count touched list");
        }
        return;
    }
    if (scenario == "invalid_config") {
        Require(CanUseSparseRanks(2, 1, 1) && CanUseSparseRanks(64, 1, 63), "existing A5 endpoints rejected");
        Require(!CanUseSparseRanks(1, 1, 0) && !CanUseSparseRanks(64, 0, 64)
            && !CanUseSparseRanks(65, 1, 64) && !CanUseSparseRanks(64, 1, 62)
            && !CanUseSparseRanks(std::numeric_limits<uint32_t>::max(), 1, 63),
            "sparse path widens existing A5 shape admission");
        return;
    }
    if (scenario == "uint16_boundary") {
        const uint16_t initial = std::numeric_limits<uint16_t>::max() - 1;
        RankList list{std::vector<uint8_t>(1, STALE_RANK)};
        Offsets offsets{std::vector<uint16_t>(1, initial)};
        TokenRankTracker tracker(1);
        tracker.BeginToken();
        Require(tracker.RecordRank<Backend>(list, 0) == Visit::First
            && tracker.RecordRank<Backend>(list, 0) == Visit::Repeated
            && offsets.values[0] == initial, "same-token wire offset advanced before token end");
        tracker.AdvanceOffsets<Backend>(list, offsets);
        Require(offsets.values[0] == std::numeric_limits<uint16_t>::max()
            && list.writes == 1 && list.reads == 1 && offsets.updates == 1,
            "valid uint16 endpoint or once-per-token increment changed");
        return;
    }
    if (scenario == "invalid_slice") {
        bool rejected = false;
        fixture.begin = fixture.end + 1;
        try { BuildOracle(fixture); } catch (const std::runtime_error &) { rejected = true; }
        Require(rejected, "backwards owned slice accepted");
        rejected = false;
        fixture.begin = 0;
        fixture.end = static_cast<uint32_t>(fixture.routes.size()) + 1;
        try { BuildOracle(fixture); } catch (const std::runtime_error &) { rejected = true; }
        Require(rejected, "owned slice beyond raw input accepted");
        return;
    }
    Require(scenario == "roundtrip" || scenario == "quantized_wire" || scenario == "owned_slice"
        || scenario == "empty_slice" || scenario == "rank_one" || scenario == "rank_62" || scenario == "full_rank_set"
        || scenario == "k_more_ranks" || scenario == "stale_tail" || scenario == "early_advance_mutation"
        || scenario == "missing_reset_mutation", "unknown sparse rank scenario");
    const auto oracle = BuildOracle(fixture);
    const Mutation mutation = scenario == "early_advance_mutation" ? Mutation::EarlyOffset
        : scenario == "missing_reset_mutation" ? Mutation::MissingReset : Mutation::None;
    const auto actual = Execute(fixture, oracle, mutation);
    if (mutation == Mutation::None) Require(SameTrace(actual, oracle.expected), "raw routing/byte/offset oracle differs");
    else Require(!SameTrace(actual, oracle.expected), "unsafe token counter mutation was not detected");
}
} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    try {
        Run(argv[1]);
        std::cout << argv[1] << " passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
