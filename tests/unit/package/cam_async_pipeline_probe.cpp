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

#include "stable_descriptors.h"
#include "simt_descriptor_budget.h"
#include "../cam_async_ccu/combine_segments.h"

namespace {
using namespace Cam::AsyncCcu;
constexpr uint64_t ORACLE_ALIGNMENT = 32;
constexpr uint64_t ORACLE_STATISTICS_ROWS = 48;
constexpr uint64_t WINDOW_CAPACITY = 262144;
constexpr uint8_t WINDOW_GUARD = 0xA5;
constexpr uint64_t STALE_WORD = std::numeric_limits<uint64_t>::max();
constexpr uint64_t NATIVE_TILE_SLOTS = 256;
static_assert(Cam::AsyncPipeline::METADATA_HEADER_FIELDS == BATCH_INFO_FIELDS);
static_assert(sizeof(SendSegment) == sizeof(uint64_t) * 4);

void Require(bool condition, const char *reason)
{
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

struct MetadataView {
    std::vector<int64_t> values;
    std::vector<uint32_t> writes;
};

struct DescriptorView {
    SendSegment *records;
    uint32_t *writes;
    uint64_t capacity;
};

struct CpuBackend {
    static uint64_t ReadInfo(MetadataView *view, uint64_t index)
    {
        Require(index < view->values.size(), "metadata read exceeded capacity");
        Require(view->values[index] >= 0, "negative preflight-checked metadata");
        return static_cast<uint64_t>(view->values[index]);
    }

    static void WriteInfo(MetadataView *view, uint64_t index, uint64_t value)
    {
        Require(index < view->values.size(), "metadata write exceeded capacity");
        Require(value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "prefix does not fit retained metadata");
        Require(++view->writes[index] == 1, "multiple lanes own one prefix cell");
        view->values[index] = static_cast<int64_t>(value);
    }

    static void WriteDescriptor(DescriptorView *view, uint64_t slot,
        uint64_t source, uint64_t destination, uint64_t bytes, uint64_t rank)
    {
        Require(slot < view->capacity, "descriptor write exceeded tile capacity");
        Require(++view->writes[slot] == 1, "multiple lanes own one descriptor slot");
        view->records[slot] = {source, destination, bytes, rank};
    }
};

bool Equal(const SendSegment &left, const SendSegment &right)
{
    return left.srcByteOffset == right.srcByteOffset
        && left.dstByteOffset == right.dstByteOffset
        && left.byteCount == right.byteCount && left.dstRank == right.dstRank;
}

struct RowTag {
    uint32_t expert;
    uint32_t tp;
    uint32_t token;
    uint32_t assignment;
};

struct Fixture {
    CombineShape shape{32, 9, 2, 7, 3, 10};
    PlanBounds bounds{0, WINDOW_CAPACITY, 21, 3};
    std::vector<std::vector<std::vector<uint32_t>>> routes;
    std::vector<int64_t> metadata;
    std::vector<uint8_t> source;
    std::vector<std::vector<RowTag>> returned;
    uint64_t startExpert = 0;
    uint64_t endExpert = 6;

    Fixture()
    {
        GenerateRoutes();
    }

    void GenerateRoutes()
    {
        routes.assign(shape.tpSize, {});
        for (uint32_t tp = 0; tp < shape.tpSize; ++tp) {
            for (uint32_t token = 0; token < 23 - tp * 4; ++token) {
                std::vector<uint32_t> assignments;
                for (uint32_t assignment = 0; assignment < 4; ++assignment) {
                    assignments.push_back(static_cast<uint32_t>(
                        (token * 5 + assignment * 3 + tp) % (shape.moeRankNum * shape.routeExpertNumPerMoe)));
                }
                if (token % 4 == 0) {
                    assignments[1] = assignments[0];
                }
                routes[tp].push_back(assignments);
            }
        }
    }

    uint64_t PayloadBase() const
    {
        // Literal wire dimensions form an oracle independent of the planner.
        const uint64_t statistics = sizeof(uint32_t) * ORACLE_STATISTICS_ROWS
            * (shape.moeRankNum + shape.moeRankNum * shape.routeExpertNumPerMoe);
        const uint64_t flags = sizeof(uint32_t) * shape.moeRankNum;
        return (statistics + ORACLE_ALIGNMENT - 1) / ORACLE_ALIGNMENT * ORACLE_ALIGNMENT
            + (flags + ORACLE_ALIGNMENT - 1) / ORACLE_ALIGNMENT * ORACLE_ALIGNMENT;
    }

    std::vector<uint8_t> Row(const RowTag &tag) const
    {
        std::vector<uint8_t> bytes(shape.hiddenSize * sizeof(uint16_t));
        for (uint64_t column = 0; column < shape.hiddenSize; ++column) {
            const uint16_t value = static_cast<uint16_t>(
                tag.expert * 97 + tag.tp * 31 + tag.token * 11 + tag.assignment * 3 + column);
            std::memcpy(bytes.data() + column * sizeof(value), &value, sizeof(value));
        }
        return bytes;
    }

    void Rebuild()
    {
        const uint64_t owner = shape.moeRankId - shape.attnRankNum;
        const uint64_t experts = shape.routeExpertNumPerMoe;
        metadata.assign(BATCH_INFO_FIELDS + shape.tpSize + shape.tpSize * experts, 0);
        metadata[1] = static_cast<int64_t>(bounds.expectedTpGroupRank);
        metadata[2] = 7;
        metadata[3] = static_cast<int64_t>(startExpert);
        metadata[4] = static_cast<int64_t>(endExpert);
        // Build metadata token-first from raw routes, retaining duplicate top-k.
        for (uint32_t tp = 0; tp < routes.size(); ++tp) {
            for (const auto &token : routes[tp]) {
                for (const uint32_t expert : token) {
                    const uint64_t moe = expert / experts;
                    if (moe < owner) {
                        ++metadata[BATCH_INFO_FIELDS + tp];
                    }
                    if (moe == owner) {
                        ++metadata[BATCH_INFO_FIELDS + shape.tpSize + tp * experts + expert % experts];
                        ++metadata[0];
                    }
                }
            }
        }
        source.clear();
        returned.assign(shape.tpSize, {});
        // Reconstruct the expert/TP/token layout from raw IDs, without reading
        // the count table, a prefix table or any generated descriptor.
        for (uint32_t expert = 0; expert < shape.moeRankNum * experts; ++expert) {
            for (uint32_t tp = 0; tp < routes.size(); ++tp) {
                for (uint32_t token = 0; token < routes[tp].size(); ++token) {
                    for (uint32_t assignment = 0; assignment < routes[tp][token].size(); ++assignment) {
                        if (routes[tp][token][assignment] != expert) {
                            continue;
                        }
                        const RowTag tag{expert, tp, token, assignment};
                        returned[tp].push_back(tag);
                        if (expert / experts == owner && expert % experts >= startExpert
                            && expert % experts <= endExpert) {
                            const auto row = Row(tag);
                            source.insert(source.end(), row.begin(), row.end());
                        }
                    }
                }
            }
        }
        bounds.sourceCapacityBytes = source.size();
        bounds.descriptorCapacity = shape.tpSize * experts;
    }

    std::vector<std::vector<uint8_t>> Expected() const
    {
        std::vector<std::vector<uint8_t>> windows(shape.tpSize,
            std::vector<uint8_t>(bounds.windowBytes, WINDOW_GUARD));
        const uint64_t owner = shape.moeRankId - shape.attnRankNum;
        for (uint32_t tp = 0; tp < returned.size(); ++tp) {
            for (uint64_t row = 0; row < returned[tp].size(); ++row) {
                const auto &tag = returned[tp][row];
                const uint64_t localExpert = tag.expert % shape.routeExpertNumPerMoe;
                if (tag.expert / shape.routeExpertNumPerMoe != owner
                    || localExpert < startExpert || localExpert > endExpert) {
                    continue;
                }
                const auto bytes = Row(tag);
                std::copy(bytes.begin(), bytes.end(), windows[tp].begin() + PayloadBase() + row * bytes.size());
            }
        }
        return windows;
    }
};

std::vector<SendSegment> Plan(const Fixture &fixture, uint32_t lanes, uint64_t tileSlots)
{
    CombineSegmentCursor preflight;
    Require(preflight.Init(fixture.metadata.data(), fixture.metadata.size(),
        fixture.shape, fixture.bounds) == Status::Success, "valid scalar preflight failed");
    Require(lanes != 0 && tileSlots != 0, "invalid test lane/tile dimensions");
    MetadataView view{fixture.metadata, std::vector<uint32_t>(fixture.metadata.size(), 0)};
    // Reverse scheduling ensures lane order does not define prefix ownership.
    for (uint32_t lane = lanes; lane > 0; --lane) {
        Cam::AsyncPipeline::PrefixTpRowsForLane<CpuBackend>(&view, fixture.shape.tpSize,
            fixture.shape.routeExpertNumPerMoe, lane - 1, lanes);
    }
    const uint64_t countsBase = BATCH_INFO_FIELDS + fixture.shape.tpSize;
    for (uint64_t index = 0; index < countsBase; ++index) {
        Require(view.values[index] == fixture.metadata[index] && view.writes[index] == 0,
            "prefix phase changed metadata headers");
    }
    for (uint64_t tp = 0; tp < fixture.shape.tpSize; ++tp) {
        uint64_t sum = 0;
        for (uint64_t expert = 0; expert < fixture.shape.routeExpertNumPerMoe; ++expert) {
            const uint64_t index = countsBase + tp * fixture.shape.routeExpertNumPerMoe + expert;
            sum += static_cast<uint64_t>(fixture.metadata[index]);
            Require(view.values[index] == static_cast<int64_t>(sum) && view.writes[index] == 1,
                "inclusive TP prefix or lane ownership changed");
        }
    }
    const SendSegment stale{STALE_WORD, STALE_WORD, STALE_WORD, STALE_WORD};
    std::vector<SendSegment> result(preflight.descriptorSlots(), stale);
    for (uint64_t begin = 0; begin < result.size(); begin += tileSlots) {
        const uint64_t active = std::min<uint64_t>(tileSlots, result.size() - begin);
        std::vector<SendSegment> tile(tileSlots + 2, stale);
        std::vector<uint32_t> writes(tileSlots, 0);
        DescriptorView output{tile.data() + 1, writes.data(), active};
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            Cam::AsyncPipeline::DescribeSlotsForLane<CpuBackend>(&view, &output,
                fixture.shape.tpSize, fixture.shape.routeExpertNumPerMoe,
                fixture.startExpert, fixture.endExpert, fixture.shape.hiddenSize * sizeof(uint16_t),
                fixture.PayloadBase(), fixture.bounds.expectedTpGroupRank, begin, active, lane, lanes);
        }
        Require(Equal(tile.front(), stale), "descriptor tile leading guard overwritten");
        for (uint64_t slot = 0; slot < active; ++slot) {
            Require(writes[slot] == 1, "descriptor slot was not assigned exactly once");
            result[begin + slot] = tile[slot + 1];
        }
        for (uint64_t slot = active; slot < tileSlots; ++slot) {
            Require(writes[slot] == 0, "inactive tile slot was written");
        }
        for (uint64_t slot = active + 1; slot < tile.size(); ++slot) {
            Require(Equal(tile[slot], stale), "descriptor tile trailing guard overwritten");
        }
    }
    return result;
}

void CheckScatter(const Fixture &fixture, uint32_t lanes, uint64_t tileSlots)
{
    const auto slots = Plan(fixture, lanes, tileSlots);
    std::vector<std::vector<uint8_t>> windows(fixture.shape.tpSize,
        std::vector<uint8_t>(fixture.bounds.windowBytes, WINDOW_GUARD));
    for (uint64_t slot = 0; slot < slots.size(); ++slot) {
        const auto &segment = slots[slot];
        if (segment.byteCount == 0) {
            Require(Equal(segment, SendSegment{}), "empty/out-of-chunk slot retained old data");
            continue;
        }
        const uint64_t tp = slot / fixture.shape.routeExpertNumPerMoe;
        Require(segment.dstRank == fixture.bounds.expectedTpGroupRank + tp,
            "fixed TP slot targeted another peer");
        Require(segment.srcByteOffset + segment.byteCount <= fixture.source.size()
            && segment.dstByteOffset + segment.byteCount <= windows[tp].size(), "descriptor exceeded byte span");
        std::copy(fixture.source.begin() + segment.srcByteOffset,
            fixture.source.begin() + segment.srcByteOffset + segment.byteCount,
            windows[tp].begin() + segment.dstByteOffset);
    }
    Require(windows == fixture.Expected(), "raw-route byte oracle or untouched guards differ");
}

void Run(const std::string &scenario)
{
    Fixture fixture;
    if (scenario == "partial_chunk") {
        fixture.startExpert = 2;
        fixture.endExpert = 4;
    } else if (scenario == "last_chunk") {
        fixture.startExpert = 5;
    } else if (scenario == "multi_tile") {
        fixture.shape.routeExpertNumPerMoe = 97;
        fixture.endExpert = 96;
        fixture.GenerateRoutes();
    } else if (scenario == "empty_slots") {
        for (auto &tp : fixture.routes) {
            for (auto &token : tp) {
                for (auto &expert : token) {
                    expert %= static_cast<uint32_t>(fixture.shape.routeExpertNumPerMoe);
                }
            }
        }
    } else if (scenario == "sparse_peer") {
        fixture.routes[1].clear();
    }
    fixture.Rebuild();
    if (scenario == "rank_rejection") {
        fixture.metadata[1] = 0;
        const auto original = fixture.metadata;
        CombineSegmentCursor preflight;
        Require(preflight.Init(fixture.metadata.data(), fixture.metadata.size(), fixture.shape,
            fixture.bounds) == Status::TargetGroupMismatch, "mismatched fixed group was accepted");
        Require(fixture.metadata == original, "rejected preflight mutated raw metadata");
    } else if (scenario == "wide_prefixes") {
        constexpr uint64_t WIDE_PREFIX = (uint64_t{1} << 31) + 17;
        for (uint64_t tp = 0; tp < fixture.shape.tpSize; ++tp) {
            fixture.metadata[BATCH_INFO_FIELDS + tp] = static_cast<int64_t>(WIDE_PREFIX + tp);
        }
        fixture.bounds.windowBytes = (WIDE_PREFIX + 4096) * fixture.shape.hiddenSize * sizeof(uint16_t);
        const auto slots = Plan(fixture, 5, 8);
        bool sawWideOffset = false;
        for (uint64_t slot = 0; slot < slots.size(); ++slot) {
            const auto &segment = slots[slot];
            if (segment.byteCount == 0) {
                Require(Equal(segment, SendSegment{}), "empty wide-offset slot retained old data");
                continue;
            }
            const uint64_t tp = slot / fixture.shape.routeExpertNumPerMoe;
            const uint64_t expert = slot % fixture.shape.routeExpertNumPerMoe;
            uint64_t previous = 0;
            for (uint64_t prior = 0; prior < expert; ++prior) {
                previous += static_cast<uint64_t>(fixture.metadata[BATCH_INFO_FIELDS + fixture.shape.tpSize
                    + tp * fixture.shape.routeExpertNumPerMoe + prior]);
            }
            Require(segment.dstByteOffset == fixture.PayloadBase()
                + (WIDE_PREFIX + tp + previous) * fixture.shape.hiddenSize * sizeof(uint16_t),
                "wide peer prefix truncated or changed");
            Require(segment.dstRank == fixture.bounds.expectedTpGroupRank + tp, "wide offset changed target rank");
            sawWideOffset |= segment.dstByteOffset > std::numeric_limits<uint32_t>::max();
        }
        Require(sawWideOffset, "wide-prefix case did not exceed 32-bit offsets");
    } else if (scenario == "uneven_lanes") {
        for (const uint32_t lanes : {1U, 2U, 5U, 32U, 127U, 257U}) {
            CheckScatter(fixture, lanes, 8);
        }
    } else if (scenario == "bounded_slots") {
        for (const uint64_t tile : std::vector<uint64_t>{1, 3, 8, NATIVE_TILE_SLOTS}) {
            CheckScatter(fixture, 5, tile);
        }
    } else if (scenario == "budget_boundaries") {
        constexpr uint64_t EXPECTED_MIN_LAUNCH_BYTES = 139296;
        constexpr uint64_t EXPECTED_MIN_SIMT_BYTES = 213024;
        Require(!Cam::AsyncPipeline::CanLaunchDescriptorProducer(EXPECTED_MIN_LAUNCH_BYTES - 1)
            && Cam::AsyncPipeline::CanLaunchDescriptorProducer(EXPECTED_MIN_LAUNCH_BYTES),
            "status-only launch reserve boundary changed");
        Require(!Cam::AsyncPipeline::CanUseDescriptorSimt(EXPECTED_MIN_SIMT_BYTES - 1)
            && Cam::AsyncPipeline::CanUseDescriptorSimt(EXPECTED_MIN_SIMT_BYTES),
            "full SIMT arena reserve boundary changed");
        Require(!Cam::AsyncPipeline::CanLaunchDescriptorProducer(0)
            && !Cam::AsyncPipeline::CanUseDescriptorSimt(0)
            && Cam::AsyncPipeline::CanLaunchDescriptorProducer(STALE_WORD)
            && Cam::AsyncPipeline::CanUseDescriptorSimt(STALE_WORD),
            "UB budget overflow/underflow boundary changed");
    } else if (scenario == "roundtrip" || scenario == "partial_chunk" || scenario == "last_chunk"
        || scenario == "multi_tile" || scenario == "empty_slots" || scenario == "sparse_peer") {
        CheckScatter(fixture, 5, NATIVE_TILE_SLOTS);
    } else {
        throw std::runtime_error("unknown pipeline probe scenario");
    }
}
} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        return 2;
    }
    try {
        Run(argv[1]);
        std::cout << argv[1] << " passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
