// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cam_async_routing_columns.h"
#include "cam_async_host_tiling.h"

namespace {
constexpr uint32_t PADDING_SENTINEL = 0xCDCDCDCD;
constexpr uint32_t UB_ALIGNMENT_BYTES = 32;
constexpr uint32_t LANE_COUNT = 128;
constexpr uint32_t SCHEDULE_COUNT = 32;
constexpr uint32_t UNEVEN_TILE_TOKENS = 17;
constexpr uint32_t ID_TILE_BYTES = 64 * 1024;
constexpr uint64_t KIB = 1024;

void Require(bool condition, const char *reason)
{
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

struct PrefixTrace {
    explicit PrefixTrace(uint32_t size)
        : values(size + 2, PADDING_SENTINEL), writes(size, 0) {}

    std::vector<uint32_t> values;
    std::vector<uint32_t> writes;
};

struct CpuBackend {
    static uint32_t ReadId(const int32_t *ids, uint64_t index)
    {
        return static_cast<uint32_t>(ids[index]);
    }

    static void AddCount(uint32_t *counts, uint64_t index, uint32_t value)
    {
        counts[index] += value;
    }

    static uint32_t ReadCount(const uint32_t *matrix, uint64_t index)
    {
        return matrix[index];
    }

    static void WritePrefix(PrefixTrace *prefix, uint64_t index, uint32_t value)
    {
        Require(index < prefix->writes.size(), "prefix write escaped its UB span");
        prefix->values[index + 1] = value;
        ++prefix->writes[index];
    }
};

struct RoutingCase {
    uint32_t ranks;
    uint32_t expertsPerRank;
    uint32_t topk;
    std::vector<uint32_t> tokensPerAiv;
    std::vector<int32_t> ids;

    uint32_t Columns() const
    {
        return ranks + ranks * expertsPerRank;
    }
};

std::vector<uint32_t> TokenFirstOracle(const RoutingCase &config)
{
    // Independent token-first reference: a set deduplicates destinations, while
    // the expert histogram records every assignment, including repeated experts.
    std::vector<uint32_t> matrix(config.tokensPerAiv.size() * config.Columns(), 0);
    uint32_t firstToken = 0;
    for (uint32_t aiv = 0; aiv < config.tokensPerAiv.size(); ++aiv) {
        for (uint32_t token = firstToken;
             token < firstToken + config.tokensPerAiv[aiv]; ++token) {
            std::set<uint32_t> destinations;
            for (uint32_t assignment = 0; assignment < config.topk; ++assignment) {
                const auto expert = static_cast<uint32_t>(config.ids[token * config.topk + assignment]);
                destinations.insert(expert / config.expertsPerRank);
                ++matrix[aiv * config.Columns() + config.ranks + expert];
            }
            for (const auto destination : destinations) {
                ++matrix[aiv * config.Columns() + destination];
            }
        }
        firstToken += config.tokensPerAiv[aiv];
    }
    return matrix;
}

std::vector<uint32_t> CountUsingProductionHelpers(
    const RoutingCase &config, uint32_t tileTokens, uint32_t seed)
{
    std::vector<uint32_t> matrix(config.tokensPerAiv.size() * config.Columns(), 0);
    std::vector<uint32_t> lanes(LANE_COUNT);
    std::iota(lanes.begin(), lanes.end(), 0);
    std::mt19937 random(seed);
    uint32_t firstToken = 0;
    for (uint32_t aiv = 0; aiv < config.tokensPerAiv.size(); ++aiv) {
        const uint32_t rowTokens = config.tokensPerAiv[aiv];
        for (uint32_t begin = 0; begin < rowTokens; begin += tileTokens) {
            const uint32_t count = std::min(tileTokens, rowTokens - begin);
            std::shuffle(lanes.begin(), lanes.end(), random);
            for (const auto lane : lanes) {
                Cam::AsyncRouting::CountTokensForLane<CpuBackend>(
                    config.ids.data() + static_cast<uint64_t>(firstToken + begin) * config.topk,
                    matrix.data() + static_cast<uint64_t>(aiv) * config.Columns(),
                    count, config.topk, config.ranks, config.expertsPerRank, lane, LANE_COUNT);
            }
        }
        firstToken += rowTokens;
    }
    return matrix;
}

std::vector<uint32_t> PrefixOracle(const RoutingCase &config,
    const std::vector<uint32_t> &raw)
{
    std::vector<uint32_t> inclusive = raw;
    for (uint32_t row = 1; row < config.tokensPerAiv.size(); ++row) {
        for (uint32_t column = 0; column < config.Columns(); ++column) {
            inclusive[row * config.Columns() + column] +=
                inclusive[(row - 1) * config.Columns() + column];
        }
    }
    return inclusive;
}

void CheckPrefixAndCombineOffsets(const RoutingCase &config,
    const std::vector<uint32_t> &raw, uint32_t seed)
{
    const uint32_t aivs = static_cast<uint32_t>(config.tokensPerAiv.size());
    const uint32_t columns = config.Columns();
    const auto before = raw;
    const auto expected = PrefixOracle(config, raw);
    std::vector<uint32_t> reconstructed = raw;
    std::vector<uint32_t> lanes(LANE_COUNT);
    std::iota(lanes.begin(), lanes.end(), 0);
    std::mt19937 random(seed);
    uint32_t columnStart = 0;
    for (uint32_t owner = 0; owner < aivs; ++owner) {
        const uint32_t columnCount = columns / aivs + (owner < columns % aivs);
        const uint32_t rowStride =
            ((columnCount * sizeof(uint32_t) + UB_ALIGNMENT_BYTES - 1) / UB_ALIGNMENT_BYTES)
            * UB_ALIGNMENT_BYTES / sizeof(uint32_t);
        PrefixTrace prefix((aivs - 1) * rowStride);
        std::shuffle(lanes.begin(), lanes.end(), random);
        for (const auto lane : lanes) {
            Cam::AsyncRouting::PrefixColumnsForLane<CpuBackend>(raw.data(), &prefix,
                aivs, columns, columnStart, columnCount, rowStride, lane, LANE_COUNT);
        }
        Require(prefix.values.front() == PADDING_SENTINEL
            && prefix.values.back() == PADDING_SENTINEL, "prefix guard overwritten");
        for (uint32_t row = 1; row < aivs; ++row) {
            for (uint32_t local = 0; local < rowStride; ++local) {
                const uint32_t index = (row - 1) * rowStride + local;
                if (local < columnCount) {
                    Require(prefix.writes[index] == 1, "prefix column has multiple lane writers");
                    Require(prefix.values[index + 1]
                        == expected[row * columns + columnStart + local], "inclusive prefix mismatch");
                    reconstructed[row * columns + columnStart + local] = prefix.values[index + 1];
                } else {
                    Require(prefix.writes[index] == 0
                        && prefix.values[index + 1] == PADDING_SENTINEL,
                        "aligned prefix padding was modified");
                }
            }
        }
        columnStart += columnCount;
    }
    Require(raw == before, "prefix helper modified the gathered raw UB matrix");
    Require(reconstructed == expected, "native scatter layout lost a prefix row");

    // CR consumes the same expert totals and exclusive AIV starts. Derive these
    // independently by summing raw earlier rows, rather than calling a helper.
    for (uint32_t expert = 0; expert < config.ranks * config.expertsPerRank; ++expert) {
        const uint32_t column = config.ranks + expert;
        uint32_t total = 0;
        for (uint32_t row = 0; row < aivs; ++row) {
            const uint32_t actualStart = row == 0 ? 0 : reconstructed[(row - 1) * columns + column];
            Require(actualStart == total, "combine-receive AIV start offset changed");
            total += raw[row * columns + column];
        }
        Require(reconstructed[(aivs - 1) * columns + column] == total,
            "combine-receive expert total changed");
    }
}

RoutingCase BuildCase(const std::string &scenario)
{
    if (scenario == "dedup") {
        return {2, 4, 3, {4}, {0, 1, 4, 2, 3, 5, 0, 4, 7, 1, 2, 3}};
    }
    if (scenario == "repeated_assignment") {
        return {2, 2, 3, {2}, {0, 0, 2, 1, 1, 3}};
    }
    if (scenario == "rank_63") {
        return {63, 2, 4, {1, 0, 1}, {0, 1, 124, 125, 125, 124, 62, 63}};
    }
    if (scenario == "ids_page_tiles" || scenario == "owner_stride_385"
        || scenario == "multi_column_300") {
        RoutingCase config{scenario == "owner_stride_385" ? 5U : 3U,
            scenario == "owner_stride_385" ? 76U : scenario == "multi_column_300" ? 99U : 4U,
            4, scenario == "owner_stride_385" ? std::vector<uint32_t>(48, 32)
                : scenario == "multi_column_300" ? std::vector<uint32_t>{257, 256}
                : std::vector<uint32_t>{20005, 0, 8193}, {}};
        const uint32_t tokens = std::accumulate(
            config.tokensPerAiv.begin(), config.tokensPerAiv.end(), uint32_t{0});
        for (uint32_t token = 0; token < tokens; ++token) {
            for (uint32_t assignment = 0; assignment < config.topk; ++assignment) {
                config.ids.push_back(static_cast<int32_t>(
                    (token + assignment * 3) % (config.ranks * config.expertsPerRank)));
            }
        }
        return config;
    }
    const bool boundary = scenario == "uint16_boundary" || scenario == "uint16_no_truncation";
    RoutingCase config{boundary ? 1U : 3U, 4, 2,
        scenario == "empty_aivs" ? std::vector<uint32_t>{0, 1, 3, 0, 2, 4, 0}
            : scenario == "tile_reuse" ? std::vector<uint32_t>{23, 0, 71, 19}
            : scenario == "single_aiv" ? std::vector<uint32_t>{9}
            : scenario == "uint16_boundary" ? std::vector<uint32_t>{32767, 0, 32768}
            : scenario == "uint16_no_truncation" ? std::vector<uint32_t>{32768, 0, 32768}
            : std::vector<uint32_t>{1, 2, 0, 4, 3}, {}};
    const uint32_t tokens = std::accumulate(
        config.tokensPerAiv.begin(), config.tokensPerAiv.end(), uint32_t{0});
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t assignment = 0; assignment < config.topk; ++assignment) {
            config.ids.push_back(static_cast<int32_t>(boundary ? assignment
                : (token + assignment * 3) % (config.ranks * config.expertsPerRank)));
        }
    }
    return config;
}

void CheckOptionalBudget()
{
    using namespace Cam::AsyncRouting;
    constexpr uint64_t activeAivs = 48;
    constexpr uint64_t topk = 8;
    const uint64_t batch = activeAivs * SIMT_MIN_TOKENS_PER_AIV;
    const uint64_t baseline = Cam::AsyncHost::DispatchSendUbBytes(
        8, 256, 7168, activeAivs, true);
    Require(!CanUseSimtRouting(192 * KIB, baseline, batch, activeAivs, topk),
        "192-KiB shape incorrectly consumes the conservative SIMT reserve");
    Require(CanUseSimtRouting(320 * KIB, baseline, batch, activeAivs, topk),
        "320-KiB quantized H7168 example should fit the optional reserve");
    const uint64_t boundary = baseline + SIMT_RESERVED_UB_BYTES;
    Require(CanUseSimtRouting(boundary, baseline, batch, activeAivs, topk),
        "exact optional UB boundary rejected");
    Require(!CanUseSimtRouting(boundary - 1, baseline, batch, activeAivs, topk),
        "insufficient optional UB did not fall back");
    Require(!CanUseSimtRouting(boundary, baseline, batch - 1, activeAivs, topk),
        "fewer than one warp of tokens per AIV admitted");
    Require(!CanUseSimtRouting(boundary, baseline, batch, activeAivs, 0),
        "zero topk admitted");
    Require(!CanUseSimtRouting(boundary, baseline, batch, activeAivs, SIMT_MAX_TOPK + 1),
        "a complete token larger than the IDs tile admitted");
    Require(!CanUseSimtRouting(boundary, baseline, batch, 0, topk), "zero AIV divisor admitted");
    Require(!CanUseSimtRouting(0, baseline, batch, activeAivs, topk), "UB underflow admitted");
    const auto maximum = std::numeric_limits<uint64_t>::max();
    Require(!CanUseSimtRouting(maximum, maximum - SIMT_RESERVED_UB_BYTES + 1,
        batch, activeAivs, topk), "UB addition overflow admitted");

    // When PreProcess dominates the original peak, an exact Count-VF budget
    // leaves no spare prefix staging. The production guard preserves the scalar
    // prefix fallback independently of the Count-VF selector.
    const uint64_t preprocessDominated = Cam::AsyncHost::DispatchSendUbBytes(
        8, 256, 32, activeAivs, false);
    const uint64_t countBoundary = preprocessDominated + SIMT_RESERVED_UB_BYTES;
    Require(CanUseSimtRouting(countBoundary, preprocessDominated, batch, activeAivs, topk),
        "Count-VF exact boundary rejected");
    Require(!CanFitPrefixStaging(countBoundary - SIMT_RESERVED_UB_BYTES,
        preprocessDominated, UB_ALIGNMENT_BYTES), "scalar prefix fallback lost");
    Require(CanFitPrefixStaging(preprocessDominated + UB_ALIGNMENT_BYTES,
        preprocessDominated, UB_ALIGNMENT_BYTES), "exact prefix staging fit rejected");
    Require(!CanFitPrefixStaging(preprocessDominated + UB_ALIGNMENT_BYTES - 1,
        preprocessDominated, UB_ALIGNMENT_BYTES), "prefix staging overflow admitted");
    Require(!CanFitPrefixStaging(0, preprocessDominated, UB_ALIGNMENT_BYTES),
        "prefix arena subtraction underflow admitted");
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "one scenario is required");
        const std::string scenario = argv[1];
        if (scenario == "budget") {
            CheckOptionalBudget();
            std::cout << scenario << " passed\n";
            return 0;
        }
        const RoutingCase config = BuildCase(scenario);
        Require(config.ranks <= Cam::AsyncRouting::MAX_MASK_RANK_COUNT,
            "scenario exceeds the admitted rank-mask bound");
        const auto expected = TokenFirstOracle(config);
        const uint32_t tileTokens = scenario == "ids_page_tiles"
            ? ID_TILE_BYTES / sizeof(int32_t) / config.topk : UNEVEN_TILE_TOKENS;
        if (scenario == "ids_page_tiles") {
            Require(tileTokens > LANE_COUNT, "IDs tile does not exercise token lane strides");
            Require(config.tokensPerAiv.front() > tileTokens * 2,
                "IDs scenario does not cross multiple complete tiles");
        }
        if (scenario == "owner_stride_385") {
            Require(config.Columns() == 385 && config.tokensPerAiv.size() == 48,
                "owner 9/8-column stride boundary scenario changed");
        }
        if (scenario == "multi_column_300") {
            Require(config.Columns() / config.tokensPerAiv.size() > LANE_COUNT,
                "prefix scenario does not exercise multiple columns per lane");
        }
        for (uint32_t seed = 0; seed < SCHEDULE_COUNT; ++seed) {
            const auto actual = CountUsingProductionHelpers(config, tileTokens, seed);
            Require(actual == expected, "rank dedup or assignment histogram mismatch");
            CheckPrefixAndCombineOffsets(config, actual, seed);
        }
        const auto inclusive = PrefixOracle(config, expected);
        const uint32_t maximum = *std::max_element(inclusive.begin(), inclusive.end());
        if (scenario == "uint16_boundary") {
            Require(maximum == std::numeric_limits<uint16_t>::max(), "compact boundary changed");
        }
        if (scenario == "uint16_no_truncation") {
            Require(maximum > std::numeric_limits<uint16_t>::max(),
                "histogram silently truncated before compact-bound validation");
        }
        std::cout << scenario << " passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
