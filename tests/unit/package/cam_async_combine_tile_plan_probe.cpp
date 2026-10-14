// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// Execute production tile planning, route capture, cursor and event leases.
// Independent raw routing builds wire rows and numerical goldens. SDK names in
// the CPU fixture are declarations only; no NPU memory or performance is modeled.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cam_async_combine_tile_plan.h"

namespace {
using Event = AscendC::HardEvent;
using Bytes = std::vector<uint8_t>;
constexpr uint32_t LOAD_PIPE = 0;
constexpr uint32_t VECTOR_PIPE = 1;
constexpr uint32_t STORE_PIPE = 2;
constexpr uint32_t PIPE_COUNT = 3;
constexpr uint32_t SCHEDULE_COUNT = 32;
constexpr uint32_t RANK_COUNT = 3;
constexpr uint32_t EXPERTS_PER_RANK = 4;
constexpr uint32_t EXPERT_COUNT = RANK_COUNT * EXPERTS_PER_RANK;
constexpr uint32_t TOKEN_BEGIN = 3;
constexpr uint32_t OWNED_TOKENS = 3;
constexpr uint32_t TEST_HIDDEN = 7168;
constexpr uint32_t WIDE_HIDDEN = 12288;
constexpr uint64_t SOURCE_WINDOW = uint64_t{1} << 34;
constexpr uint64_t WIRE_HEADER = 32;
constexpr uint64_t NO_VERSION = 0;
constexpr float POISON_WEIGHT = 19.0F;

void Require(bool condition, const char *reason)
{
    if (!condition) throw std::runtime_error(reason);
}

uint32_t SourcePipe(Event event)
{
    if (event == Event::V_MTE2 || event == Event::V_MTE3 || event == Event::V_S) return VECTOR_PIPE;
    if (event == Event::MTE3_V || event == Event::MTE3_MTE2) return STORE_PIPE;
    return LOAD_PIPE;
}
uint32_t DestinationPipe(Event event)
{
    if (event == Event::MTE2_V || event == Event::MTE3_V) return VECTOR_PIPE;
    if (event == Event::V_MTE3 || event == Event::MTE2_MTE3) return STORE_PIPE;
    return LOAD_PIPE;
}

enum class Action { Work, Signal, Wait };
struct Command {
    Action action;
    Event event{Event::MTE2_S};
    int id{0};
    std::function<void()> work{};
};

// Three independently scheduled FIFO engines. Scalar WaitFlag(MTE2_S) blocks
// the issuing scalar until that engine reaches its fence; it is not a fourth
// DMA queue. No work is considered complete when the scalar merely enqueues it.
class Machine {
public:
    explicit Machine(uint32_t seed) : random_(seed) {}
    void Work(uint32_t pipe, std::function<void()> work)
    {
        queues_[pipe].push_back({Action::Work, Event::MTE2_S, 0, std::move(work)});
    }
    int Allocate(Event event)
    {
        int id = 2;
        while (leases_[{event, id}]) ++id;
        leases_[{event, id}] = true;
        return id;
    }
    void Signal(Event event, int id)
    {
        queues_[SourcePipe(event)].push_back({Action::Signal, event, id, {}});
    }
    void Wait(Event event, int id)
    {
        queues_[DestinationPipe(event)].push_back({Action::Wait, event, id, {}});
    }
    void FenceToScalar()
    {
        Signal(Event::MTE2_S, 0);
        while (tokens_[{Event::MTE2_S, 0}] == 0) Step();
        tokens_[{Event::MTE2_S, 0}] = 0;
    }
    void FenceVectorToScalar()
    {
        Signal(Event::V_S, 0);
        while (tokens_[{Event::V_S, 0}] == 0) Step();
        tokens_[{Event::V_S, 0}] = 0;
        ++vectorScalarWaits_;
    }
    uint32_t VectorScalarWaits() const { return vectorScalarWaits_; }
    bool HasQueuedVectorWork() const
    {
        for (const auto &command : queues_[VECTOR_PIPE])
            if (command.action == Action::Work) return true;
        return false;
    }
    // Test scheduler observation only: advance exactly one engine command so a
    // fault can be injected between multiplication and its pending Add consumer.
    void ProgressOneCommand() { Step(); }
    void Release(Event event, int id)
    {
        Require(leases_[{event, id}], "release without a live event lease");
        Require(tokens_[{event, id}] == 0, "release before token consumption");
        for (const auto &queue : queues_) {
            for (const auto &command : queue) {
                Require(command.action == Action::Work || command.event != event || command.id != id,
                    "release before a queued event operation completes");
            }
        }
        leases_[{event, id}] = false;
    }
    void Finish()
    {
        while (Pending()) Step();
        for (const auto &entry : tokens_) Require(entry.second == 0, "unconsumed event token");
        for (const auto &entry : leases_) Require(!entry.second, "event lease survives drain");
    }
    bool SawOverlap() const { return overlap_; }
    void ObserveLoad()
    {
        overlap_ = overlap_ || !queues_[VECTOR_PIPE].empty() || !queues_[STORE_PIPE].empty();
    }
private:
    bool Pending() const
    {
        for (const auto &queue : queues_) if (!queue.empty()) return true;
        return false;
    }
    void Step()
    {
        std::vector<uint32_t> eligible;
        for (uint32_t pipe = 0; pipe < PIPE_COUNT; ++pipe) {
            if (!queues_[pipe].empty()) {
                const auto &command = queues_[pipe].front();
                if (command.action != Action::Wait || tokens_[{command.event, command.id}] != 0)
                    eligible.push_back(pipe);
            }
        }
        Require(!eligible.empty(), "deadlock or an unmatched readiness/credit wait");
        const uint32_t pipe = eligible[random_() % eligible.size()];
        Command command = std::move(queues_[pipe].front());
        queues_[pipe].pop_front();
        auto &token = tokens_[{command.event, command.id}];
        if (command.action == Action::Work) command.work();
        else if (command.action == Action::Signal) {
            Require(token == 0, "event signaled twice without consumption");
            token = 1;
        } else {
            Require(token == 1, "wait without an event token");
            token = 0;
        }
    }
    std::mt19937 random_;
    std::array<std::deque<Command>, PIPE_COUNT> queues_{};
    std::map<std::pair<Event, int>, bool> leases_{};
    std::map<std::pair<Event, int>, uint32_t> tokens_{};
    bool overlap_{false};
    uint32_t vectorScalarWaits_{0};
};

class Backend {
public:
    using EventId = int;
    explicit Backend(Machine *machine) : machine_(machine) {}
    void Init(AscendC::TPipe *) {}
    template <Event event> int Allocate() { return machine_->Allocate(event); }
    template <Event event> void Signal(int id) { machine_->Signal(event, id); }
    template <Event event> void Wait(int id) { machine_->Wait(event, id); }
    template <Event event> void Release(int id) { machine_->Release(event, id); }
    void FenceToScalar() { machine_->FenceToScalar(); }
private:
    Machine *machine_;
};

uint32_t FloatBits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
float BitsFloat(uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
uint32_t RoundRight(uint32_t value, uint32_t shift)
{
    const uint32_t base = value >> shift;
    const uint32_t tail = value & ((uint32_t{1} << shift) - 1);
    const uint32_t half = uint32_t{1} << (shift - 1);
    return base + (tail > half || (tail == half && (base & 1)));
}
enum class Dtype { Fp16, Bf16 };
uint16_t Encode(float value, Dtype dtype)
{
    const uint32_t bits = FloatBits(value);
    if (dtype == Dtype::Bf16) return static_cast<uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t mantissa = bits & 0x7FFFFFU;
    int exponent = static_cast<int>((bits >> 23) & 0xFFU) - 127 + 15;
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7C00U);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        return static_cast<uint16_t>(sign | RoundRight(mantissa | 0x800000U, static_cast<uint32_t>(14 - exponent)));
    }
    uint32_t rounded = RoundRight(mantissa, 13);
    if (rounded == 1024) { rounded = 0; ++exponent; }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | rounded);
}
float Decode(uint16_t value, Dtype dtype)
{
    if (dtype == Dtype::Bf16) return BitsFloat(static_cast<uint32_t>(value) << 16);
    const uint32_t exponent = (value >> 10) & 31U;
    const uint32_t mantissa = value & 1023U;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -24)
        : std::ldexp(static_cast<float>(1024 + mantissa), static_cast<int>(exponent) - 25);
    return value & 0x8000U ? -magnitude : magnitude;
}
void PutWord(Bytes &bytes, uint32_t element, uint16_t value)
{
    bytes[element * sizeof(value)] = static_cast<uint8_t>(value);
    bytes[element * sizeof(value) + 1] = static_cast<uint8_t>(value >> 8);
}
uint16_t GetWord(const Bytes &bytes, uint32_t element)
{
    return static_cast<uint16_t>(bytes[element * sizeof(uint16_t)]
        | (static_cast<uint16_t>(bytes[element * sizeof(uint16_t) + 1]) << 8));
}

// Golden multiplication and addition use double temporaries and explicit FP32
// rounding at EACH operation. The simulated vector work below uses separate
// volatile float operations. The cancellation fixture distinguishes both from
// FMA, including after final FP16/BF16 nearest-even conversion.
float GoldenAdd(float sum, float input, float scale)
{
    const float product = static_cast<float>(static_cast<double>(input) * scale);
    return static_cast<float>(static_cast<double>(sum) + product);
}


uint64_t legacy_baseline(uint32_t hidden, uint64_t prefix)
{
    uint64_t floats, inputs;
    using namespace Cam::AsyncCombineTiles;
    Require(aligned_product(hidden, sizeof(float), floats), "float alignment");
    Require(aligned_product(hidden, sizeof(uint16_t), inputs), "input alignment");
    return TILE_FLOAT_ROWS * floats + inputs + prefix;
}
void planning()
{
    using namespace Cam::AsyncCombineTiles;
    uint64_t result;
    Require(!aligned_product(MAX_TILE_VALUE, sizeof(uint64_t), result), "descriptor multiplication overflow");
    for (const uint32_t hidden : {TEST_HIDDEN, WIDE_HIDDEN}) {
        const uint64_t baseline = legacy_baseline(hidden, WIRE_HEADER);
        for (const uint32_t topk : {uint32_t{1}, uint32_t{8}, uint32_t{512}}) {
            const auto plan = plan_tile_scratch(baseline, hidden, sizeof(uint16_t), topk, 2, WIRE_HEADER);
            Require(plan.enabled && plan.tile_elements < hidden && plan.total_bytes <= baseline,
                "exact legacy minimum lost a feasible tile plan");
            Require(plan.float_row_bytes == static_cast<uint64_t>(plan.tile_elements) * sizeof(float),
                "FP32 scratch retained full hidden length");
            Require(plan.address_bytes >= static_cast<uint64_t>(topk) * sizeof(uint64_t)
                && plan.scale_bytes >= static_cast<uint64_t>(topk) * sizeof(float), "descriptor capacity truncates K");
            std::vector<std::pair<uint64_t, uint64_t>> zones{{plan.address_offset, plan.address_bytes},
                {plan.scale_offset, plan.scale_bytes}, {plan.prefix_offset, plan.prefix_bytes},
                {plan.row_offset, plan.float_row_bytes}, {plan.product_offset, plan.float_row_bytes},
                {plan.sum_offset, plan.float_row_bytes}, {plan.input_offset[0], plan.input_row_bytes},
                {plan.input_offset[1], plan.input_row_bytes}};
            std::sort(zones.begin(), zones.end());
            uint64_t previous_end = 0;
            for (const auto &zone : zones) {
                Require(zone.first % TILE_ALIGNMENT_BYTES == 0 && zone.first >= previous_end,
                    "tile UB allocations overlap or lose alignment");
                previous_end = zone.first + zone.second;
            }
            Require(previous_end == plan.total_bytes, "plan omits an owned allocation");
        }
        Require(!plan_tile_scratch(baseline - 1, hidden, 2, 8, 2, WIRE_HEADER).enabled,
            "new plan relaxed baseline minimum");
        Require(!plan_tile_scratch(baseline, hidden, 2, 32768, 2, WIRE_HEADER).enabled,
            "oversized descriptor table must select existing path");
        const auto roomy = plan_tile_scratch(baseline + hidden * sizeof(uint16_t) + 1024,
            hidden, 2, 8, 2, WIRE_HEADER);
        Require(!roomy.enabled, "full-row double buffering gained needless descriptor overhead");
    }
    Require(!plan_tile_scratch(200000, TEST_HIDDEN, 2, 0, 2, WIRE_HEADER).enabled, "zero K");
    Require(!plan_tile_scratch(200000, TEST_HIDDEN, 2, 8, 0, WIRE_HEADER).enabled, "empty owner");
    Require(!plan_tile_scratch(200000, TEST_HIDDEN, 3, 8, 2, WIRE_HEADER).enabled, "invalid wire type");
}

struct Fixture {
    uint32_t hidden;
    uint32_t topk;
    Dtype dtype;
    std::vector<uint32_t> routes;
    std::vector<float> weights;
    std::array<uint32_t, EXPERT_COUNT> counts{};
    std::array<uint32_t, EXPERT_COUNT> expert_prefix{};
    std::array<uint32_t, RANK_COUNT> rank_prefix{};
    std::vector<uint64_t> oracle_addresses;
    std::map<uint64_t, Bytes> wire;
    std::vector<Bytes> golden;
    Fixture(uint32_t hidden_elements, uint32_t logical_count, Dtype value_type)
        : hidden(hidden_elements), topk(logical_count), dtype(value_type)
    {
        const uint32_t contributions = (TOKEN_BEGIN + OWNED_TOKENS) * topk;
        routes.resize(contributions);
        weights.resize(contributions);
        oracle_addresses.resize(contributions);
        std::array<std::vector<uint32_t>, EXPERT_COUNT> grouped;
        // Raw routing deliberately repeats an expert inside top-k. Those are
        // separate occurrence rows and must never be collapsed by expert ID.
        for (uint32_t index = 0; index < contributions; ++index) {
            routes[index] = ((index / 2) * 5 + (index / topk) / 2) % EXPERT_COUNT;
            weights[index] = index % 2 ? -0.375F : 0.75F;
            grouped[routes[index]].push_back(index);
            ++counts[routes[index]];
        }
        if (topk > 1) {
            const float input = dtype == Dtype::Fp16 ? 1025.0F : 1032.0F;
            const float scale = std::nextafter(1.0F, 2.0F);
            weights[TOKEN_BEGIN * topk] = -static_cast<float>(static_cast<double>(input) * scale);
            weights[TOKEN_BEGIN * topk + 1] = scale;
        }
        uint32_t physical_row = 0;
        for (uint32_t rank = 0; rank < RANK_COUNT; ++rank) {
            rank_prefix[rank] = physical_row;
            uint32_t local_prefix = 0;
            for (uint32_t local_expert = 0; local_expert < EXPERTS_PER_RANK; ++local_expert) {
                const uint32_t expert = rank * EXPERTS_PER_RANK + local_expert;
                expert_prefix[expert] = local_prefix;
                // Independent token-first occurrence lists establish physical
                // wire placement, without calling the production address helper.
                for (const uint32_t original : grouped[expert]) {
                    const uint64_t address = SOURCE_WINDOW + WIRE_HEADER
                        + static_cast<uint64_t>(physical_row) * hidden * sizeof(uint16_t);
                    Bytes bytes(hidden * sizeof(uint16_t));
                    for (uint32_t element = 0; element < hidden; ++element) {
                        float value = (static_cast<int>(element % 13) - 6) * 0.25F + (original % 5) * 0.5F;
                        if (topk > 1 && original == TOKEN_BEGIN * topk) value = 1.0F;
                        if (topk > 1 && original == TOKEN_BEGIN * topk + 1)
                            value = dtype == Dtype::Fp16 ? 1025.0F : 1032.0F;
                        PutWord(bytes, element, Encode(value, dtype));
                    }
                    oracle_addresses[original] = address;
                    wire.emplace(address, std::move(bytes));
                    ++physical_row;
                }
                local_prefix += counts[expert];
            }
        }
        golden.assign(OWNED_TOKENS, Bytes(hidden * sizeof(uint16_t)));
        for (uint32_t token = TOKEN_BEGIN; token < TOKEN_BEGIN + OWNED_TOKENS; ++token) {
            for (uint32_t element = 0; element < hidden; ++element) {
                float sum = 0.0F;
                for (uint32_t logical = 0; logical < topk; ++logical) {
                    const uint32_t contribution = token * topk + logical;
                    const auto &bytes = wire.at(oracle_addresses[contribution]);
                    sum = GoldenAdd(sum, Decode(GetWord(bytes, element), dtype), weights[contribution]);
                }
                PutWord(golden[token - TOKEN_BEGIN], element, Encode(sum, dtype));
            }
        }
    }
};
struct Routes {
    Fixture *fixture;
    std::array<uint32_t, EXPERT_COUNT> occurrences{};
    std::vector<uint64_t> addresses;
    std::vector<float> scales;
    explicit Routes(Fixture *source) : fixture(source), addresses(source->topk), scales(source->topk) {}
    uint32_t route_expert(uint32_t contribution) { return fixture->routes.at(contribution); }
    uint32_t take_occurrence(uint32_t expert) { return occurrences[expert]++; }
    uint32_t rank_prefix(uint32_t expert) { return fixture->rank_prefix[expert / EXPERTS_PER_RANK]; }
    uint32_t expert_prefix(uint32_t expert) { return fixture->expert_prefix[expert]; }
    float weight(uint32_t contribution) { return fixture->weights.at(contribution); }
    void store_descriptor(uint32_t logical, uint64_t address, float scale)
    {
        addresses[logical] = address;
        scales[logical] = scale;
    }
};
struct Slot { Bytes bytes{}; uint64_t address{0}; uint64_t version{NO_VERSION}; };
struct LoadArgs { uint64_t address; uint32_t offset; uint32_t count; };

void run_tiled(const std::string &scenario, uint32_t seed)
{
    using namespace Cam::AsyncCombineTiles;
    const uint32_t hidden = scenario == "wide_hidden" ? WIDE_HIDDEN
        : scenario == "odd_tail" ? TEST_HIDDEN + 1 : TEST_HIDDEN;
    const uint32_t topk = scenario == "topk_one" ? 1 : scenario == "topk_eight" ? 8 : 2;
    Fixture fixture(hidden, topk, scenario == "bf16" ? Dtype::Bf16 : Dtype::Fp16);
    const auto plan = plan_tile_scratch(legacy_baseline(hidden, WIRE_HEADER), hidden, 2,
        topk, OWNED_TOKENS, WIRE_HEADER);
    Require(plan.enabled, "fixture did not enter production tile path");
    Routes routes(&fixture);
    std::array<uint32_t, EXPERT_COUNT> expected_occurrences{};
    for (uint32_t index = 0; index < TOKEN_BEGIN * topk; ++index) ++expected_occurrences[fixture.routes[index]];
    if (scenario == "prefix") routes.occurrences = expected_occurrences;
    else for (uint32_t index = 0; index < TOKEN_BEGIN * topk; ++index)
        routes.take_occurrence(routes.route_expert(index));
    Machine machine(seed);
    Backend backend(&machine);
    Cam::AsyncCombine::PipelineEvents<Backend> pipeline(backend);
    pipeline.Init(nullptr);
    std::array<Slot, TILE_INPUT_SLOTS> slots{};
    // One shared tile-sized scratch triplet is reused across all K/tiles/tokens.
    std::vector<float> row(plan.tile_elements), product(plan.tile_elements), sum(plan.tile_elements);
    std::vector<uint64_t> row_version(plan.tile_elements), product_version(plan.tile_elements);
    std::vector<uint32_t> next_k(plan.tile_elements);
    std::vector<Bytes> output(OWNED_TOKENS, Bytes(hidden * sizeof(uint16_t), 0xA5));
    LoadArgs live_load{};
    float live_scale = 0.0F;
    uint64_t issued_version = 0;
    for (uint32_t token = TOKEN_BEGIN; token < TOKEN_BEGIN + OWNED_TOKENS; ++token) {
        capture_token_sources(token, topk, hidden, 2, SOURCE_WINDOW, WIRE_HEADER, routes);
        for (uint32_t logical = 0; logical < topk; ++logical) {
            const uint32_t contribution = token * topk + logical;
            Require(routes.addresses[logical] == fixture.oracle_addresses[contribution]
                && FloatBits(routes.scales[logical]) == FloatBits(fixture.weights[contribution]),
                "captured descriptor lost occurrence or scale bits");
            ++expected_occurrences[fixture.routes[contribution]];
        }
        HiddenKCursor cursor;
        cursor.init(hidden, plan.tile_elements, topk);
        std::vector<HiddenKTile> tiles;
        std::vector<uint32_t> coverage(static_cast<size_t>(hidden) * topk);
        uint32_t expected_logical = 0;
        uint32_t expected_hidden = 0;
        while (cursor.has_next()) {
            const auto tile = cursor.current();
            Require(tile.logical_index == expected_logical && tile.hidden_offset == expected_hidden,
                "cursor is not hidden-first logical-K order");
            for (uint32_t element = tile.hidden_offset; element < tile.hidden_offset + tile.hidden_elements; ++element)
                ++coverage[static_cast<size_t>(tile.logical_index) * hidden + element];
            tiles.push_back(tile);
            ++expected_logical;
            if (expected_logical == topk) { expected_logical = 0; expected_hidden += tile.hidden_elements; }
            cursor.advance();
        }
        for (const auto count : coverage) Require(count == 1, "cursor repeated or omitted an element");
        if (scenario == "reverse_k" && topk > 1) std::swap(tiles[0], tiles[1]);
        auto load = [&](uint32_t slot, const HiddenKTile &tile, bool acquire_credit) {
            if (acquire_credit) pipeline.WaitForReuse(slot);
            live_load = {routes.addresses[tile.logical_index], tile.hidden_offset, tile.hidden_elements};
            const LoadArgs captured = live_load;
            machine.Work(LOAD_PIPE, [&, slot, captured] {
                const auto args = scenario == "late_dma" ? live_load : captured;
                const auto &wire = fixture.wire.at(args.address);
                machine.ObserveLoad();
                slots[slot].bytes.assign(wire.begin() + args.offset * sizeof(uint16_t),
                    wire.begin() + (args.offset + args.count) * sizeof(uint16_t));
                slots[slot].address = args.address;
                slots[slot].version = NO_VERSION;
            });
            pipeline.MarkLoaded(slot);
        };
        load(0, tiles[0], false);
        for (size_t index = 0; index < tiles.size(); ++index) {
            const uint32_t slot = static_cast<uint32_t>(index % TILE_INPUT_SLOTS);
            const auto tile = tiles[index];
            const uint32_t contribution = token * topk + tile.logical_index;
            const float weight = routes.scales[tile.logical_index];
            live_scale = weight;
            const uint64_t version = ++issued_version;
            pipeline.WaitLoaded(slot);
            if (index + 1 < tiles.size()) load((slot + 1) % TILE_INPUT_SLOTS, tiles[index + 1], true);
            if (tile.logical_index == 0) machine.Work(VECTOR_PIPE, [&, tile] {
                std::fill_n(sum.begin(), tile.hidden_elements, 0.0F);
                std::fill_n(next_k.begin(), tile.hidden_elements, 0);
            });
            machine.Work(VECTOR_PIPE, [&, slot, tile, contribution, version] {
                const auto &expected = fixture.wire.at(fixture.oracle_addresses[contribution]);
                Require(slots[slot].address == fixture.oracle_addresses[contribution]
                    && slots[slot].bytes.size() == tile.hidden_elements * sizeof(uint16_t)
                    && std::equal(slots[slot].bytes.begin(), slots[slot].bytes.end(),
                        expected.begin() + tile.hidden_offset * sizeof(uint16_t)), "Cast used late or overwritten DMA arguments");
                for (uint32_t local = 0; local < tile.hidden_elements; ++local) {
                    row[local] = Decode(GetWord(slots[slot].bytes, local), fixture.dtype);
                    row_version[local] = version;
                }
            });
            // Independent operator jobs correspond to the retained PIPE_V fences.
            machine.Work(VECTOR_PIPE, [&, tile, weight, version] {
                const float operand = scenario == "late_scale" ? live_scale : weight;
                for (uint32_t local = 0; local < tile.hidden_elements; ++local) {
                    Require(row_version[local] == version, "row scratch reused before Muls");
                    volatile float rounded = row[local] * operand;
                    product[local] = rounded;
                    product_version[local] = version;
                }
            });
            machine.Work(VECTOR_PIPE, [&, tile, weight, version] {
                for (uint32_t local = 0; local < tile.hidden_elements; ++local) {
                    Require(product_version[local] == version && next_k[local] == tile.logical_index,
                        "shared tile scratch or logical K order changed");
                    if (scenario == "fused") sum[local] = std::fma(row[local], weight, sum[local]);
                    else { volatile float rounded = sum[local] + product[local]; sum[local] = rounded; }
                    ++next_k[local];
                }
            });
            if (tile.logical_index + 1 == topk) {
                machine.Work(VECTOR_PIPE, [&, slot, tile, version] {
                    Bytes converted(tile.hidden_elements * sizeof(uint16_t));
                    for (uint32_t local = 0; local < tile.hidden_elements; ++local) {
                        Require(next_k[local] == topk, "final cast precedes completed K reduction");
                        PutWord(converted, local, Encode(sum[local], fixture.dtype));
                    }
                    slots[slot].bytes = std::move(converted);
                    slots[slot].version = version;
                });
                machine.Signal(Event::V_MTE3, 0);
                machine.Wait(Event::V_MTE3, 0);
                machine.Work(STORE_PIPE, [&, slot, tile, token, version] {
                    Require(slots[slot].version == version, "output source slot overwritten before MTE3");
                    std::copy(slots[slot].bytes.begin(), slots[slot].bytes.end(),
                        output[token - TOKEN_BEGIN].begin() + tile.hidden_offset * sizeof(uint16_t));
                });
                machine.Signal(Event::MTE3_V, 0);
                if (scenario != "early_output_credit") machine.Wait(Event::MTE3_V, 0);
            }
            pipeline.MarkSourceConsumed(slot);
            live_scale = POISON_WEIGHT;
        }
        if (scenario != "missing_token_retirement") retire_token_descriptor_use(pipeline, backend);
        // Overwrite descriptors only after the production per-token closure.
        // Payload packets in a good implementation retain issue-time scalar values.
        std::fill(routes.addresses.begin(), routes.addresses.end(), SOURCE_WINDOW);
        std::fill(routes.scales.begin(), routes.scales.end(), POISON_WEIGHT);
    }
    Require(routes.occurrences == expected_occurrences, "occurrence advanced once per hidden tile instead of per contribution");
    if (scenario != "missing_tail_drain") pipeline.Drain();
    Require(output == fixture.golden, "tile scratch arithmetic/capture differs from independent FP32/RNE oracle");
    machine.Finish();
    Require(machine.SawOverlap(), "tile loads never overlap vector or output work");
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "one scenario required");
        const std::string scenario = argv[1];
        if (scenario == "planning") planning();
        else {
            const bool negative = scenario == "late_dma" || scenario == "late_scale"
                || scenario == "missing_token_retirement" || scenario == "early_output_credit"
                || scenario == "missing_tail_drain" || scenario == "reverse_k" || scenario == "fused";
            bool detected = false;
            for (uint32_t seed = 0; seed < SCHEDULE_COUNT; ++seed) {
                try { run_tiled(scenario, seed); }
                catch (const std::runtime_error &) { if (!negative) throw; detected = true; break; }
            }
            if (negative) Require(detected, "invalid lifetime/arithmetic mutation escaped the asynchronous oracle");
        }
        std::cout << scenario << " passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
