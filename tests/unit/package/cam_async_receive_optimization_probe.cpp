// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cam_async_combine_prefetch.h"
#include "cam_async_mte_pipeline.h"
#include "cam_async_receive_cache.h"

namespace {
using Event = AscendC::HardEvent;
using Bytes = std::vector<uint8_t>;
constexpr uint32_t LOAD_PIPE = 0;
constexpr uint32_t VECTOR_PIPE = 1;
constexpr uint32_t STORE_PIPE = 2;
constexpr uint32_t PIPE_COUNT = 3;
constexpr uint32_t SCHEDULE_COUNT = 64;
constexpr uint32_t SLOT_COUNT = 2;
constexpr uint32_t FULL_HIDDEN = 1024;
constexpr uint32_t PARTIAL_HIDDEN = 7168;
constexpr uint32_t LARGE_HIDDEN = 12288;
constexpr uint32_t PRECEDING_TOKENS = 3;
constexpr uint32_t CACHE_HIDDEN = 64;
constexpr uint64_t SOURCE_ADDRESS = uint64_t{1} << 33;
constexpr uint64_t SCALE_PADDING = 32;
constexpr uint64_t NO_SCRATCH_VERSION = 0;
constexpr float POISON_SCALAR_SCALE = 19.0F;
constexpr float POISON_SCRATCH_VALUE = 1024.0F;

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

void CheckPlanning()
{
    using namespace Cam::AsyncCombine;
    using Cam::AsyncReceiveCache::CanUseCache;
    Require(!CanUseCache(3, 3, 16, 8), "single expert must retain alternating P0 slots");
    Require(!CanUseCache(3, 5, 7, 8), "slice inside the initial expert enables cache");
    Require(!CanUseCache(3, 5, 8, 8), "slice ending at the initial boundary enables cache");
    Require(CanUseCache(3, 5, 9, 8), "slice crossing the initial expert boundary rejects cache");
    // With the native strict-< initial walk, a slice beginning at row 8 still
    // names the preceding expert. It is conservatively eligible when end > 8.
    Require(CanUseCache(3, 5, 10, 8), "exact-begin-boundary conservative cache eligibility lost");
    Require(!PlanCombinePrefetch(31, PARTIAL_HIDDEN, 2, 2).enabled, "31-byte spare admitted");
    Require(!PlanCombinePrefetch(32, PARTIAL_HIDDEN, 2, 2).enabled, "32-byte microtile admitted");
    Require(!PlanCombinePrefetch(2047, PARTIAL_HIDDEN, 2, 2).enabled, "subminimum main tile admitted");
    const auto boundary = PlanCombinePrefetch(2048, PARTIAL_HIDDEN, 2, 2);
    Require(boundary.enabled && boundary.tileElements == FULL_HIDDEN, "2048-byte threshold rejected");
    Require(!PlanCombinePrefetch(2048, FULL_HIDDEN, 2, 1).enabled, "single full row enables prefetch");
    Require(PlanCombinePrefetch(2048, PARTIAL_HIDDEN, 2, 1).enabled, "single contribution split must prefetch");
    Require(!PlanCombinePrefetch(2048, 0, 2, 2).enabled, "zero hidden accepted");
    Require(!PlanCombinePrefetch(2048, PARTIAL_HIDDEN, 3, 2).enabled, "unaligned element size accepted");
    Require(!PlanCombinePrefetch(2048, PARTIAL_HIDDEN, 2, 0).enabled, "empty owner admitted");
    for (const uint32_t hidden : {FULL_HIDDEN, PARTIAL_HIDDEN, LARGE_HIDDEN}) {
        for (uint64_t spare = 0; spare <= 25000; spare += 31) {
            const auto plan = PlanCombinePrefetch(spare, hidden, 2, 3);
            Require(plan.secondSlotBytes <= spare && plan.secondSlotBytes % 32 == 0,
                "optional slot exceeds or misaligns spare UB");
            if (plan.enabled) {
                Require(plan.tileElements > 0 && plan.tileElements <= hidden
                    && static_cast<uint64_t>(plan.tileElements) * 2 <= plan.secondSlotBytes,
                    "tile extent exceeds optional slot");
            }
        }
    }
    Require(Encode(1.00048828125F, Dtype::Fp16) == Encode(1.F, Dtype::Fp16), "FP16 even tie rounding");
    Require(Encode(1.00146484375F, Dtype::Fp16) == Encode(1.001953125F, Dtype::Fp16), "FP16 odd tie rounding");
    Require(Encode(1.00390625F, Dtype::Bf16) == Encode(1.F, Dtype::Bf16), "BF16 even tie rounding");
    Require(Encode(1.01171875F, Dtype::Bf16) == Encode(1.015625F, Dtype::Bf16), "BF16 odd tie rounding");
}

struct Slot { Bytes bytes{}; uint64_t address{0}; uint32_t index{0}; };

bool RunPrefetch(const std::string &scenario, uint32_t seed)
{
    using namespace Cam::AsyncCombine;
    const Dtype dtype = scenario == "prefetch_bf16" ? Dtype::Bf16 : Dtype::Fp16;
    const bool zero = scenario == "prefetch_zero";
    const uint32_t hidden = scenario == "prefetch_partial_hidden" || scenario == "prefetch_topk_one_split"
        ? PARTIAL_HIDDEN : scenario == "prefetch_large_hidden" ? LARGE_HIDDEN : FULL_HIDDEN;
    const uint32_t topk = scenario == "prefetch_topk_one" || scenario == "prefetch_topk_one_split"
        ? 1 : scenario == "prefetch_topk_eight" ? 8 : 2;
    const uint32_t tokenCount = zero ? 0 : scenario == "prefetch_topk_one_split" ? 1 : 3;
    const uint32_t contributions = tokenCount * topk;
    const uint32_t contributionBegin = PRECEDING_TOKENS * topk;
    const auto plan = PlanCombinePrefetch(hidden == PARTIAL_HIDDEN ? 6144
        : hidden == LARGE_HIDDEN ? 8192 : 2048, hidden, 2, contributions);
    Require(zero || plan.enabled, "positive prefetch case did not select a real main tile");
    const uint32_t tileElements = zero ? FULL_HIDDEN : plan.tileElements;
    std::vector<Bytes> source(contributions, Bytes(hidden * sizeof(uint16_t)));
    std::vector<float> scales(contributions);
    for (uint32_t contribution = 0; contribution < contributions; ++contribution) {
        scales[contribution] = contribution % 2 ? -0.375F : 0.75F;
        for (uint32_t element = 0; element < hidden; ++element) {
            const float value = (static_cast<int>(element % 13) - 6) * 0.25F + (contribution % 5) * 0.5F;
            PutWord(source[contribution], element, Encode(value, dtype));
        }
    }
    if (topk >= 2 && contributions != 0) {
        const float input = dtype == Dtype::Fp16 ? 1025.F : 1032.F;
        const float scale = std::nextafter(1.F, 2.F);
        const float product = static_cast<float>(static_cast<double>(input) * scale);
        scales[0] = -product;
        scales[1] = scale;
        for (uint32_t element = 0; element < hidden; ++element) {
            PutWord(source[0], element, Encode(1.F, dtype));
            PutWord(source[1], element, Encode(input, dtype));
        }
        Require(GoldenAdd(-product, input, scale) == 0.F && std::fma(input, scale, -product) != 0.F,
            "numeric witness does not distinguish multiply/add from FMA");
    }
    std::vector<Bytes> golden(tokenCount, Bytes(hidden * sizeof(uint16_t)));
    for (uint32_t token = 0; token < tokenCount; ++token) {
        for (uint32_t element = 0; element < hidden; ++element) {
            float sum = 0;
            for (uint32_t logical = 0; logical < topk; ++logical) {
                const uint32_t contribution = token * topk + logical;
                sum = GoldenAdd(sum, Decode(GetWord(source[contribution], element), dtype), scales[contribution]);
            }
            PutWord(golden[token], element, Encode(sum, dtype));
        }
    }
    std::vector<CombineTile> tiles;
    CombineTileCursor cursor;
    cursor.Init(contributionBegin, contributionBegin + contributions, hidden, tileElements);
    std::vector<uint32_t> coverage(static_cast<size_t>(contributions) * hidden);
    while (cursor.HasNext()) {
        const auto tile = cursor.Current();
        Require(tile.contributionIndex >= contributionBegin && tile.contributionIndex < contributionBegin + contributions
            && tile.hiddenElements != 0 && tile.hiddenOffset + tile.hiddenElements <= hidden, "cursor interval invalid");
        for (uint32_t element = tile.hiddenOffset; element < tile.hiddenOffset + tile.hiddenElements; ++element)
            ++coverage[(tile.contributionIndex - contributionBegin) * hidden + element];
        tiles.push_back(tile);
        cursor.Advance();
    }
    for (const auto count : coverage) Require(count == 1, "cursor skipped or repeated a hidden element");
    Machine machine(seed);
    PipelineEvents<Backend> pipeline{Backend{&machine}};
    pipeline.Init(nullptr);
    std::array<Slot, SLOT_COUNT> slots{};
    std::vector<std::vector<float>> sums(tokenCount, std::vector<float>(hidden));
    std::vector<std::vector<uint32_t>> nextLogical(tokenCount, std::vector<uint32_t>(hidden));
    std::vector<Bytes> output(tokenCount, Bytes(hidden * sizeof(uint16_t), 0xA5));
    auto load = [&](uint32_t slot, CombineTile tile) {
        pipeline.WaitForReuse(slot);
        machine.Work(LOAD_PIPE, [&, slot, tile] {
            machine.ObserveLoad();
            const uint32_t contribution = tile.contributionIndex - contributionBegin;
            const uint64_t address = SOURCE_ADDRESS + static_cast<uint64_t>(contribution) * hidden * sizeof(uint16_t)
                + static_cast<uint64_t>(tile.hiddenOffset) * sizeof(uint16_t);
            slots[slot].bytes.assign(source[contribution].begin() + tile.hiddenOffset * sizeof(uint16_t),
                source[contribution].begin() + (tile.hiddenOffset + tile.hiddenElements) * sizeof(uint16_t));
            slots[slot].address = address;
            slots[slot].index = tile.contributionIndex;
        });
        pipeline.MarkLoaded(slot);
    };
    if (!tiles.empty()) load(0, tiles[0]);
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
        const uint32_t slot = static_cast<uint32_t>(tileIndex % SLOT_COUNT);
        const CombineTile tile = tiles[tileIndex];
        if (scenario != "prefetch_missing_ready") pipeline.WaitLoaded(slot);
        if (tileIndex + 1 < tiles.size()) load((slot + 1) % SLOT_COUNT, tiles[tileIndex + 1]);
        const uint32_t contribution = tile.contributionIndex - contributionBegin;
        const uint32_t token = contribution / topk;
        const uint32_t logical = contribution % topk;
        const bool final = logical + 1 == topk;
        const uint32_t outputSlot = scenario == "prefetch_wrong_output_slot" ? 0 : slot;
        machine.Work(VECTOR_PIPE, [&, slot, outputSlot, tile, contribution, token, logical, final] {
            const uint64_t expectedAddress = SOURCE_ADDRESS + static_cast<uint64_t>(contribution) * hidden * sizeof(uint16_t)
                + static_cast<uint64_t>(tile.hiddenOffset) * sizeof(uint16_t);
            Require(slots[slot].address == expectedAddress && slots[slot].index == tile.contributionIndex
                && slots[slot].bytes.size() == tile.hiddenElements * sizeof(uint16_t), "vector read unready or overwritten source bytes");
            Require(std::equal(slots[slot].bytes.begin(), slots[slot].bytes.end(),
                source[contribution].begin() + tile.hiddenOffset * sizeof(uint16_t)),
                "vector read the wrong source byte stream");
            Bytes converted(tile.hiddenElements * sizeof(uint16_t));
            for (uint32_t local = 0; local < tile.hiddenElements; ++local) {
                const uint32_t element = tile.hiddenOffset + local;
                Require(nextLogical[token][element] == logical, "top-k arithmetic reordered");
                const float input = Decode(GetWord(slots[slot].bytes, local), dtype);
                if (scenario == "prefetch_fused_arithmetic") sums[token][element] = std::fma(input, scales[contribution], sums[token][element]);
                else {
                    volatile float product = input * scales[contribution];
                    volatile float sum = sums[token][element] + product;
                    sums[token][element] = sum;
                }
                ++nextLogical[token][element];
                if (final) PutWord(converted, local, Encode(sums[token][element], dtype));
            }
            if (final) slots[outputSlot].bytes = std::move(converted);
        });
        if (final) {
            if (scenario == "prefetch_unsafe_credit") pipeline.MarkSourceConsumed(slot);
            machine.Signal(Event::V_MTE3, 0);
            machine.Wait(Event::V_MTE3, 0);
            machine.Work(STORE_PIPE, [&, outputSlot, tile, token] {
                Require(slots[outputSlot].bytes.size() == tile.hiddenElements * sizeof(uint16_t), "output slot overwritten before MTE3 read");
                std::copy(slots[outputSlot].bytes.begin(), slots[outputSlot].bytes.end(),
                    output[token].begin() + tile.hiddenOffset * sizeof(uint16_t));
            });
            machine.Signal(Event::MTE3_V, 0);
            machine.Wait(Event::MTE3_V, 0);
        }
        if (!final || scenario != "prefetch_unsafe_credit") pipeline.MarkSourceConsumed(slot);
    }
    if (scenario != "prefetch_missing_tail_drain") { pipeline.Drain(); pipeline.Drain(); }
    Require(output == golden, "output publication is early or differs from separate FP32 arithmetic/RNE golden");
    machine.Finish();
    return machine.SawOverlap();
}


struct LoadArguments {
    uint32_t contribution{0};
    uint32_t hiddenOffset{0};
    uint32_t hiddenElements{0};
    uint64_t sourceAddress{0};
};
struct ArithmeticArguments {
    Cam::AsyncCombine::CombineTile tile{};
    uint32_t token{0};
    uint32_t logical{0};
    float scale{0.0F};
    uint64_t scratchVersion{NO_SCRATCH_VERSION};
};

// The former model performed Cast/Mul/Add in one indivisible vector job. This
// version keeps the native full-H row/product/sum scratch and schedules each
// operator separately. PIPE_V orders these vector jobs; neither input readiness
// nor a vector scratch result requires Scalar to finish that job before issuing
// another command. Issue-time values are copied into the asynchronous packets.
// Real CANN IR/ISA must independently establish the native operand capture.
void CheckAsynchronousArithmetic(const std::string &scenario, uint32_t seed, bool waitPerTile)
{
    using namespace Cam::AsyncCombine;
    const Dtype dtype = scenario == "prefetch_async_bf16" ? Dtype::Bf16 : Dtype::Fp16;
    const uint32_t hidden = scenario == "prefetch_async_partial_hidden" ? PARTIAL_HIDDEN : FULL_HIDDEN;
    const uint32_t topk = scenario == "prefetch_async_topk_eight" ? 8 : 2;
    constexpr uint32_t ASYNC_TOKEN_COUNT = 3;
    const uint32_t contributions = ASYNC_TOKEN_COUNT * topk;
    const uint32_t contributionBegin = PRECEDING_TOKENS * topk;
    const auto plan = PlanCombinePrefetch(hidden == PARTIAL_HIDDEN ? 6144 : 2048, hidden, 2, contributions);
    Require(plan.enabled, "asynchronous case requires a production prefetch plan");
    std::vector<Bytes> source(contributions, Bytes(hidden * sizeof(uint16_t)));
    std::vector<float> scales(contributions);
    for (uint32_t contribution = 0; contribution < contributions; ++contribution) {
        scales[contribution] = contribution % 2 ? -0.375F : 0.75F;
        for (uint32_t element = 0; element < hidden; ++element) {
            const float value = (static_cast<int>(element % 13) - 6) * 0.25F + (contribution % 5) * 0.5F;
            PutWord(source[contribution], element, Encode(value, dtype));
        }
    }
    const float input = dtype == Dtype::Fp16 ? 1025.0F : 1032.0F;
    const float weight = std::nextafter(1.0F, 2.0F);
    scales[0] = -static_cast<float>(static_cast<double>(input) * weight);
    scales[1] = weight;
    for (uint32_t element = 0; element < hidden; ++element) {
        PutWord(source[0], element, Encode(1.0F, dtype));
        PutWord(source[1], element, Encode(input, dtype));
    }
    std::vector<Bytes> golden(ASYNC_TOKEN_COUNT, Bytes(hidden * sizeof(uint16_t)));
    for (uint32_t token = 0; token < ASYNC_TOKEN_COUNT; ++token) {
        for (uint32_t element = 0; element < hidden; ++element) {
            float sum = 0.0F;
            for (uint32_t logical = 0; logical < topk; ++logical) {
                const uint32_t contribution = token * topk + logical;
                sum = GoldenAdd(sum, Decode(GetWord(source[contribution], element), dtype), scales[contribution]);
            }
            PutWord(golden[token], element, Encode(sum, dtype));
        }
    }
    CombineTileCursor cursor;
    cursor.Init(contributionBegin, contributionBegin + contributions, hidden, plan.tileElements);
    std::vector<CombineTile> tiles;
    while (cursor.HasNext()) { tiles.push_back(cursor.Current()); cursor.Advance(); }
    Machine machine(seed);
    PipelineEvents<Backend> pipeline{Backend{&machine}};
    pipeline.Init(nullptr);
    std::array<Slot, SLOT_COUNT> slots{};
    // Unlike the former per-token sum matrix, these are the three scratch rows
    // actually reused by the native function, including across token boundaries.
    std::vector<float> row(hidden), product(hidden), sum(hidden);
    std::vector<uint64_t> rowVersion(hidden), productVersion(hidden);
    std::vector<uint32_t> sumToken(hidden), nextLogical(hidden);
    std::array<uint64_t, SLOT_COUNT> outputVersion{};
    std::vector<Bytes> output(ASYNC_TOKEN_COUNT, Bytes(hidden * sizeof(uint16_t), 0xA5));
    LoadArguments liveLoad{};
    float liveScale = 0.0F;
    auto load = [&](uint32_t slot, const CombineTile &tile) {
        pipeline.WaitForReuse(slot);
        const uint32_t contribution = tile.contributionIndex - contributionBegin;
        liveLoad = {contribution, tile.hiddenOffset, tile.hiddenElements,
            SOURCE_ADDRESS + static_cast<uint64_t>(contribution) * hidden * sizeof(uint16_t)
                + static_cast<uint64_t>(tile.hiddenOffset) * sizeof(uint16_t)};
        const LoadArguments captured = liveLoad;
        machine.Work(LOAD_PIPE, [&, slot, captured] {
            const LoadArguments args = scenario == "prefetch_async_live_address" ? liveLoad : captured;
            Require(args.contribution < contributions && args.hiddenElements != 0
                && args.hiddenOffset + args.hiddenElements <= hidden, "DMA used overwritten scalar address arguments");
            machine.ObserveLoad();
            slots[slot].bytes.assign(source[args.contribution].begin() + args.hiddenOffset * sizeof(uint16_t),
                source[args.contribution].begin() + (args.hiddenOffset + args.hiddenElements) * sizeof(uint16_t));
            slots[slot].address = args.sourceAddress;
            slots[slot].index = contributionBegin + args.contribution;
            outputVersion[slot] = NO_SCRATCH_VERSION;
        });
        pipeline.MarkLoaded(slot);
    };
    load(0, tiles[0]);
    bool scalarGotAhead = false;
    bool injectedScratchFault = false;
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
        const uint32_t slot = static_cast<uint32_t>(tileIndex % SLOT_COUNT);
        const auto tile = tiles[tileIndex];
        const uint32_t contribution = tile.contributionIndex - contributionBegin;
        liveScale = scales[contribution];
        const ArithmeticArguments args{tile, contribution / topk, contribution % topk,
            liveScale, static_cast<uint64_t>(tileIndex) + 1};
        scalarGotAhead = scalarGotAhead || machine.HasQueuedVectorWork();
        pipeline.WaitLoaded(slot);
        if (tileIndex + 1 < tiles.size()) load((slot + 1) % SLOT_COUNT, tiles[tileIndex + 1]);
        if (args.logical == 0) {
            machine.Work(VECTOR_PIPE, [&, args] {
                for (uint32_t local = 0; local < args.tile.hiddenElements; ++local) {
                    const uint32_t element = args.tile.hiddenOffset + local;
                    sum[element] = 0.0F;
                    sumToken[element] = args.token;
                    nextLogical[element] = 0;
                }
            });
            if (waitPerTile) machine.FenceVectorToScalar();
        }
        machine.Work(VECTOR_PIPE, [&, slot, args, contribution] {
            const uint64_t expectedAddress = SOURCE_ADDRESS
                + static_cast<uint64_t>(contribution) * hidden * sizeof(uint16_t)
                + static_cast<uint64_t>(args.tile.hiddenOffset) * sizeof(uint16_t);
            Require(slots[slot].address == expectedAddress && slots[slot].index == args.tile.contributionIndex
                && slots[slot].bytes.size() == args.tile.hiddenElements * sizeof(uint16_t),
                "Cast read data loaded with stale address/width operands");
            for (uint32_t local = 0; local < args.tile.hiddenElements; ++local) {
                const uint32_t element = args.tile.hiddenOffset + local;
                row[element] = Decode(GetWord(slots[slot].bytes, local), dtype);
                rowVersion[element] = args.scratchVersion;
            }
        });
        // The two FIFO boundaries here represent the retained PIPE_V barriers
        // between Cast/Muls and Muls/Add, not a Scalar observation of completion.
        machine.Work(VECTOR_PIPE, [&, args] {
            const float scale = scenario == "prefetch_async_live_scale" ? liveScale : args.scale;
            for (uint32_t local = 0; local < args.tile.hiddenElements; ++local) {
                const uint32_t element = args.tile.hiddenOffset + local;
                Require(rowVersion[element] == args.scratchVersion, "row scratch overwritten before Muls");
                volatile float rounded = row[element] * scale;
                product[element] = rounded;
                productVersion[element] = args.scratchVersion;
            }
        });
        machine.Work(VECTOR_PIPE, [&, args] {
            for (uint32_t local = 0; local < args.tile.hiddenElements; ++local) {
                const uint32_t element = args.tile.hiddenOffset + local;
                Require(productVersion[element] == args.scratchVersion, "product scratch overwritten before Add");
                Require(sumToken[element] == args.token && nextLogical[element] == args.logical,
                    "shared sum scratch reused or top-k arithmetic reordered");
                volatile float rounded = sum[element] + product[element];
                sum[element] = rounded;
                ++nextLogical[element];
            }
        });
        if (scenario == "prefetch_async_scalar_scratch" && !injectedScratchFault && !waitPerTile) {
            // Deterministically stop at a real pending consumer interval, then
            // inject an erroneous Scalar write into live FP32 product storage.
            const uint32_t element = args.tile.hiddenOffset;
            while (productVersion[element] != args.scratchVersion) machine.ProgressOneCommand();
            product[element] = POISON_SCRATCH_VALUE;
            injectedScratchFault = true;
        }
        if (waitPerTile) machine.FenceVectorToScalar();
        if (args.logical + 1 == topk) {
            machine.Work(VECTOR_PIPE, [&, slot, args] {
                Bytes converted(args.tile.hiddenElements * sizeof(uint16_t));
                for (uint32_t local = 0; local < args.tile.hiddenElements; ++local) {
                    const uint32_t element = args.tile.hiddenOffset + local;
                    Require(sumToken[element] == args.token && nextLogical[element] == topk,
                        "final Cast read incomplete or overwritten sum scratch");
                    PutWord(converted, local, Encode(sum[element], dtype));
                }
                slots[slot].bytes = std::move(converted);
                outputVersion[slot] = args.scratchVersion;
            });
            machine.Signal(Event::V_MTE3, 0);
            machine.Wait(Event::V_MTE3, 0);
            machine.Work(STORE_PIPE, [&, slot, args] {
                Require(outputVersion[slot] == args.scratchVersion,
                    "MTE3 output source overwritten before its read");
                std::copy(slots[slot].bytes.begin(), slots[slot].bytes.end(),
                    output[args.token].begin() + args.tile.hiddenOffset * sizeof(uint16_t));
            });
            machine.Signal(Event::MTE3_V, 0);
            machine.Wait(Event::MTE3_V, 0);
        }
        pipeline.MarkSourceConsumed(slot);
        // Scalar moves ahead immediately. Positive packets retain the old scale;
        // the negative reference-capture mutation observes this replacement.
        liveScale = POISON_SCALAR_SCALE;
    }
    liveLoad = {contributions, hidden, 0, SOURCE_ADDRESS};
    pipeline.Drain();
    Require(output == golden, "asynchronous Cast/Muls/Add changed arithmetic or operand capture");
    machine.Finish();
    const uint32_t firstKTiles = ASYNC_TOKEN_COUNT * ((hidden + plan.tileElements - 1) / plan.tileElements);
    Require(machine.VectorScalarWaits() == (waitPerTile ? tiles.size() + firstKTiles : 0),
        "unexpected per-tile Scalar wait count");
    if (!waitPerTile) Require(scalarGotAhead, "Scalar never issued ahead of vector completion");
}
void RunAsynchronousPrefetch(const std::string &scenario, uint32_t seed)
{
    // Each valid schedule is checked both with the former two V_S sites and
    // without them. Both use the same production cursor/event leases and golden.
    const bool negative = scenario == "prefetch_async_live_scale"
        || scenario == "prefetch_async_live_address" || scenario == "prefetch_async_scalar_scratch";
    if (!negative) CheckAsynchronousArithmetic(scenario, seed, true);
    CheckAsynchronousArithmetic(scenario, seed, false);
}

struct WireRow { Cam::AsyncReceiveCache::RowKey key; Bytes bytes; };
WireRow MakeWireRow(uint64_t source, uint32_t index, bool quantized, uint32_t epoch)
{
    const uint32_t payloadBytes = quantized ? CACHE_HIDDEN : CACHE_HIDDEN * sizeof(uint16_t);
    const uint32_t rowBytes = payloadBytes + (quantized ? SCALE_PADDING : 0);
    WireRow row{{source, index}, Bytes(rowBytes)};
    for (uint32_t byte = 0; byte < rowBytes; ++byte)
        row.bytes[byte] = static_cast<uint8_t>(epoch * 31 + index * 47 + byte * 7 + (source >> 5));
    if (quantized) {
        const float scale = static_cast<float>(epoch + index + 1) * 0.125F;
        std::memcpy(row.bytes.data() + payloadBytes, &scale, sizeof(scale));
    }
    return row;
}

void CacheCall(const std::string &scenario, Machine &machine,
    Cam::AsyncReceiveCache::TwoRowCache &cache, const std::vector<WireRow> &rows,
    std::array<Slot, SLOT_COUNT> &slots, bool quantized, uint32_t expectedLoads)
{
    using namespace Cam::AsyncMte;
    PipelineEvents<Backend> pipeline{Backend{&machine}};
    pipeline.Init(nullptr);
    const uint32_t payloadBytes = quantized ? CACHE_HIDDEN : CACHE_HIDDEN * sizeof(uint16_t);
    std::vector<Bytes> output(rows.size(), Bytes(payloadBytes + (quantized ? sizeof(float) : 0), 0xA5));
    uint32_t loads = 0;
    for (uint32_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        const WireRow row = rows[rowIndex];
        const auto selection = cache.Select(row.key);
        const uint32_t slot = selection.slot;
        Require(slot < SLOT_COUNT, "cache returned an invalid slot");
        if (!(selection.hit && scenario == "cache_pending_hit_without_credit")) pipeline.WaitForReuse(slot);
        if (!selection.hit) {
            ++loads;
            machine.Work(LOAD_PIPE, [&, slot, row] {
                machine.ObserveLoad();
                slots[slot] = {row.bytes, row.key.sourceAddress, row.key.wireIndex};
            });
        }
        pipeline.MarkLoaded(slot);
        pipeline.WaitLoaded(slot);
        machine.Work(STORE_PIPE, [&, slot, row, rowIndex] {
            Require(slots[slot].address == row.key.sourceAddress && slots[slot].index == row.key.wireIndex
                && slots[slot].bytes == row.bytes, "payload read unready, stale or overwritten complete wire row");
            std::copy_n(slots[slot].bytes.begin(), payloadBytes, output[rowIndex].begin());
        });
        if (scenario == "cache_early_scale_credit") pipeline.MarkWritten(slot);
        if (quantized) {
            machine.Work(STORE_PIPE, [&, slot, row, rowIndex] {
                Require(slots[slot].bytes == row.bytes, "scale and payload did not retain one wire lease");
                std::copy_n(slots[slot].bytes.begin() + payloadBytes, sizeof(float), output[rowIndex].begin() + payloadBytes);
            });
        }
        if (scenario != "cache_early_scale_credit") pipeline.MarkWritten(slot);
        cache.Remember(slot, row.key);
    }
    Require(loads == expectedLoads, "cache hit/miss/LRU policy changed real DMA count");
    if (scenario != "cache_missing_tail_drain") { pipeline.Drain(); pipeline.Drain(); }
    for (uint32_t row = 0; row < rows.size(); ++row) {
        Require(std::equal(output[row].begin(), output[row].end(), rows[row].bytes.begin()),
            "publish before all payload/scale output or stale cache hit");
    }
    machine.Finish();
}

void RunCache(const std::string &scenario, uint32_t seed)
{
    const bool quantized = scenario == "cache_quantized" || scenario == "cache_early_scale_credit";
    const auto a = MakeWireRow(SOURCE_ADDRESS, 0, quantized, 1);
    const auto b = MakeWireRow(SOURCE_ADDRESS + 4096, 1, quantized, 1);
    const auto c = MakeWireRow(SOURCE_ADDRESS + 8192, 2, quantized, 1);
    const auto otherSource = MakeWireRow(SOURCE_ADDRESS + 16384, 0, quantized, 1);
    const auto otherIndex = MakeWireRow(SOURCE_ADDRESS, 7, quantized, 1);
    std::vector<WireRow> rows{a, b, a};
    uint32_t expectedLoads = 2;
    if (scenario == "cache_aa" || scenario == "cache_pending_hit_without_credit") { rows = {a, a}; expectedLoads = 1; }
    if (scenario == "cache_lru") { rows = {a, b, c, b, a}; expectedLoads = 4; }
    if (scenario == "cache_source_keys") { rows = {a, otherSource, a, otherIndex}; expectedLoads = 3; }
    if (scenario == "cache_early_scale_credit") { rows = {a, b, c, a}; expectedLoads = 4; }
    Machine machine(seed);
    std::array<Slot, SLOT_COUNT> slots{};
    Cam::AsyncReceiveCache::TwoRowCache cache;
    CacheCall(scenario, machine, cache, rows, slots, quantized, expectedLoads);
    if (scenario == "cache_reset" || scenario == "cache_missing_reset") {
        if (scenario == "cache_reset") cache = Cam::AsyncReceiveCache::TwoRowCache{};
        const auto newer = MakeWireRow(SOURCE_ADDRESS, 0, quantized, 2);
        CacheCall(scenario, machine, cache, {newer}, slots, quantized,
            scenario == "cache_reset" ? 1 : 0);
    }
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "one scenario is required");
        const std::string scenario = argv[1];
        if (scenario == "planning") CheckPlanning();
        else {
            const bool negative = scenario == "prefetch_missing_ready" || scenario == "prefetch_unsafe_credit"
                || scenario == "prefetch_wrong_output_slot" || scenario == "prefetch_missing_tail_drain"
                || scenario == "prefetch_fused_arithmetic" || scenario == "cache_pending_hit_without_credit"
                || scenario == "cache_early_scale_credit" || scenario == "cache_missing_reset"
                || scenario == "cache_missing_tail_drain"
                || scenario == "prefetch_async_live_scale" || scenario == "prefetch_async_live_address"
                || scenario == "prefetch_async_scalar_scratch";
            bool detected = false;
            bool overlap = false;
            for (uint32_t seed = 0; seed < SCHEDULE_COUNT; ++seed) {
                try {
                    if (scenario.find("prefetch_async_") == 0) RunAsynchronousPrefetch(scenario, seed);
                    else if (scenario.find("prefetch_") == 0) overlap = RunPrefetch(scenario, seed) || overlap;
                    else if (scenario.find("cache_") == 0) RunCache(scenario, seed);
                    else throw std::runtime_error("unknown scenario");
                } catch (const std::runtime_error &error) {
                    if (!negative) throw;
                    const std::string reason = error.what();
                    if (scenario == "prefetch_async_live_address")
                        Require(reason == "DMA used overwritten scalar address arguments"
                            || reason == "Cast read data loaded with stale address/width operands",
                            "address mutation failed for an unrelated reason");
                    if (scenario == "prefetch_async_live_scale" || scenario == "prefetch_async_scalar_scratch")
                        Require(reason == "asynchronous Cast/Muls/Add changed arithmetic or operand capture",
                            "arithmetic mutation failed for an unrelated reason");
                    detected = true;
                    break;
                }
            }
            if (negative) Require(detected, "invalid protocol/arithmetic mutation was not detected");
            else if (scenario.find("prefetch_") == 0 && scenario.find("prefetch_async_") != 0
                && scenario != "prefetch_zero")
                Require(overlap, "all schedules serialized load behind compute/store");
        }
        std::cout << scenario << " passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
