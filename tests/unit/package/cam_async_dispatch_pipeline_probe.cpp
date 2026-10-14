// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// FIFO event model for the shared production plan/leases. Input and wire bytes
// are independent opaque fixtures; original quantization math is not simulated.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cam_async_dispatch_pipeline.h"
#include "cam_async_sparse_ranks.h"

namespace {
using Event = AscendC::HardEvent;
using Bytes = std::vector<uint8_t>;
constexpr uint32_t LOAD = 0;
constexpr uint32_t VECTOR = 1;
constexpr uint32_t STORE = 2;
constexpr uint32_t PIPE_COUNT = 3;
constexpr uint32_t HIDDEN = 64;
constexpr uint32_t TOKEN_COUNT = 12;
constexpr uint32_t RANK_COUNT = 3;
constexpr uint32_t EXPERTS_PER_RANK = 2;
constexpr uint32_t EXPERT_COUNT = RANK_COUNT * EXPERTS_PER_RANK;
constexpr uint32_t BATCH_ENTRIES = 16;
constexpr uint32_t WIRE_PADDING = 32;
constexpr uint32_t SCALE_BITS = 0x7FC12345;
constexpr uint16_t ADDRESS_GUARD = 0xA5A5;
constexpr uint32_t STALE_OWNER = 0xFFFFFFFF;
constexpr uint32_t MAX_STEPS = 1000000;
using Key = std::pair<Event, int>;

void require(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }
uint32_t source_pipe(Event event)
{
    if (event == Event::MTE3_MTE2) return STORE;
    if (event == Event::V_MTE3) return VECTOR;
    return LOAD;
}
uint32_t target_pipe(Event event)
{
    if (event == Event::MTE2_V) return VECTOR;
    if (event == Event::MTE3_MTE2) return LOAD;
    return STORE;
}

struct Job { uint32_t kind; Key key; std::function<void()> action; uint32_t delay; };
class Machine {
public:
    explicit Machine(uint32_t seed) : random_(seed) {}
    int allocate(Event event)
    {
        const int id = next_[event]++;
        require(id < 8, "event pool exhausted");
        owned_[{event, id}] = true;
        return id;
    }
    void release(Event event, int id)
    {
        require(owned_[{event, id}] && credits_[{event, id}] == 0, "release of pending/unowned event");
        owned_[{event, id}] = false;
    }
    void work(uint32_t pipe, std::function<void()> action, uint32_t delay = 1)
    { queues_[pipe].push_back({0, {Event::MTE2_V, 0}, std::move(action), delay}); }
    void signal(Event event, int id)
    {
        const Key key{event, id};
        if (event == Event::S_MTE3) {
            require(credits_[key] == 0, "borrowed scale-ready flag reused before consumer wait");
            ++credits_[key];
        } else queues_[source_pipe(event)].push_back({1, key, {}, 0});
    }
    void wait(Event event, int id) { queues_[target_pipe(event)].push_back({2, {event, id}, {}, 0}); }
    void step()
    {
        random_ = random_ * 1664525U + 1013904223U;
        for (uint32_t turn = 0; turn < PIPE_COUNT; ++turn) {
            const uint32_t pipe = (random_ + turn) % PIPE_COUNT;
            if (queues_[pipe].empty()) continue;
            auto &job = queues_[pipe].front();
            if (job.delay != 0) { --job.delay; return; }
            if (job.kind == 2 && credits_[job.key] == 0) continue;
            if (job.kind == 1) {
                require(credits_[job.key] == 0, "ready/free flag overwritten before wait");
                ++credits_[job.key];
            } else if (job.kind == 2) --credits_[job.key];
            else job.action();
            queues_[pipe].pop_front();
            return;
        }
    }
    void fence(uint32_t pipe)
    {
        bool reached = false;
        work(pipe, [&] { reached = true; }, 0);
        for (uint32_t stepCount = 0; !reached && stepCount < MAX_STEPS; ++stepCount) step();
        require(reached, "pipeline fence deadlocked");
    }
    void finish()
    {
        for (const auto &queue : queues_) require(queue.empty(), "publication before pending work drained");
        for (const auto &entry : owned_) require(!entry.second, "event lease leaked");
        for (const auto &entry : credits_) require(entry.second == 0, "unconsumed event credit");
    }
private:
    std::array<std::deque<Job>, PIPE_COUNT> queues_;
    std::map<Event, int> next_;
    std::map<Key, bool> owned_;
    std::map<Key, uint32_t> credits_;
    uint32_t random_;
};

struct Backend {
    using EventId = int;
    Machine *machine;
    bool borrowed{false};
    void init(AscendC::TPipe *) {}
    template <Event E> int allocate() { return machine->allocate(E); }
    template <Event E> void release(int id) { machine->release(E, id); }
    template <Event E> int key(int id) const
    { return borrowed && (E == Event::V_MTE3 || E == Event::S_MTE3) ? 900 : id; }
    template <Event E> void signal(int id) { machine->signal(E, key<E>(id)); }
    template <Event E> void wait(int id) { machine->wait(E, key<E>(id)); }
    void fence_to_scalar() { machine->fence(LOAD); }
};
struct RankBackend {
    static void WriteRank(std::vector<uint8_t> &list, uint32_t index, uint32_t rank) { list.at(index) = static_cast<uint8_t>(rank); }
    static uint32_t ReadRank(std::vector<uint8_t> &list, uint32_t index) { return list.at(index); }
    static void IncrementOffset(std::vector<uint16_t> &offsets, uint32_t rank) { ++offsets.at(rank); }
};

Bytes input_bytes(uint32_t token)
{
    Bytes bytes(HIDDEN * sizeof(uint16_t));
    for (uint32_t index = 0; index < bytes.size(); ++index) bytes[index] = static_cast<uint8_t>(token * 37 + index * 11);
    return bytes;
}
Bytes wire_bytes(uint32_t token, bool quantized)
{
    Bytes bytes = input_bytes(token);
    if (quantized) {
        bytes.resize(HIDDEN + WIRE_PADDING);
        for (uint32_t index = 0; index < HIDDEN; ++index) bytes[index] = static_cast<uint8_t>(token * 71 + index * 13);
        const uint32_t scale = SCALE_BITS + token;
        std::memcpy(bytes.data() + HIDDEN, &scale, sizeof(scale));
    }
    return bytes;
}
using Routes = std::vector<std::vector<uint32_t>>;
Routes raw_routes()
{
    Routes routes(TOKEN_COUNT, {0, 0, 0, 2, 4, 4});
    for (uint32_t token = 1; token < TOKEN_COUNT; token += 3) routes[token] = {5, 2, 1, 2, 0, 5};
    return routes;
}
struct Metadata { uint32_t rank; uint32_t slot; uint16_t wire; };
struct Slot { Bytes bytes; uint32_t owner{STALE_OWNER}; };

template <bool Quantized>
void run_model(const std::string &mutation, uint32_t seed, bool ownedSlice)
{
    using Pipeline = Cam::AsyncDispatch::PipelineEvents<Backend, Quantized>;
    const Routes routes = raw_routes();
    const uint32_t begin = ownedSlice ? 2 : 0;
    const uint32_t end = ownedSlice ? 9 : TOKEN_COUNT;
    std::vector<std::vector<uint32_t>> rankTokens(RANK_COUNT);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> expertRows(EXPERT_COUNT);
    std::vector<uint16_t> offsets(RANK_COUNT, 0), occurrences(EXPERT_COUNT, 0);
    // Independent token-first raw-ID oracle, without the production bitmap.
    for (uint32_t token = 0; token < TOKEN_COUNT; ++token) {
        std::vector<bool> seen(RANK_COUNT, false);
        for (uint32_t assignment = 0; assignment < routes[token].size(); ++assignment) {
            const uint32_t expert = routes[token][assignment], rank = expert / EXPERTS_PER_RANK;
            expertRows[expert].push_back({token, assignment});
            if (token < begin) ++occurrences[expert];
            if (!seen[rank]) { seen[rank] = true; rankTokens[rank].push_back(token); if (token < begin) ++offsets[rank]; }
        }
    }
    const uint32_t wireBytes = static_cast<uint32_t>(wire_bytes(0, Quantized).size());
    std::vector<uint32_t> prefix(EXPERT_COUNT, 0);
    std::vector<Bytes> actual(RANK_COUNT), expected(RANK_COUNT);
    std::vector<std::vector<uint16_t>> addresses(RANK_COUNT), expectedAddresses(RANK_COUNT);
    for (uint32_t rank = 0; rank < RANK_COUNT; ++rank) {
        actual[rank].assign(rankTokens[rank].size() * wireBytes + WIRE_PADDING, 0xA5);
        expected[rank] = actual[rank];
        uint32_t entries = 0;
        for (uint32_t local = 0; local < EXPERTS_PER_RANK; ++local) {
            const uint32_t expert = rank * EXPERTS_PER_RANK + local;
            prefix[expert] = entries; entries += static_cast<uint32_t>(expertRows[expert].size());
        }
        addresses[rank].assign(entries + BATCH_ENTRIES, ADDRESS_GUARD);
        expectedAddresses[rank] = addresses[rank];
        for (uint32_t wire = 0; wire < rankTokens[rank].size(); ++wire) {
            const uint32_t token = rankTokens[rank][wire];
            if (token >= begin && token < end) {
                const auto row = wire_bytes(token, Quantized);
                std::copy(row.begin(), row.end(), expected[rank].begin() + wire * wireBytes);
            }
        }
    }
    for (uint32_t expert = 0; expert < EXPERT_COUNT; ++expert) {
        const uint32_t rank = expert / EXPERTS_PER_RANK;
        for (uint32_t position = 0; position < expertRows[expert].size(); ++position) {
            const uint32_t token = expertRows[expert][position].first;
            if (token >= begin && token < end) expectedAddresses[rank][prefix[expert] + position] =
                static_cast<uint16_t>(std::find(rankTokens[rank].begin(), rankTokens[rank].end(), token) - rankTokens[rank].begin());
        }
    }
    const auto plan = Cam::AsyncDispatch::plan_dispatch_buffers(HIDDEN * sizeof(uint16_t), HIDDEN,
        sizeof(uint16_t), Quantized ? HIDDEN + WIRE_PADDING : HIDDEN, Quantized ? 1 : sizeof(uint16_t), end - begin);
    require(plan.enabled && plan.secondSlotBytes == HIDDEN * sizeof(uint16_t), "opaque case not admitted by actual buffer plan");
    Machine machine(seed);
    Pipeline pipeline{Backend{&machine}};
    pipeline.init(nullptr);
    std::array<Slot, Cam::AsyncDispatch::TOKEN_SLOT_COUNT> slots;
    Slot scratch;
    std::array<std::vector<Metadata>, EXPERT_COUNT> batches;
    std::vector<uint8_t> visited(RANK_COUNT, 0xFF);
    Cam::AsyncSparseRanks::TokenRankTracker tracker(RANK_COUNT);
    uint32_t currentToken = begin;
    uint32_t materializations = 0;
    auto load = [&](uint32_t slot, uint32_t token) {
        pipeline.wait_for_reuse(slot);
        machine.work(LOAD, [&, slot, token] {
            const uint32_t captured = mutation == "capture" ? currentToken : token;
            slots[slot] = {input_bytes(captured), captured};
        });
        pipeline.mark_loaded(slot);
    };
    auto flush = [&](uint32_t expert) {
        if (batches[expert].empty()) return;
        const auto expectedCount = batches[expert].size();
        machine.work(STORE, [&, expert, expectedCount] {
            require(batches[expert].size() == expectedCount, "metadata scratch reused without its scalar fence");
            for (const auto &entry : batches[expert]) addresses[entry.rank][entry.slot] = entry.wire;
        }, 7);
        if (mutation != "metadata_fence") machine.fence(STORE);
        batches[expert].clear();
    };
    load(0, begin);
    for (uint32_t token = begin; token < end; ++token) {
        currentToken = token;
        const uint32_t slot = (token - begin) % Cam::AsyncDispatch::TOKEN_SLOT_COUNT;
        pipeline.wait_loaded(slot);
        if (token + 1 < end) load((slot + 1) % Cam::AsyncDispatch::TOKEN_SLOT_COUNT, token + 1);
        if constexpr (Quantized) {
            machine.work(VECTOR, [&, slot, token] {
                require(slots[slot].owner == token && slots[slot].bytes == input_bytes(token), "quantization sees wrong/unready input capture");
                scratch = {wire_bytes(token, true), token};
                ++materializations;
            });
            // The original quotient V_S handoff first finishes the input read.
            machine.fence(VECTOR);
            if (mutation == "scratch" && token == begin + 1) machine.work(LOAD, [&] { scratch.owner = STALE_OWNER; }, 0);
            machine.work(VECTOR, [&, slot, token] {
                require(scratch.owner == token, "shared FP32 scratch overwritten outside ordered V lifetime");
                std::copy_n(scratch.bytes.begin(), HIDDEN, slots[slot].bytes.begin());
            }, mutation == "scratch" ? 100 : 1);
            pipeline.mark_quantized(slot);
            const uint32_t scale = SCALE_BITS + token;
            std::memcpy(slots[slot].bytes.data() + HIDDEN, &scale, sizeof(scale));
            pipeline.mark_scale_ready(slot);
        } else ++materializations;
        tracker.BeginToken();
        bool postedEarly = false;
        for (uint32_t assignment = 0; assignment < routes[token].size(); ++assignment) {
            const uint32_t expert = routes[token][assignment], rank = expert / EXPERTS_PER_RANK;
            const uint16_t wire = offsets[rank];
            if (tracker.RecordRank<RankBackend>(visited, rank) == Cam::AsyncSparseRanks::Visit::First) {
                // Split beats model delayed peers and a delayed four-byte scale.
                for (uint32_t part = 0; part < 2; ++part) {
                    const uint32_t first = part == 0 ? 0 : HIDDEN;
                    const uint32_t last = part == 0 ? HIDDEN : wireBytes;
                    machine.work(STORE, [&, slot, token, rank, wire, first, last] {
                        const auto row = wire_bytes(token, Quantized);
                        require(slots[slot].owner == token && std::equal(row.begin() + first, row.begin() + last,
                            slots[slot].bytes.begin() + first), "peer read stale payload/scale or prematurely reused slot");
                        std::copy(slots[slot].bytes.begin() + first, slots[slot].bytes.begin() + last,
                            actual[rank].begin() + wire * wireBytes + first);
                    }, 2 + rank * 3 + part);
                }
                if (mutation == "early_credit" && !postedEarly) { pipeline.mark_token_sent(slot); postedEarly = true; }
            }
            batches[expert].push_back({rank, prefix[expert] + occurrences[expert], wire});
            ++occurrences[expert];
            if (batches[expert].size() == BATCH_ENTRIES) flush(expert);
        }
        tracker.AdvanceOffsets<RankBackend>(visited, offsets);
        if (!postedEarly) pipeline.mark_token_sent(slot);
    }
    if (mutation != "tail_drain") pipeline.drain();
    for (uint32_t expert = 0; expert < EXPERT_COUNT; ++expert) flush(expert);
    require(actual == expected && addresses == expectedAddresses, "publication/opaque wire/raw-routing metadata oracle differs");
    require(materializations == end - begin, "token quantization/materialization repeated or omitted");
    machine.finish();
}

void borrowed_ready()
{
    Machine machine(1);
    Cam::AsyncDispatch::PipelineEvents<Backend, true> pipeline{Backend{&machine, true}};
    pipeline.init(nullptr);
    machine.work(STORE, [] {}, 100);
    pipeline.mark_quantized(0); pipeline.mark_scale_ready(0);
    pipeline.mark_quantized(1); pipeline.mark_scale_ready(1);
}

void planning()
{
    using Cam::AsyncDispatch::plan_dispatch_buffers;
    require(!plan_dispatch_buffers(127, 64, 2, 96, 1, 2).enabled, "insufficient spare accepted");
    require(plan_dispatch_buffers(128, 64, 2, 96, 1, 2).enabled, "exact q1 spare rejected");
    require(plan_dispatch_buffers(64, 32, 2, 64, 1, 2).enabled, "small exact q1 row alias rejected");
    require(!plan_dispatch_buffers(128, 16, 2, 48, 1, 2).enabled, "wire exceeds original mandatory row");
    require(!plan_dispatch_buffers(128, 64, 2, 64, 2, 1).enabled, "one token should keep serial path");
    require(!plan_dispatch_buffers(128, 64, 2, 64, 2, 0).enabled, "empty owner should keep serial path");
    require(!plan_dispatch_buffers(128, 64, 3, 64, 2, 2).enabled, "unsupported element alignment accepted");
    require(!plan_dispatch_buffers(~uint64_t{0}, 64, ~uint64_t{0}, 64, 2, 2).enabled, "overflowing element size accepted");
}
} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    const std::string scenario = argv[1];
    const bool negative = scenario == "early_credit" || scenario == "capture" || scenario == "metadata_fence"
        || scenario == "scratch" || scenario == "tail_drain" || scenario == "borrowed_ready";
    bool detected = false;
    try {
        if (scenario == "planning") planning();
        else if (scenario == "borrowed_ready") borrowed_ready();
        else {
            require(scenario == "raw_multi_peer" || scenario == "quantized_multi_peer" || scenario == "owned_slice"
                || negative, "unknown dispatch pipeline scenario");
            for (uint32_t seed = 1; seed <= 16; ++seed) {
                if (scenario == "raw_multi_peer") run_model<false>(scenario, seed, false);
                else run_model<true>(scenario, seed, scenario == "owned_slice");
            }
        }
    } catch (const std::exception &error) {
        if (!negative) { std::cerr << error.what() << '\n'; return 1; }
        detected = true;
    }
    if (negative && !detected) { std::cerr << "unsafe mutation unexpectedly passed the model\n"; return 1; }
    std::cout << scenario << " passed\n";
    return 0;
}
