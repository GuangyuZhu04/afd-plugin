// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <array>
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

#include "cam_async_receive_runs.h"
#include "cam_async_mte_pipeline.h"

namespace {
using Bytes = std::vector<uint8_t>;
using Event = AscendC::HardEvent;
using Cam::AsyncReceiveRuns::ReceiveRun;
using Cam::AsyncReceiveRuns::ReceiveRunCursor;
constexpr uint32_t LOAD_PIPE = 0;
constexpr uint32_t STORE_PIPE = 1;
constexpr uint32_t SCALAR_PIPE = 2;
constexpr uint32_t PIPE_COUNT = 3;
constexpr uint32_t SCHEDULE_COUNT = 64;
constexpr uint32_t GUARD_BYTES = 32;
constexpr uint32_t PAYLOAD_BYTES = 64;
constexpr uint32_t WIRE_ROWS = 64;
constexpr uint32_t TILE_ROWS = 3;
constexpr uint8_t UNTOUCHED = 0xA5;
constexpr uint64_t SOURCE_BASE = uint64_t{1} << 34;

void require_condition(bool condition, const char *reason)
{
    if (!condition) throw std::runtime_error(reason);
}

struct Fixture {
    std::vector<std::vector<std::vector<uint16_t>>> peerExperts{
        {{}, {4, 5, 6, 7, 8}, {}, {20, 21, 25, 26}, {9, 10, 11}},
        {{0, 1}, {}, {12, 13, 14}, {6, 7}, {2, 3, 4, 5}},
        {{5}, {30, 31, 32}, {}, {}, {16, 18}}};
    uint32_t startExpert{1};
    uint32_t endExpert{4};

    Bytes source_bytes(uint32_t peer, bool quantized) const
    {
        const uint32_t wireBytes = PAYLOAD_BYTES + (quantized ? GUARD_BYTES : 0);
        Bytes source(GUARD_BYTES + WIRE_ROWS * wireBytes + GUARD_BYTES, UNTOUCHED);
        for (uint32_t row = 0; row < WIRE_ROWS; ++row) {
            for (uint32_t byte = 0; byte < wireBytes; ++byte)
                source[GUARD_BYTES + row * wireBytes + byte] = static_cast<uint8_t>(peer * 101 + row * 13 + byte * 7);
            if (quantized) {
                // Opaque exact IEEE bits, including NaN payloads. DMA must not
                // perform floating-point conversion or copy the padding block.
                const uint32_t scaleBits = 0x7FC10000U + peer * WIRE_ROWS + row;
                std::memcpy(source.data() + GUARD_BYTES + row * wireBytes + PAYLOAD_BYTES,
                    &scaleBits, sizeof(scaleBits));
            }
        }
        return source;
    }
};

struct Assignment { uint32_t expert; uint16_t wire; uint32_t output; };
struct View {
    std::vector<uint32_t> counts, totals, priors;
    std::vector<uint16_t> addresses;
    uint32_t read_count(uint32_t expert) const { return counts.at(expert); }
    uint32_t read_total(uint32_t expert) const { return totals.at(expert); }
    uint32_t read_prior(uint32_t expert) const { return priors.at(expert); }
    uint32_t read_wire(uint32_t index) const { return addresses.at(index); }
};

// Independent expert-major, TP-major oracle uses original per-peer row lists.
// It does not reproduce the cursor's prefix walk or contiguous-run algorithm.
std::vector<Assignment> raw_assignments(const Fixture &fixture, uint32_t peer, uint32_t &outputRows)
{
    std::vector<Assignment> assignments;
    outputRows = 0;
    for (uint32_t expert = fixture.startExpert; expert <= fixture.endExpert; ++expert) {
        for (uint32_t source = 0; source < fixture.peerExperts.size(); ++source) {
            for (const uint16_t wire : fixture.peerExperts[source][expert]) {
                if (source == peer) assignments.push_back({expert, wire, outputRows});
                ++outputRows;
            }
        }
    }
    return assignments;
}

View make_view(const Fixture &fixture, uint32_t peer, const std::vector<Assignment> &owned)
{
    const uint32_t experts = static_cast<uint32_t>(fixture.peerExperts[peer].size());
    View view{std::vector<uint32_t>(experts), std::vector<uint32_t>(experts), std::vector<uint32_t>(experts), {}};
    for (uint32_t expert = 0; expert < experts; ++expert) {
        view.counts[expert] = static_cast<uint32_t>(fixture.peerExperts[peer][expert].size());
        for (uint32_t source = 0; source < fixture.peerExperts.size(); ++source) {
            view.totals[expert] += static_cast<uint32_t>(fixture.peerExperts[source][expert].size());
            if (source < peer) view.priors[expert] += static_cast<uint32_t>(fixture.peerExperts[source][expert].size());
        }
    }
    for (const auto &assignment : owned) view.addresses.push_back(assignment.wire);
    return view;
}

enum class Action { Work, Signal, Wait, Release };
struct Command {
    Action action;
    Event event{Event::MTE2_S};
    int id{0};
    std::function<void()> work{};
};
uint32_t source_pipe(Event event) { return event == Event::MTE3_MTE2 ? STORE_PIPE : LOAD_PIPE; }
uint32_t destination_pipe(Event event)
{
    return event == Event::MTE2_MTE3 ? STORE_PIPE : event == Event::MTE3_MTE2 ? LOAD_PIPE : SCALAR_PIPE;
}

// Independent MTE2/MTE3/Scalar FIFO scheduling. Scalar publication is a real
// queue operation, so removing the final MTE2_S handoff cannot silently pass.
class Machine {
public:
    void append(uint32_t pipe, Command command) { queues_[pipe].push_back(std::move(command)); }
    void work(uint32_t pipe, std::function<void()> work)
    {
        append(pipe, {Action::Work, Event::MTE2_S, 0, std::move(work)});
    }
    int allocate(Event event)
    {
        int id = 2;
        while (leases_[{event, id}]) ++id;
        leases_[{event, id}] = true;
        return id;
    }
    void run(uint32_t seed)
    {
        std::mt19937 random(seed);
        while (pending()) {
            std::vector<uint32_t> eligible;
            for (uint32_t pipe = 0; pipe < PIPE_COUNT; ++pipe) {
                if (!queues_[pipe].empty()) {
                    const auto &command = queues_[pipe].front();
                    if (command.action != Action::Wait || tokens_[{command.event, command.id}] != 0)
                        eligible.push_back(pipe);
                }
            }
            require_condition(!eligible.empty(), "unmatched event or deadlock");
            const uint32_t pipe = eligible[random() % eligible.size()];
            Command command = std::move(queues_[pipe].front());
            queues_[pipe].pop_front();
            auto &token = tokens_[{command.event, command.id}];
            if (command.action == Action::Work) command.work();
            else if (command.action == Action::Signal) {
                require_condition(token == 0, "event overwritten before consumption");
                token = 1;
            } else if (command.action == Action::Wait) {
                require_condition(token == 1, "wait consumed missing event");
                token = 0;
            } else {
                require_condition(leases_[{command.event, command.id}] && token == 0,
                    "event released before lease drain");
                leases_[{command.event, command.id}] = false;
            }
        }
        for (const auto &entry : leases_) require_condition(!entry.second, "tile event lease survives drain");
        for (const auto &entry : tokens_) require_condition(entry.second == 0, "tile event token survives drain");
    }
private:
    bool pending() const { for (const auto &queue : queues_) if (!queue.empty()) return true; return false; }
    std::array<std::deque<Command>, PIPE_COUNT> queues_{};
    std::map<std::pair<Event, int>, bool> leases_{};
    std::map<std::pair<Event, int>, uint32_t> tokens_{};
};
struct Backend {
    using EventId = int;
    Machine *machine;
    void Init(AscendC::TPipe *) {}
    template <Event event> int Allocate() { return machine->allocate(event); }
    template <Event event> void Signal(int id) { machine->append(source_pipe(event), {Action::Signal, event, id, {}}); }
    template <Event event> void Wait(int id) { machine->append(destination_pipe(event), {Action::Wait, event, id, {}}); }
    template <Event event> void Release(int id) { machine->append(SCALAR_PIPE, {Action::Release, event, id, {}}); }
    void FenceToScalar() { Signal<Event::MTE2_S>(0); Wait<Event::MTE2_S>(0); }
};

void check_planning()
{
    using namespace Cam::AsyncReceiveRuns;
    require_condition(!plan_receive_runs(1024, 1025, 64, 4).enabled, "address slice loses priority");
    require_condition(!plan_receive_runs(1024, 1, 64, 4).enabled, "unaligned address slice admits misaligned new tiles");
    require_condition(!plan_receive_runs(255, 0, 64, 4).enabled, "two tiles overlap below exact budget");
    const auto exact = plan_receive_runs(256, 0, 64, 4);
    require_condition(exact.enabled && exact.rowsPerTile == 2 && exact.bytesPerTile == 128, "exact dual-tile budget rejected");
    require_condition(!plan_receive_runs(65536, 0, 64, 1).enabled, "singleton requires a run tile");
    require_condition(!plan_receive_runs(65536, 0, 0, 4).enabled, "zero wire size accepted");
    require_condition(!plan_receive_runs(65536, 0, 65, 4).enabled, "misaligned wire row accepted");
    for (uint64_t spare = 0; spare < 8192; ++spare) {
        const auto plan = plan_receive_runs(spare + 32, 32, 96, 17);
        if (plan.enabled) require_condition(plan.bytesPerTile * RUN_TILE_COUNT <= spare
            && plan.rowsPerTile >= MIN_TILE_ROWS && plan.rowsPerTile <= MAX_TILE_ROWS
            && plan.bytesPerTile <= MAX_TILE_BYTES && plan.bytesPerTile % RUN_ALIGNMENT_BYTES == 0,
            "run plan overcommits separate tile arenas");
    }
    const auto huge = plan_receive_runs(std::numeric_limits<uint64_t>::max(), 32, 96, 262144);
    require_condition(huge.enabled && huge.rowsPerTile == MAX_TILE_ROWS, "large-budget overflow or uncapped block count");
    const auto plain = plan_payload_copy(64, 64, 3);
    const auto quant = plan_payload_copy(64, 96, 3);
    require_condition(plain.blockCount == 1 && plain.blockBytes == 192 && plain.sourceStrideBlocks == 0,
        "q0 payload is not one contiguous byte block");
    require_condition(quant.blockCount == 3 && quant.blockBytes == 64 && quant.sourceStrideBlocks == 1,
        "q1 stride is not exactly one 32B source block");
    require_condition(plan_payload_copy(0, 0, 3).blockCount == 0
        && plan_payload_copy(64, 128, 3).blockCount == 0
        && plan_payload_copy(64, 96, 129).blockCount == 0, "invalid payload layout admitted");
}

void run_slice(const Fixture &fixture, uint32_t peer, uint32_t begin, uint32_t end,
    bool quantized, const std::string &scenario, uint32_t seed)
{
    using namespace Cam::AsyncReceiveRuns;
    uint32_t outputRows;
    const auto all = raw_assignments(fixture, peer, outputRows);
    require_condition(begin <= end && end <= all.size(), "invalid fixture source interval");
    const std::vector<Assignment> owned(all.begin() + begin, all.begin() + end);
    const auto metadata = make_view(fixture, peer, owned);
    const uint32_t wireBytes = PAYLOAD_BYTES + (quantized ? GUARD_BYTES : 0);
    const uint64_t addressBytes = (owned.size() * sizeof(uint16_t) + GUARD_BYTES - 1) / GUARD_BYTES * GUARD_BYTES;
    const auto plan = plan_receive_runs(addressBytes + RUN_TILE_COUNT * wireBytes * TILE_ROWS,
        addressBytes, wireBytes, static_cast<uint32_t>(owned.size()));
    if (!plan.enabled) { require_condition(owned.size() < MIN_TILE_ROWS, "unexpected resource fallback"); return; }
    ReceiveRunCursor<View> cursor(metadata);
    cursor.init(begin, end, fixture.startExpert, fixture.endExpert, plan.rowsPerTile);
    std::vector<ReceiveRun> runs;
    ReceiveRun current;
    uint32_t visited = 0;
    while (cursor.next(current)) {
        for (uint32_t row = 0; row < current.rows; ++row) {
            require_condition(visited < owned.size() && current.expert == owned[visited].expert
                && current.wireIndex + row == owned[visited].wire && current.outputRow + row == owned[visited].output,
                "run changes expert/TP/source/output ordering");
            ++visited;
        }
        runs.push_back(current);
    }
    require_condition(visited == owned.size(), "run cursor omitted source rows");
    require_condition(has_contiguous_run(metadata, begin, end, fixture.startExpert, fixture.endExpert, plan.rowsPerTile),
        "positive fixture lost contiguous-run qualification");
    const Bytes source = fixture.source_bytes(peer, quantized);
    Bytes goldenPayload(GUARD_BYTES + outputRows * PAYLOAD_BYTES + GUARD_BYTES, UNTOUCHED);
    Bytes goldenScale(GUARD_BYTES + outputRows * sizeof(uint32_t) + GUARD_BYTES, UNTOUCHED);
    for (const auto &assignment : owned) {
        std::copy_n(source.begin() + GUARD_BYTES + assignment.wire * wireBytes, PAYLOAD_BYTES,
            goldenPayload.begin() + GUARD_BYTES + assignment.output * PAYLOAD_BYTES);
        if (quantized) std::copy_n(source.begin() + GUARD_BYTES + assignment.wire * wireBytes + PAYLOAD_BYTES,
            sizeof(uint32_t), goldenScale.begin() + GUARD_BYTES + assignment.output * sizeof(uint32_t));
    }
    Bytes outputPayload(goldenPayload.size(), UNTOUCHED), outputScale(goldenScale.size(), UNTOUCHED);
    const uint64_t mandatoryRowEnd = GUARD_BYTES + wireBytes;
    const uint64_t firstTile = mandatoryRowEnd + addressBytes;
    Bytes ub(firstTile + RUN_TILE_COUNT * plan.bytesPerTile + GUARD_BYTES, UNTOUCHED);
    std::array<uint64_t, RUN_TILE_COUNT> tags{};
    Machine machine;
    Cam::AsyncMte::PipelineEvents<Backend> pipeline{Backend{&machine}};
    pipeline.Init(nullptr);
    uint32_t slot = 0;
    for (const auto run : runs) {
        const uint64_t tileBase = firstTile + slot * plan.bytesPerTile;
        const uint64_t tag = SOURCE_BASE + peer * 65536 + run.wireIndex * wireBytes;
        if (scenario != "unsafe_reuse") pipeline.WaitForReuse(slot);
        machine.work(LOAD_PIPE, [&, slot, tileBase, run, tag] {
            require_condition(run.rows * wireBytes <= plan.bytesPerTile, "run load exceeds its own tile");
            std::copy_n(source.begin() + GUARD_BYTES + run.wireIndex * wireBytes, run.rows * wireBytes,
                ub.begin() + tileBase);
            tags[slot] = tag;
        });
        pipeline.MarkLoaded(slot);
        pipeline.WaitLoaded(slot);
        auto copy = plan_payload_copy(PAYLOAD_BYTES, wireBytes, run.rows);
        if (scenario == "wrong_padding_stride") copy.sourceStrideBlocks = 0;
        machine.work(STORE_PIPE, [&, slot, tileBase, run, tag, copy] {
            require_condition(tags[slot] == tag, "tile overwritten before payload read");
            const uint64_t sourceStep = copy.blockBytes + copy.sourceStrideBlocks * RUN_ALIGNMENT_BYTES;
            for (uint32_t block = 0; block < copy.blockCount; ++block) {
                require_condition(block * sourceStep + copy.blockBytes <= plan.bytesPerTile, "payload stride leaves its own tile");
                std::copy_n(ub.begin() + tileBase + block * sourceStep, copy.blockBytes,
                    outputPayload.begin() + GUARD_BYTES + run.outputRow * PAYLOAD_BYTES + block * copy.blockBytes);
            }
        });
        if (scenario == "early_scale_credit") pipeline.MarkWritten(slot);
        if (quantized) {
            for (uint32_t row = 0; row < run.rows; ++row) {
                machine.work(STORE_PIPE, [&, slot, tileBase, run, row, tag] {
                    require_condition(tags[slot] == tag, "tile overwritten before scale read");
                    const uint64_t sourceStep = scenario == "wrong_scale_stride" ? PAYLOAD_BYTES : wireBytes;
                    std::copy_n(ub.begin() + tileBase + row * sourceStep + PAYLOAD_BYTES, sizeof(uint32_t),
                        outputScale.begin() + GUARD_BYTES + (run.outputRow + row) * sizeof(uint32_t));
                });
            }
        }
        if (scenario != "early_scale_credit") pipeline.MarkWritten(slot);
        slot = (slot + 1) % RUN_TILE_COUNT;
    }
    if (scenario != "missing_tail_drain") { pipeline.Drain(); pipeline.Drain(); }
    machine.work(SCALAR_PIPE, [&] {
        require_condition(outputPayload == goldenPayload && outputScale == goldenScale,
            "payload/scale byte oracle differs or publication precedes drain");
        require_condition(std::all_of(ub.begin(), ub.begin() + firstTile, [](uint8_t byte) { return byte == UNTOUCHED; })
            && std::all_of(ub.end() - GUARD_BYTES, ub.end(), [](uint8_t byte) { return byte == UNTOUCHED; }),
            "new tile aliases mandatory row/address slice or UB end guard");
    });
    machine.run(seed);
}

void run_case(const std::string &scenario, uint32_t seed)
{
    using namespace Cam::AsyncReceiveRuns;
    if (scenario == "planning") { check_planning(); return; }
    Fixture fixture;
    if (scenario == "empty_experts") fixture.startExpert = 0;
    if (scenario == "empty_aiv") {
        View view{{0, 4}, {0, 4}, {0, 0}, {}};
        ReceiveRunCursor<View> cursor(view);
        cursor.init(2, 2, 0, 1, 3);
        ReceiveRun run;
        require_condition(!cursor.next(run) && !has_contiguous_run(view, 2, 2, 0, 1, 3)
            && !plan_receive_runs(1024, 0, 64, 0).enabled, "empty AIV issues a source load or run tile");
        return;
    }
    if (scenario == "spare_fallback") {
        require_condition(!plan_receive_runs(255 + 32, 32, 64, 7).enabled, "insufficient dual tile budget loses P0 fallback");
        return;
    }
    if (scenario == "wire_wrap") {
        View view{{3}, {3}, {0}, {65535, 0, 1}};
        ReceiveRunCursor<View> cursor(view);
        cursor.init(0, 3, 0, 0, 3);
        ReceiveRun run;
        require_condition(cursor.next(run) && run.wireIndex == 65535 && run.rows == 1
            && cursor.next(run) && run.wireIndex == 0 && run.rows == 2, "uint16 wrap was coalesced into one run");
        return;
    }
    if (scenario == "random_fallback") {
        View view{{5}, {5}, {0}, {4, 8, 2, 9, 3}};
        require_condition(!has_contiguous_run(view, 0, 5, 0, 0, 3), "random addresses replace original cache/P0");
        return;
    }
    const bool quantized = scenario == "q1_runs" || scenario == "early_scale_credit"
        || scenario == "wrong_padding_stride" || scenario == "wrong_scale_stride";
    uint32_t outputRows;
    const auto all = raw_assignments(fixture, 0, outputRows);
    if (scenario == "aiv_boundaries") {
        constexpr uint32_t AIV_COUNT = 3;
        for (uint32_t aiv = 0; aiv < AIV_COUNT; ++aiv) {
            const uint32_t begin = static_cast<uint32_t>(all.size()) * aiv / AIV_COUNT;
            const uint32_t end = static_cast<uint32_t>(all.size()) * (aiv + 1) / AIV_COUNT;
            run_slice(fixture, 0, begin, end, false, scenario, seed + aiv);
        }
    } else if (scenario == "peer_isolation") {
        for (uint32_t peer = 0; peer < fixture.peerExperts.size(); ++peer) {
            const auto assignments = raw_assignments(fixture, peer, outputRows);
            run_slice(fixture, peer, 0, static_cast<uint32_t>(assignments.size()), true, scenario, seed + peer);
        }
    } else if (scenario == "partial_runs") {
        run_slice(fixture, 0, 1, static_cast<uint32_t>(all.size()) - 1, false, scenario, seed);
    } else {
        run_slice(fixture, 0, 0, static_cast<uint32_t>(all.size()), quantized, scenario, seed);
    }
}
} // namespace

int main(int argc, char **argv)
{
    try {
        require_condition(argc == 2, "one scenario required");
        const std::string scenario = argv[1];
        const bool negative = scenario == "unsafe_reuse" || scenario == "early_scale_credit"
            || scenario == "wrong_padding_stride" || scenario == "wrong_scale_stride" || scenario == "missing_tail_drain";
        bool detected = false;
        for (uint32_t seed = 0; seed < SCHEDULE_COUNT; ++seed) {
            try { run_case(scenario, seed); }
            catch (const std::runtime_error &error) {
                if (!negative) throw;
                const std::string reason = error.what();
                require_condition(reason == "tile overwritten before payload read" || reason == "tile overwritten before scale read"
                    || reason == "payload/scale byte oracle differs or publication precedes drain"
                    || reason == "event overwritten before consumption" || reason == "tile event lease survives drain",
                    "negative run failed for an unrelated reason");
                detected = true;
                break;
            }
        }
        require_condition(!negative || detected, "unsafe DMA mutation was not detected");
        std::cout << scenario << " passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
