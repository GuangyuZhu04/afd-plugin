// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cam_async_mte_pipeline.h"

namespace {
using Event = AscendC::HardEvent;
constexpr uint32_t PIPE_COUNT = 3;
constexpr uint32_t LOAD_PIPE = 0;
constexpr uint32_t STORE_PIPE = 1;
constexpr uint32_t SCALAR_PIPE = 2;
constexpr uint32_t SCHEDULE_COUNT = 256;
constexpr uint32_t REUSE_ROUNDS = 16;
constexpr uint64_t ROW_BYTES = 64;

void Require(bool condition, const char *reason)
{
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

struct Row {
    int payload;
    int scale;
};

struct Transfer {
    uint32_t slot;
    uint32_t source;
    uint32_t destination;
    uint32_t rows;
};

enum class Action { Signal, Wait, Load, Payload, Scale, Release, Publish };

struct Command {
    Action action;
    Event event{Event::MTE2_S};
    int id{0};
    Transfer transfer{};
};

uint32_t SourcePipe(Event event)
{
    return event == Event::MTE3_MTE2 ? STORE_PIPE : LOAD_PIPE;
}

uint32_t DestinationPipe(Event event)
{
    return event == Event::MTE2_MTE3 ? STORE_PIPE
        : event == Event::MTE3_MTE2 ? LOAD_PIPE : SCALAR_PIPE;
}

class Simulator {
public:
    explicit Simulator(bool quantized) : quantized_(quantized) {}

    int Allocate(Event event)
    {
        // Preexisting leases make fixed 0/1 event IDs incorrect in this model.
        int id = 2;
        while (leases_[{event, id}]) {
            ++id;
        }
        leases_[{event, id}] = true;
        return id;
    }

    void Append(uint32_t pipe, Command command)
    {
        queues_[pipe].push_back(command);
    }

    void PrepareSource(uint32_t rows, uint32_t epoch)
    {
        input_.clear();
        output_.assign(rows, Row{-1, -1});
        for (uint32_t row = 0; row < rows; ++row) {
            input_.push_back(Row{static_cast<int>(epoch * 101 + row * 7 + 3),
                static_cast<int>(epoch * 103 + row * 11 + 5)});
        }
    }

    void Run(uint32_t seed)
    {
        std::mt19937 random(seed);
        while (Pending()) {
            std::vector<uint32_t> ready;
            for (uint32_t pipe = 0; pipe < PIPE_COUNT; ++pipe) {
                if (!queues_[pipe].empty() && Ready(queues_[pipe].front())) {
                    ready.push_back(pipe);
                }
            }
            Require(!ready.empty(), "pipeline deadlock or unmatched wait");
            const uint32_t pipe = ready[random() % ready.size()];
            const Command command = queues_[pipe].front();
            queues_[pipe].pop_front();
            Execute(command);
        }
        for (const auto &token : tokens_) {
            Require(token.second == 0, "event token survives Drain");
        }
        for (const auto &lease : leases_) {
            Require(!lease.second, "event lease survives Drain");
        }
    }

    bool SawPrefetch() const
    {
        return sawPrefetch_;
    }

private:
    bool Pending() const
    {
        for (const auto &queue : queues_) {
            if (!queue.empty()) {
                return true;
            }
        }
        return false;
    }

    bool Ready(const Command &command)
    {
        return command.action != Action::Wait
            || tokens_[{command.event, command.id}] != 0;
    }

    void Execute(const Command &command)
    {
        const auto key = std::make_pair(command.event, command.id);
        const auto &transfer = command.transfer;
        switch (command.action) {
            case Action::Signal:
                Require(tokens_[key] == 0, "event signaled twice before consumption");
                tokens_[key] = 1;
                break;
            case Action::Wait:
                Require(tokens_[key] == 1, "wait consumed a missing event");
                tokens_[key] = 0;
                break;
            case Action::Load:
                if (!queues_[STORE_PIPE].empty()) {
                    sawPrefetch_ = true;
                }
                slots_[transfer.slot].assign(input_.begin() + transfer.source,
                    input_.begin() + transfer.source + transfer.rows);
                slotSource_[transfer.slot] = transfer.source;
                break;
            case Action::Payload:
            case Action::Scale:
                Require(slotSource_[transfer.slot] == transfer.source,
                    "slot overwritten while a previous store is pending");
                Require(slots_[transfer.slot].size() == transfer.rows,
                    "store observed an unready or differently sized load");
                for (uint32_t row = 0; row < transfer.rows; ++row) {
                    if (command.action == Action::Payload) {
                        output_[transfer.destination + row].payload =
                            slots_[transfer.slot][row].payload;
                    } else {
                        output_[transfer.destination + row].scale =
                            slots_[transfer.slot][row].scale;
                    }
                }
                break;
            case Action::Release:
                Require(leases_[key], "event released without its lease");
                Require(tokens_[key] == 0, "event released before token consumption");
                leases_[key] = false;
                break;
            case Action::Publish:
                for (uint32_t row = 0; row < input_.size(); ++row) {
                    Require(output_[row].payload == input_[row].payload,
                        "completion published before all payload stores");
                    if (quantized_) {
                        Require(output_[row].scale == input_[row].scale,
                            "completion published before all scale stores");
                    }
                }
                break;
        }
    }

    bool quantized_;
    bool sawPrefetch_{false};
    std::array<std::deque<Command>, PIPE_COUNT> queues_;
    std::map<std::pair<Event, int>, bool> leases_;
    std::map<std::pair<Event, int>, uint32_t> tokens_;
    std::array<std::vector<Row>, Cam::AsyncMte::BUFFER_COUNT> slots_;
    std::array<uint32_t, Cam::AsyncMte::BUFFER_COUNT> slotSource_{};
    std::vector<Row> input_;
    std::vector<Row> output_;
};

class SimulatorBackend {
public:
    using EventId = int;
    explicit SimulatorBackend(Simulator *simulator, bool omitScalarFence = false)
        : simulator_(simulator), omitScalarFence_(omitScalarFence) {}
    void Init(AscendC::TPipe *) {}

    template <Event event>
    int Allocate()
    {
        return simulator_->Allocate(event);
    }

    template <Event event>
    void Release(int id)
    {
        simulator_->Append(SCALAR_PIPE, {Action::Release, event, id});
    }

    template <Event event>
    void Signal(int id)
    {
        simulator_->Append(SourcePipe(event), {Action::Signal, event, id});
    }

    template <Event event>
    void Wait(int id)
    {
        simulator_->Append(DestinationPipe(event), {Action::Wait, event, id});
    }

    void FenceToScalar()
    {
        if (omitScalarFence_) {
            return;
        }
        Signal<Event::MTE2_S>(0);
        Wait<Event::MTE2_S>(0);
    }

private:
    Simulator *simulator_;
    bool omitScalarFence_;
};

using Pipeline = Cam::AsyncMte::PipelineEvents<SimulatorBackend>;

void RunTransfers(const std::vector<uint32_t> &cellRows, uint32_t rowsPerSlot,
    bool quantized, bool skipReuse, bool earlyCredit, bool omitScalarFence,
    Simulator &simulator)
{
    Pipeline pipeline{SimulatorBackend{&simulator, omitScalarFence}};
    pipeline.Init(nullptr);
    uint32_t cursor = 0;
    uint32_t slot = 0;
    for (const auto cell : cellRows) {
        for (uint32_t offset = 0; offset < cell;) {
            const uint32_t count = std::min(rowsPerSlot, cell - offset);
            const Transfer transfer{slot, cursor, cursor, count};
            if (!skipReuse) {
                pipeline.WaitForReuse(slot);
            }
            simulator.Append(LOAD_PIPE, {Action::Load, Event::MTE2_S, 0, transfer});
            pipeline.MarkLoaded(slot);
            pipeline.WaitLoaded(slot);
            simulator.Append(STORE_PIPE, {Action::Payload, Event::MTE2_S, 0, transfer});
            if (earlyCredit) {
                pipeline.MarkWritten(slot);
            }
            if (quantized) {
                simulator.Append(STORE_PIPE, {Action::Scale, Event::MTE2_S, 0, transfer});
            }
            if (!earlyCredit) {
                pipeline.MarkWritten(slot);
            }
            cursor += count;
            offset += count;
            slot = (slot + 1) % Cam::AsyncMte::BUFFER_COUNT;
        }
    }
    pipeline.Drain();
    // Idempotent Drain must neither wait on a consumed credit nor release twice.
    pipeline.Drain();
    simulator.Append(SCALAR_PIPE, {Action::Publish});
}

void CheckPlanning()
{
    using namespace Cam::AsyncMte;
    const auto single = PlanCombineSendBuffers(ROW_BYTES, ROW_BYTES, true);
    Require(!single.doubleBuffered && single.rowsPerSlot == 1, "one-row fallback lost");
    const auto disabled = PlanCombineSendBuffers(ROW_BYTES * 5, ROW_BYTES, false);
    Require(!disabled.doubleBuffered && disabled.rowsPerSlot == 5,
        "B0 single-buffer capacity was narrowed");
    for (uint64_t available = 0; available <= ROW_BYTES * 17; ++available) {
        const auto plan = PlanCombineSendBuffers(available, ROW_BYTES, true);
        const uint64_t slots = plan.doubleBuffered ? BUFFER_COUNT : 1;
        Require(plan.bytesPerSlot * slots <= available, "slot span exceeds UB");
        Require(plan.bytesPerSlot % ROW_BYTES == 0, "partial-row slot");
        Require(plan.doubleBuffered == (available >= ROW_BYTES * BUFFER_COUNT),
            "two rows do not select the expected branch");
    }
    Require(!CanUseReceiveSecondSlot(ROW_BYTES, ROW_BYTES, ROW_BYTES, 2),
        "address slice lost priority over optional row");
    Require(CanUseReceiveSecondSlot(ROW_BYTES * 2, ROW_BYTES, ROW_BYTES, 2),
        "exact second-row boundary rejected");
    Require(!CanUseReceiveSecondSlot(ROW_BYTES * 2, ROW_BYTES, ROW_BYTES, 1),
        "single row unnecessarily enables pipeline");
    Require(!CanUseReceiveSecondSlot(ROW_BYTES, ROW_BYTES + 1, ROW_BYTES, 2),
        "receive planner subtraction underflow");
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "one scenario is required");
        const std::string scenario = argv[1];
        if (scenario == "planning") {
            CheckPlanning();
            std::cout << "planning passed\n";
            return 0;
        }
        const bool earlyCredit = scenario == "early_quant_credit";
        const bool omitScalarFence = scenario == "missing_scalar_fence";
        const bool skipReuse = scenario == "missing_reuse_wait";
        const bool quantized = scenario == "quantized" || earlyCredit;
        const bool negative = skipReuse || earlyCredit || omitScalarFence;
        const std::vector<uint32_t> cells = scenario == "zero" ? std::vector<uint32_t>{0, 0}
            : scenario == "one" ? std::vector<uint32_t>{1}
            : scenario == "cell_boundaries" ? std::vector<uint32_t>{0, 3, 1, 0, 5}
            : std::vector<uint32_t>{7};
        uint32_t rows = 0;
        for (const auto cell : cells) {
            rows += cell;
        }
        bool sawPrefetch = false;
        bool detected = false;
        for (uint32_t seed = 0; seed < SCHEDULE_COUNT; ++seed) {
            Simulator simulator(quantized);
            const uint32_t rounds = scenario == "reuse_rounds" ? REUSE_ROUNDS : 1;
            for (uint32_t round = 0; round < rounds; ++round) {
                simulator.PrepareSource(rows, round);
                RunTransfers(cells, scenario == "odd_tile_tail" ? 3 : 1,
                    quantized, skipReuse, earlyCredit, omitScalarFence, simulator);
                try {
                    simulator.Run(seed + round * SCHEDULE_COUNT);
                } catch (const std::runtime_error &) {
                    if (!negative) {
                        throw;
                    }
                    detected = true;
                    break;
                }
                sawPrefetch = sawPrefetch || simulator.SawPrefetch();
            }
        }
        if (negative) {
            Require(detected, "simulator failed to detect an invalid lifecycle");
        } else if (rows > 1) {
            Require(sawPrefetch, "all schedules serialized load behind store");
        }
        std::cout << scenario << " passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
