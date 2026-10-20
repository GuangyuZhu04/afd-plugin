// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cam_async_a5_control.h"

namespace {
using Cam::AsyncA5::Control;
using Cam::AsyncA5::Error;
using Cam::AsyncA5::Mode;
using Cam::AsyncA5::RuntimeArgs;
constexpr uint64_t RANKS = 4;
constexpr uint64_t ATTENTION = 2;
constexpr uint64_t WINDOW_BYTES = 1ULL << 20;
constexpr uint64_t SOURCE = 0;
constexpr uint64_t FIRST_FFN = 2;

void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

struct World {
    std::array<std::vector<uint64_t>, RANKS> windows;
    std::array<uint64_t, RANKS> abort{};
    std::array<std::array<uint64_t, 2>, RANKS> status{};
    std::array<AscendC::MemoryHooks, RANKS> hooks;
    AscendC::HcclCombineOpParam context;
    World()
    {
        for (uint64_t rank = 0; rank < RANKS; ++rank) {
            windows[rank].resize(WINDOW_BYTES / sizeof(uint64_t));
            context.windowsIn[rank] = reinterpret_cast<uint64_t>(windows[rank].data());
        }
    }
    RuntimeArgs Args(uint64_t rank)
    {
        RuntimeArgs args;
        args.domain = 53;
        args.windowBytes = WINDOW_BYTES;
        args.slotCount = 2;
        args.localRank = rank;
        args.attentionRanks = ATTENTION;
        args.moeRanks = RANKS - ATTENTION;
        args.tpSize = 2;
        args.expertsPerRank = 2;
        args.hiddenSize = 32;
        args.topk = 2;
        args.abortAddress = reinterpret_cast<uint64_t>(&abort[rank]);
        args.planStatus = reinterpret_cast<uint64_t>(status[rank].data());
        return args;
    }
    RuntimeArgs Send(uint64_t rank, uint64_t slot)
    {
        auto args = Args(rank);
        AscendC::TPipe pipe(&hooks[rank]);
        Control control;
        control.Init(&pipe, reinterpret_cast<uint8_t *>(&args),
            Cam::AsyncA5::EncodeCall(Mode::Dispatch, args.slotCount, slot), reinterpret_cast<uint8_t *>(&context));
        control.BeforeDispatch();
        if (control.Ok()) control.AfterDispatch();
        control.Save();
        return args;
    }
    RuntimeArgs Receive(uint64_t rank, bool final = true)
    {
        auto args = Args(rank);
        AscendC::TPipe pipe(&hooks[rank]);
        Control control;
        control.Init(&pipe, reinterpret_cast<uint8_t *>(&args),
            Cam::AsyncA5::EncodeCall(Mode::Receive, args.slotCount), reinterpret_cast<uint8_t *>(&context));
        control.BeforeReceive();
        if (control.Ok()) {
            const std::array<uint64_t, 5> metadata{4, control.Args().attentionGroup, 7, 0, final ? 1U : 0U};
            control.AfterReceive(reinterpret_cast<uint8_t *>(const_cast<uint64_t *>(metadata.data())));
        }
        control.Save();
        return args;
    }
    RuntimeArgs Publish(RuntimeArgs args)
    {
        AscendC::TPipe pipe(&hooks[args.localRank]);
        Control control;
        control.Init(&pipe, reinterpret_cast<uint8_t *>(&args),
            Cam::AsyncA5::EncodeCall(Mode::PublishCombine, args.slotCount), reinterpret_cast<uint8_t *>(&context));
        std::array<uint64_t, 7> metadata{4, args.attentionGroup, 7, 0, 1, 0, 0};
        control.PublishCombine(reinterpret_cast<uint8_t *>(metadata.data()));
        control.Save();
        return args;
    }
    RuntimeArgs Combine(RuntimeArgs args)
    {
        AscendC::TPipe pipe(&hooks[args.localRank]);
        Control control;
        control.Init(&pipe, reinterpret_cast<uint8_t *>(&args),
            Cam::AsyncA5::EncodeCall(Mode::Combine, args.slotCount), reinterpret_cast<uint8_t *>(&context));
        control.BeforeCombine();
        if (control.Ok()) control.AfterCombine();
        control.Save();
        return args;
    }
    void AbortAfterReads(uint64_t rank, uint64_t count)
    {
        hooks[rank].beforeRead = [this, rank, count, reads = uint64_t{0}](uint64_t, uint64_t) mutable {
            if (++reads >= count) abort[rank] = 1;
        };
    }
};

void Cycle()
{
    World world;
    auto first = world.Send(0, 0);
    auto second = world.Send(1, 0);
    Require(first.epoch == 1 && second.epoch == 1, "paired source epoch");
    auto left = world.Receive(2);
    auto right = world.Receive(3);
    Require(left.error == 0 && right.error == 0 && left.slot == 0 && right.slot == 0, "receive selection");
    Require(world.Publish(left).error == 0 && world.Publish(right).error == 0, "completion publication");
    Require(world.Combine(first).error == 0 && world.Combine(second).error == 0, "ordered consumption");
    Require(world.Send(0, 0).epoch == 2 && world.Send(1, 0).epoch == 2, "credited reuse");
}
void PartialChunk()
{
    World world;
    world.Send(0, 0);
    world.Send(1, 0);
    world.Send(0, 1);
    world.Send(1, 1);
    const auto partial = world.Receive(2, false);
    Require(partial.error == 0 && partial.slot == 0 && partial.lastChunk == 0, "partial remains selected");
    const auto final = world.Receive(2, true);
    Require(final.error == 0 && final.slot == 0 && final.epoch == partial.epoch, "same chunk lease");
    const auto next = world.Receive(2, true);
    Require(next.error == 0 && next.slot == 1 && next.epoch == 2, "fair next slot");
    // An older work item's immutable ticket is still publishable after another
    // slot has been selected. A shared mutable receive context would fail this.
    Require(world.Publish(final).error == 0, "retained receive ticket");
}
void MissingTransportCompletion()
{
    World world;
    const auto first = world.Send(0, 0);
    world.Send(1, 0);
    const auto received = world.Receive(2);
    world.status[2][1] = 1;  // Planner succeeded; CCU task retirement is absent.
    Require(world.Publish(received).error == static_cast<uint64_t>(Error::InvalidMetadata), "no premature ready");
    world.AbortAfterReads(SOURCE, 20);
    Require(world.Combine(first).error == static_cast<uint64_t>(Error::Aborted), "receiver did not accept missing peer");
}
void EpochMismatch()
{
    World world;
    world.Send(0, 0);
    world.windows[1][Cam::AsyncA5::HEADER_EPOCH_WORD] = 1;
    world.Send(1, 0);
    Require(world.Receive(FIRST_FFN).error == static_cast<uint64_t>(Error::InvalidRecord), "mixed TP epochs rejected");
}
void CreditExhaustion()
{
    World world;
    world.Send(0, 0);
    world.AbortAfterReads(SOURCE, 20);
    const auto blocked = world.Send(0, 0);
    Require(blocked.error == static_cast<uint64_t>(Error::Aborted), "no uncredited slot reuse");
    Require(world.windows[0][Cam::AsyncA5::HEADER_EPOCH_WORD] == 1, "blocked issue cannot advance epoch");
}
void AbortSelection()
{
    World world;
    world.AbortAfterReads(FIRST_FFN, 12);
    Require(world.Receive(FIRST_FFN).error == static_cast<uint64_t>(Error::Aborted), "empty poll observes abort");
}
} // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "scenario required");
        const std::string scenario = argv[1];
        if (scenario == "cycle") Cycle();
        else if (scenario == "partial_chunk") PartialChunk();
        else if (scenario == "transport_marker") MissingTransportCompletion();
        else if (scenario == "tp_epoch") EpochMismatch();
        else if (scenario == "credit_exhaustion") CreditExhaustion();
        else if (scenario == "abort_selection") AbortSelection();
        else throw std::runtime_error("unknown scenario");
        std::cout << scenario << " passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
