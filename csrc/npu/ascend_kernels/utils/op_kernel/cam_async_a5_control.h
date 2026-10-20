// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_A5_CONTROL_H
#define AFD_CAM_ASYNC_A5_CONTROL_H
#include "kernel_operator.h"
#define AFD_CAM_A5_INLINE __aicore__ inline
#include "cam_async_a5_contract.h"
#undef AFD_CAM_A5_INLINE

namespace Cam::AsyncA5 {
constexpr uint32_t CONTROL_SCRATCH_BYTES = 512;
constexpr uint32_t WORD_BYTES = sizeof(uint64_t);
constexpr uint32_t RECORD_WORDS = sizeof(AsyncPipeline::SlotControlRecord) / WORD_BYTES;
constexpr uint32_t COMMIT_WORD = RECORD_WORDS - 1;
constexpr uint32_t HEADER_EPOCH_WORD = 2;
constexpr uint32_t HEADER_CURSOR_WORD = 3;
constexpr uint32_t TRANSPORT_STATUS_WORD = 1;

// One AIV coordinates each control phase; the surrounding entry synchronizes
// all launched AIVs before any data kernel touches this invocation's slot.
// Record publication invalidates commit, copies the body, then publishes epoch.
// A reader samples commit/body/commit through MTE2, not cached scalar GM loads.
// These are device operations, not claims inferred from a successful host launch.
class Control {
public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR arguments, uint64_t call, GM_ADDR context)
    {
        pipe_ = pipe;
        arguments_ = arguments;
        call_ = call;
        pipe_->InitBuffer(storage_, CONTROL_SCRATCH_BYTES);
        words_ = storage_.Get<uint64_t>();
        LoadWords(arguments_, ARGUMENT_WORDS);
        auto *destination = reinterpret_cast<uint64_t *>(&args_);
        for (uint32_t i = 0; i < ARGUMENT_WORDS; ++i) destination[i] = words_.GetValue(i);
        context_ = (__gm__ AscendC::HcclCombineOpParam *)context;
        if (!ValidateArgs(args_) || CallSlots(call_) != args_.slotCount
            || !PlanLayout(args_.windowBytes, args_.attentionRanks + args_.moeRanks,
                args_.slotCount, args_.domain, layout_)) Fail(Error::InvalidContext);
    }

    __aicore__ inline RuntimeArgs &Args() { return args_; }
    __aicore__ inline const AsyncPipeline::RingLayout &Layout() const { return layout_; }
    __aicore__ inline GM_ADDR Window(uint64_t rank) const
    {
        return reinterpret_cast<GM_ADDR>(context_->windowsIn[rank]);
    }
    __aicore__ inline bool Ok() const { return args_.error == 0; }

    __aicore__ inline void Save()
    {
        const auto *source = reinterpret_cast<const uint64_t *>(&args_);
        for (uint32_t i = 0; i < ARGUMENT_WORDS; ++i) words_.SetValue(i, source[i]);
        StoreWords(arguments_, ARGUMENT_WORDS);
    }
    __aicore__ inline void Fail(Error error) { args_.error = static_cast<uint64_t>(error); }

    __aicore__ inline void Initialize(uint64_t physicalUbBytes, uint64_t activeAivs)
    {
        args_.localWindow = reinterpret_cast<uint64_t>(Window(args_.localRank));
        args_.physicalUbBytes = physicalUbBytes;
        args_.activeAivs = activeAivs;
        Save();
    }

    __aicore__ inline void BeforeDispatch()
    {
        args_.slot = CallSlot(call_);
        if (args_.slot >= args_.slotCount) { Fail(Error::InvalidContext); return; }
        GM_ADDR const local = Window(args_.localRank);
        uint64_t readyOffset = 0, creditOffset = 0;
        AsyncPipeline::ControlOffset(layout_, args_.slot, AsyncPipeline::RecordRole::ProducerReady, readyOffset);
        AsyncPipeline::ControlOffset(layout_, args_.slot, AsyncPipeline::RecordRole::ReceiverCredit, creditOffset);
        AsyncPipeline::SlotControlRecord previous{};
        const bool occupied = ReadRecord(local + readyOffset, previous);
        if (occupied) {
            if (!Matches(previous, previous.epoch, args_.slot)) { Fail(Error::InvalidRecord); return; }
            AsyncPipeline::SlotControlRecord credit{};
            bool acquired = false;
            for (uint64_t poll = 0; poll < POLL_LIMIT && Ok(); ++poll) {
                if (Aborted()) break;
                if (ReadRecord(local + creditOffset, credit)
                    && Matches(credit, previous.epoch, args_.slot)
                    && credit.state == static_cast<uint64_t>(AsyncPipeline::SlotState::Consumed)
                    && (credit.boundaryFlags & AsyncPipeline::ALL_BOUNDARY_FLAGS)
                        == AsyncPipeline::ALL_BOUNDARY_FLAGS) { acquired = true; break; }
            }
            if (!acquired && Ok()) Fail(Error::Timeout);
            if (!Ok()) return;
        }
        const uint64_t epoch = ReadWord(local + HEADER_EPOCH_WORD * WORD_BYTES);
        if (epoch >= MAX_EPOCH) { Fail(Error::StaleEpoch); return; }
        args_.epoch = epoch + 1;
        args_.attentionGroup = args_.localRank / args_.tpSize * args_.tpSize;
        WriteWord(local + HEADER_EPOCH_WORD * WORD_BYTES, args_.epoch);
    }

    __aicore__ inline void AfterDispatch()
    {
        const auto record = MakeRecord(AsyncPipeline::SlotState::Ready,
            AsyncPipeline::SOURCE_CONSUMED | AsyncPipeline::REMOTE_VISIBLE);
        for (uint64_t rank = args_.attentionRanks; rank < args_.attentionRanks + args_.moeRanks; ++rank)
            WriteRecord(Window(rank) + WriterRecordOffset(layout_, args_.slot, args_.localRank,
                AsyncPipeline::RecordRole::ProducerReady), record);
        uint64_t offset = 0;
        AsyncPipeline::ControlOffset(layout_, args_.slot, AsyncPipeline::RecordRole::ProducerReady, offset);
        WriteRecord(Window(args_.localRank) + offset, record);
    }

    __aicore__ inline void BeforeReceive()
    {
        GM_ADDR const local = Window(args_.localRank);
        const uint64_t groups = args_.attentionRanks / args_.tpSize;
        const uint64_t candidates = groups * args_.slotCount;
        const uint64_t first = ReadWord(local + HEADER_CURSOR_WORD * WORD_BYTES) % candidates;
        for (uint64_t poll = 0; poll < POLL_LIMIT && Ok(); ++poll) {
            if (Aborted()) break;
            for (uint64_t offset = 0; offset < candidates; ++offset) {
                    const uint64_t candidate = (first + offset) % candidates;
                    const uint64_t slot = candidate / groups;
                    const uint64_t group = candidate % groups * args_.tpSize;
                    uint64_t epoch = 0;
                    bool ready = true;
                    for (uint64_t tp = 0; tp < args_.tpSize; ++tp) {
                        AsyncPipeline::SlotControlRecord record{};
                        if (!ReadRecord(local + WriterRecordOffset(layout_, slot, group + tp,
                                AsyncPipeline::RecordRole::ProducerReady), record)
                            || record.state != static_cast<uint64_t>(AsyncPipeline::SlotState::Ready)) {
                            ready = false;
                            break;
                        }
                        if (!Matches(record, record.epoch, slot) || (tp != 0 && record.epoch != epoch)) {
                            Fail(Error::InvalidRecord);
                            return;
                        }
                        epoch = record.epoch;
                    }
                    if (ready) {
                        args_.slot = slot;
                        args_.epoch = epoch;
                        args_.attentionGroup = group;
                        WriteWord(local + HEADER_CURSOR_WORD * WORD_BYTES, candidate);
                        // The original DR's first candidate is this complete TP
                        // group. Only one receive stream owns an FFN session.
                        WriteWord(local + FrameOffset(layout_, slot) + layout_.payloadCapacity - 32, group);
                        return;
                    }
            }
        }
        if (Ok()) Fail(Error::Timeout);
    }

    __aicore__ inline void AfterReceive(GM_ADDR metadata)
    {
        LoadWords(metadata, 5);
        if (words_.GetValue(1) != args_.attentionGroup) { Fail(Error::InvalidMetadata); return; }
        args_.lastChunk = words_.GetValue(4) + 1 == args_.expertsPerRank;
        if (!args_.lastChunk) return;
        const auto consumed = MakeRecord(AsyncPipeline::SlotState::Consumed,
            AsyncPipeline::SOURCE_CONSUMED | AsyncPipeline::REMOTE_VISIBLE
                | AsyncPipeline::RECEIVER_CONSUMED);
        for (uint64_t tp = 0; tp < args_.tpSize; ++tp)
            WriteRecord(Window(args_.localRank) + WriterRecordOffset(layout_, args_.slot,
                args_.attentionGroup + tp, AsyncPipeline::RecordRole::ProducerReady), consumed);
        const uint64_t groups = args_.attentionRanks / args_.tpSize;
        WriteWord(Window(args_.localRank) + HEADER_CURSOR_WORD * WORD_BYTES,
            (args_.slot * groups + args_.attentionGroup / args_.tpSize + 1) % (groups * args_.slotCount));
    }

    // Runs on the same runtime stream AFTER all CCU tasks. The original prefix
    // ready is published only for the final expert chunk, including an empty one.
    __aicore__ inline void PublishCombine(GM_ADDR metadata)
    {
        if (ReadWord(reinterpret_cast<GM_ADDR>(args_.planStatus)) != 0
            || ReadWord(reinterpret_cast<GM_ADDR>(args_.planStatus) + TRANSPORT_STATUS_WORD * WORD_BYTES) != 0) {
            Fail(Error::InvalidMetadata);
            return;
        }
        LoadWords(metadata, 5);
        if (words_.GetValue(1) != args_.attentionGroup) { Fail(Error::InvalidMetadata); return; }
        if (words_.GetValue(4) + 1 != args_.expertsPerRank) return;
        const uint64_t statistics = Align32(4 * (args_.moeRanks + args_.moeRanks * args_.expertsPerRank) * 48);
        const auto record = MakeRecord(AsyncPipeline::SlotState::Ready,
            AsyncPipeline::SOURCE_CONSUMED | AsyncPipeline::REMOTE_VISIBLE | AsyncPipeline::TASK_RETIRED);
        for (uint64_t tp = 0; tp < args_.tpSize; ++tp) {
            const uint64_t rank = args_.attentionGroup + tp;
            const uint64_t prefix = ReadWord(metadata + (5 + tp) * WORD_BYTES);
            GM_ADDR const flag = Window(rank) + FrameOffset(layout_, args_.slot) + statistics
                + sizeof(uint32_t) * (args_.localRank - args_.attentionRanks);
            WriteUint32(flag, static_cast<uint32_t>(prefix + 1));
            WriteRecord(Window(rank) + WriterRecordOffset(layout_, args_.slot, args_.localRank,
                AsyncPipeline::RecordRole::ProducerReady), record);
        }
    }

    __aicore__ inline void BeforeCombine()
    {
        for (uint64_t peer = args_.attentionRanks; peer < args_.attentionRanks + args_.moeRanks; ++peer) {
            bool ready = false;
            for (uint64_t poll = 0; poll < POLL_LIMIT && Ok(); ++poll) {
                if (Aborted()) break;
                AsyncPipeline::SlotControlRecord record{};
                if (ReadRecord(Window(args_.localRank) + WriterRecordOffset(layout_, args_.slot, peer,
                        AsyncPipeline::RecordRole::ProducerReady), record)
                    && record.state == static_cast<uint64_t>(AsyncPipeline::SlotState::Ready)
                    && Matches(record, args_.epoch, args_.slot)) { ready = true; break; }
            }
            if (!ready && Ok()) Fail(Error::Timeout);
            if (!Ok()) return;
        }
    }

    __aicore__ inline void AfterCombine()
    {
        // The data kernel has drained every input/output slot and synchronized
        // its AIVs. This profile retains the whole round until ordered reduction
        // finishes; it does not return a credit merely because a copy was issued.
        const auto credit = MakeRecord(AsyncPipeline::SlotState::Consumed, AsyncPipeline::ALL_BOUNDARY_FLAGS);
        for (uint64_t peer = args_.attentionRanks; peer < args_.attentionRanks + args_.moeRanks; ++peer)
            WriteRecord(Window(peer) + WriterRecordOffset(layout_, args_.slot, args_.localRank,
                AsyncPipeline::RecordRole::ReceiverCredit), credit);
        uint64_t offset = 0;
        AsyncPipeline::ControlOffset(layout_, args_.slot, AsyncPipeline::RecordRole::ReceiverCredit, offset);
        WriteRecord(Window(args_.localRank) + offset, credit);
    }

    __aicore__ inline uint64_t ReadWord(GM_ADDR source)
    {
        LoadWords(source, 1);
        return words_.GetValue(0);
    }
    __aicore__ inline void WriteWord(GM_ADDR destination, uint64_t value)
    {
        words_.SetValue(0, value);
        StoreWords(destination, 1);
    }

private:
    template <AscendC::HardEvent Event>
    __aicore__ inline void Fence()
    {
        const auto id = pipe_->FetchEventID<Event>();
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
    }
    __aicore__ inline uint64_t Align32(uint64_t bytes) const { return (bytes + 31) / 32 * 32; }
    __aicore__ inline void LoadWords(GM_ADDR source, uint32_t count)
    {
        AscendC::GlobalTensor<uint64_t> input;
        input.SetGlobalBuffer((__gm__ uint64_t *)source);
        AscendC::DataCopyPad(words_, input, {1U, count * WORD_BYTES, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        Fence<AscendC::HardEvent::MTE2_S>();
        Fence<AscendC::HardEvent::S_MTE2>();
    }
    __aicore__ inline void StoreWords(GM_ADDR destination, uint32_t count)
    {
        AscendC::GlobalTensor<uint64_t> output;
        output.SetGlobalBuffer((__gm__ uint64_t *)destination);
        Fence<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(output, words_, {1U, count * WORD_BYTES, 0U, 0U, 0U});
        Fence<AscendC::HardEvent::MTE3_S>();
    }
    __aicore__ inline void WriteUint32(GM_ADDR destination, uint32_t value)
    {
        auto word = words_.ReinterpretCast<uint32_t>();
        word.SetValue(0, value);
        AscendC::GlobalTensor<uint32_t> output;
        output.SetGlobalBuffer((__gm__ uint32_t *)destination);
        Fence<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(output, word, {1U, sizeof(uint32_t), 0U, 0U, 0U});
        Fence<AscendC::HardEvent::MTE3_S>();
    }
    __aicore__ inline bool Aborted()
    {
        if (args_.abortAddress != 0 && ReadWord(reinterpret_cast<GM_ADDR>(args_.abortAddress)) != 0)
            Fail(Error::Aborted);
        return !Ok();
    }
    __aicore__ inline bool ReadRecord(GM_ADDR address, AsyncPipeline::SlotControlRecord &record)
    {
        const uint64_t commit = ReadWord(address + COMMIT_WORD * WORD_BYTES);
        if (commit == 0) return false;
        LoadWords(address, RECORD_WORDS);
        auto *output = reinterpret_cast<uint64_t *>(&record);
        for (uint32_t i = 0; i < RECORD_WORDS; ++i) output[i] = words_.GetValue(i);
        return CommitSampleMatches(commit, record, ReadWord(address + COMMIT_WORD * WORD_BYTES));
    }
    __aicore__ inline void WriteRecord(GM_ADDR address, const AsyncPipeline::SlotControlRecord &record)
    {
        WriteWord(address + COMMIT_WORD * WORD_BYTES, 0);
        const auto *input = reinterpret_cast<const uint64_t *>(&record);
        for (uint32_t i = 0; i < COMMIT_WORD; ++i) words_.SetValue(i, input[i]);
        StoreWords(address, COMMIT_WORD);
        WriteWord(address + COMMIT_WORD * WORD_BYTES, record.epoch);
    }
    __aicore__ inline bool Matches(const AsyncPipeline::SlotControlRecord &record,
        uint64_t epoch, uint64_t slot) const
    {
        return FrameRecordMatches(record, args_, layout_, epoch, slot);
    }
    __aicore__ inline AsyncPipeline::SlotControlRecord MakeRecord(AsyncPipeline::SlotState state,
        uint64_t boundaries) const
    {
        return FrameRecord(args_, layout_, state, boundaries);
    }
    AscendC::TPipe *pipe_{nullptr};
    GM_ADDR arguments_{nullptr};
    __gm__ AscendC::HcclCombineOpParam *context_{nullptr};
    uint64_t call_{0};
    AscendC::TBuf<> storage_;
    AscendC::LocalTensor<uint64_t> words_;
    RuntimeArgs args_{};
    AsyncPipeline::RingLayout layout_{};
};

// These phases bracket the unchanged numerical kernels. Reset is safe here:
// the preceding phase has completed all MTE accesses and leased-event drains.
__aicore__ inline bool BeforeCall(AscendC::TPipe &pipe, GM_ADDR arguments,
    uint64_t call, uint64_t ubBytes, uint64_t activeAivs)
{
    if (!IsFullCall(call)) return true;
    if (AscendC::GetBlockIdx() == 0) {
        Control control;
        control.Init(&pipe, arguments, call,
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
        if (control.Ok()) {
            switch (CallMode(call)) {
                case Mode::Initialize: control.Initialize(ubBytes, activeAivs); break;
                case Mode::Dispatch: control.BeforeDispatch(); break;
                case Mode::Receive: control.BeforeReceive(); break;
                case Mode::Combine: control.BeforeCombine(); break;
                default: break;
            }
        }
        control.Save();
    }
    AscendC::SyncAll<true>();
    pipe.Reset();
    bool run;
    bool failed;
    {
        Control control;
        control.Init(&pipe, arguments, call,
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
        run = control.Ok() && CallMode(call) != Mode::Initialize;
        failed = !control.Ok();
    }
    pipe.Reset();
    if (failed) AscendC::Trap();
    return run;
}

__aicore__ inline void AfterCall(AscendC::TPipe &pipe, GM_ADDR arguments,
    uint64_t call, GM_ADDR metadata = nullptr)
{
    if (!IsFullCall(call)) return;
    AscendC::SyncAll<true>();
    pipe.Reset();
    if (AscendC::GetBlockIdx() == 0) {
        Control control;
        control.Init(&pipe, arguments, call,
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
        if (control.Ok()) {
            switch (CallMode(call)) {
                case Mode::Dispatch: control.AfterDispatch(); break;
                case Mode::Receive: control.AfterReceive(metadata); break;
                case Mode::Combine: control.AfterCombine(); break;
                case Mode::PublishCombine: control.PublishCombine(metadata); break;
                default: break;
            }
        }
        control.Save();
    }
    AscendC::SyncAll<true>();
    pipe.Reset();
    {
        Control control;
        control.Init(&pipe, arguments, call,
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
        if (!control.Ok()) AscendC::Trap();
    }
}

__aicore__ inline uint64_t GetFrameOffset(AscendC::TPipe &pipe, GM_ADDR arguments, uint64_t call)
{
    if (!IsFullCall(call)) return 0;
    uint64_t offset;
    {
        Control control;
        control.Init(&pipe, arguments, call,
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
        offset = FrameOffset(control.Layout(), control.Args().slot);
    }
    pipe.Reset();
    return offset;
}
} // namespace Cam::AsyncA5
#endif // AFD_CAM_ASYNC_A5_CONTROL_H
