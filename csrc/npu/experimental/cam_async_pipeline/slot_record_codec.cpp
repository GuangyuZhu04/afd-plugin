// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#define CAM_ASYNC_PIPELINE_INLINE __aicore__ inline
#include "ring_layout.h"
#undef CAM_ASYNC_PIPELINE_INLINE

namespace Cam::AsyncPipeline {
constexpr uint64_t STATUS_BYTES = 32;
struct StagingWordBackend {
    static __aicore__ inline uint64_t ReadWord(AscendC::LocalTensor<uint64_t> &input, uint64_t index)
    {
        return input.GetValue(index);
    }
};

template <AscendC::HardEvent Event>
__aicore__ inline void CodecSynchronize(AscendC::TPipe &pipe)
{
    const auto id = pipe.FetchEventID<Event>();
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}
} // namespace Cam::AsyncPipeline

// Single-AIV immutable-snapshot codec, not a distributed publication protocol.
// sourceRecord is a completed local staging snapshot owned by the caller. The
// caller establishes actual release/acquire/cache/transport ordering separately.
// copyValidatedRecord=1 copies to another dedicated staging allocation only;
// it must not be used as proof of remote ready or receiver credit. This 128B
// MTE copy is not atomic and no concurrent writer may change source/target.
extern "C" __global__ __aicore__ void afd_async_slot_record_codec(
    GM_ADDR sourceRecord, GM_ADDR targetStaging, GM_ADDR statusOutput,
    uint64_t domainId, uint64_t slotCount, uint64_t payloadCapacity,
    uint64_t metadataCapacity, uint64_t registeredBytes, uint64_t expectedSlot,
    uint64_t expectedEpoch, uint64_t expectedRequest, uint64_t expectedChunk,
    uint64_t recordRole, uint64_t copyValidatedRecord)
{
    using namespace AscendC;
    using namespace Cam::AsyncPipeline;
    if (GetBlockIdx() != 0) return;
    TPipe pipe;
    TBuf<> recordStorage, statusStorage;
    pipe.InitBuffer(recordStorage, sizeof(SlotControlRecord));
    pipe.InitBuffer(statusStorage, STATUS_BYTES);
    auto local = recordStorage.Get<uint64_t>();
    auto status = statusStorage.Get<uint64_t>();
    GlobalTensor<uint64_t> input, target, result;
    input.SetGlobalBuffer((__gm__ uint64_t *)sourceRecord);
    target.SetGlobalBuffer((__gm__ uint64_t *)targetStaging);
    result.SetGlobalBuffer((__gm__ uint64_t *)statusOutput);
    RingLayout layout;
    const ProtocolConfig config{domainId, slotCount, payloadCapacity, metadataCapacity};
    WireStatus code = BuildRingLayout(config, registeredBytes, layout);
    if (code == WireStatus::Success) {
        DataCopyPad(local, input, {1U, sizeof(SlotControlRecord), 0U, 0U, 0U}, {false, 0U, 0U, 0});
        CodecSynchronize<HardEvent::MTE2_S>(pipe);
        CodecSynchronize<HardEvent::S_MTE2>(pipe);
        SlotControlRecord snapshot{};
        DecodeControlWords<StagingWordBackend>(local, snapshot);
        const Lease expected{domainId, expectedSlot, expectedEpoch, expectedRequest, expectedChunk};
        code = ValidateWireRecord(snapshot, layout, expected, static_cast<RecordRole>(recordRole));
        if (code == WireStatus::Success && copyValidatedRecord == 1) {
            CodecSynchronize<HardEvent::S_MTE3>(pipe);
            DataCopyPad(target, local, {1U, sizeof(SlotControlRecord), 0U, 0U, 0U});
            CodecSynchronize<HardEvent::MTE3_S>(pipe);
        }
    }
    status.SetValue(0, static_cast<uint64_t>(code));
    CodecSynchronize<HardEvent::S_MTE3>(pipe);
    DataCopyPad(result, status, {1U, sizeof(uint64_t), 0U, 0U, 0U});
    CodecSynchronize<HardEvent::MTE3_S>(pipe);
}
