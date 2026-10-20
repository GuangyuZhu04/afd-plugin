// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "cam_async_a5_transport.h"

#include <memory>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "hccl.h"
#include "hccl_rank_graph.h"
#include "hcom.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "../experimental/cam_async_ccu/ccu_segment_kernel.h"

namespace {
namespace CcuRep = hcomm::CcuRep;
constexpr uint32_t TOPOLOGY_LAYER = 0;
constexpr uint32_t MAX_FULL_RANKS = 64;
constexpr uint32_t THREAD_NOTIFICATIONS = 0;

class BatchKernelArg final : public hcomm::CcuKernelArg {
public:
    BatchKernelArg(ChannelHandle channel, uint32_t peer, uint32_t count)
        : peer(peer), count(count) { channels.push_back(channel); }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        signature.Append("afd_a5_full_combine_v1|");
        signature.Append(peer);
        signature.Append('|');
        signature.Append(count);
        signature.Append('|');
        signature.Append(channels.at(0));
        return signature;
    }
    const uint32_t peer;
    const uint32_t count;
};

// One runtime task per peer, not per expert. A bounded CCU loop consumes that
// peer's fixed descriptor row; dynamic counts/offsets never travel to the host.
class BatchKernel final : public hcomm::CcuKernel {
public:
    explicit BatchKernel(const hcomm::CcuKernelArg &arg)
        : hcomm::CcuKernel(arg), peer_(static_cast<const BatchKernelArg &>(arg).peer),
          count_(static_cast<const BatchKernelArg &>(arg).count) {}
protected:
    HcclResult Algorithm() override
    {
        auto descriptor = CreateVariable();
        auto sourceBase = CreateVariable();
        auto destinationBase = CreateVariable();
        auto local = CreateLocalAddr();
        auto remote = CreateRemoteAddr();
        auto statusAddress = CreateVariable();
        auto planStatus = CreateVariable();
        Load(descriptor);
        Load(sourceBase);
        Load(local.token);
        Load(destinationBase);
        Load(remote.token);
        Load(statusAddress);
        LoadVariable(statusAddress, planStatus);
        auto offset = CreateVariable();
        auto field = CreateVariable();
        auto sourceOffset = CreateVariable();
        auto destinationOffset = CreateVariable();
        auto byteCount = CreateVariable();
        auto destinationRank = CreateVariable();
        auto cursor = CreateVariable();
        auto one = CreateVariable();
        auto stride = CreateVariable();
        cursor = 0;
        one = 1;
        stride = sizeof(Cam::AsyncCcu::SendSegment);
        auto completed = CreateCompletedEvent();
        CCU_IF(planStatus == 0) {
            CCU_WHILE(cursor != count_) {
                LoadVariable(descriptor, sourceOffset);
                offset = offsetof(Cam::AsyncCcu::SendSegment, dstByteOffset);
                field = descriptor + offset;
                LoadVariable(field, destinationOffset);
                offset = offsetof(Cam::AsyncCcu::SendSegment, byteCount);
                field = descriptor + offset;
                LoadVariable(field, byteCount);
                offset = offsetof(Cam::AsyncCcu::SendSegment, dstRank);
                field = descriptor + offset;
                LoadVariable(field, destinationRank);
                CCU_IF(destinationRank == peer_) {
                    CCU_IF(byteCount != 0) {
                        local.addr = sourceBase;
                        local.addr += sourceOffset;
                        remote.addr = destinationBase;
                        remote.addr += destinationOffset;
                        auto result = WriteNb(channels_[0], remote, local, byteCount, completed);
                        if (result != HCCL_SUCCESS) return result;
                        result = WaitEvent(completed);
                        if (result != HCCL_SUCCESS) return result;
                    }
                }
                descriptor += stride;
                cursor += one;
            }
        }
        return HCCL_SUCCESS;
    }
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto &task = static_cast<const Cam::AsyncCcu::SegmentTaskArg &>(arg);
        return {task.descriptorAddress, task.source.base, task.source.token,
            task.destination.base, task.destination.token, task.planStatusAddress};
    }
private:
    const uint32_t peer_;
    const uint32_t count_;
};

struct PeerKernel {
    PeerKernel(ChannelHandle channel, uint32_t peer, uint32_t experts)
        : arg(channel, peer, experts), creator([](const hcomm::CcuKernelArg &value) {
            return std::make_unique<BatchKernel>(value);
        }) {}
    BatchKernelArg arg;
    hcomm::KernelCreator creator;
    CcuKernelHandle handle{0};
};

HcclResult Lookup(const char *group, HcclComm &comm)
{
    return group == nullptr ? HCCL_E_PARA : HcomGetCommHandleByGroup(group, &comm);
}
} // namespace

struct AfdA5Transport {
    HcclComm comm{nullptr};
    uint32_t rank{0};
    uint32_t attention{0};
    uint32_t experts{0};
    AfdA5Span source{};
    std::vector<AfdA5Span> windows;
    std::vector<HcclMemHandle> memory;
    std::vector<std::unique_ptr<PeerKernel>> kernels;
};

extern "C" int afd_a5_query_group(const char *group, AfdA5GroupInfo *info)
{
    if (info == nullptr) return HCCL_E_PARA;
    HcclComm comm = nullptr;
    HcclResult status = Lookup(group, comm);
    if (status != HCCL_SUCCESS) return status;
    status = HcclGetRankId(comm, &info->rank);
    if (status != HCCL_SUCCESS) return status;
    status = HcclGetRankSize(comm, &info->world_size);
    if (status != HCCL_SUCCESS) return status;
    void *buffer = nullptr;
    return HcclGetHcclBuffer(comm, &buffer, &info->window_bytes);
}

extern "C" int afd_a5_register_window(const char *group, uint64_t base, uint64_t bytes,
    void **handle, uint64_t *token)
{
    if (base == 0 || bytes == 0 || token == nullptr || handle == nullptr) return HCCL_E_PARA;
    try {
        HcclComm comm = nullptr;
        auto status = Lookup(group, comm);
        if (status != HCCL_SUCCESS) return status;
        const CommMem window{COMM_MEM_TYPE_DEVICE, reinterpret_cast<void *>(base), bytes};
        status = HcclCommMemReg(comm, "afd_a5_full_window_v1", &window, handle);
        if (status != HCCL_SUCCESS) return status;
        *token = CcuRep::GetTokenInfo(base, bytes);
        return *token == 0 ? HCCL_E_PARA : HCCL_SUCCESS;
    } catch (...) { return HCCL_E_INTERNAL; }
}

extern "C" int afd_a5_transport_create(const char *group, uint32_t attention_ranks,
    uint32_t experts_per_rank, void *window_handle, const AfdA5Span *windows, uint32_t window_count,
    AfdA5Span source, AfdA5Transport **transport)
{
    if (transport == nullptr || windows == nullptr || window_handle == nullptr || attention_ranks == 0
        || attention_ranks >= window_count || window_count > MAX_FULL_RANKS || experts_per_rank == 0)
        return HCCL_E_PARA;
    *transport = nullptr;
    try {
        // Transfer ownership before registering anything. Even a failed setup
        // leaves creators alive until the caller tears down its failed domain.
        *transport = new AfdA5Transport();
        AfdA5Transport *owned = *transport;
        auto status = Lookup(group, owned->comm);
        if (status != HCCL_SUCCESS) return status;
        status = HcclGetRankId(owned->comm, &owned->rank);
        if (status != HCCL_SUCCESS) return status;
        owned->attention = attention_ranks;
        owned->experts = experts_per_rank;
        owned->source = source;
        owned->windows.assign(windows, windows + window_count);
        HcclMemHandle handle = nullptr;
        owned->memory.push_back(window_handle);
        if (owned->rank >= attention_ranks) {
            const CommMem input{COMM_MEM_TYPE_DEVICE, reinterpret_cast<void *>(source.base), source.bytes};
            status = HcclCommMemReg(owned->comm, "afd_a5_full_source_v1", &input, &handle);
            if (status != HCCL_SUCCESS) return status;
            owned->memory.push_back(handle);
            owned->source.token = CcuRep::GetTokenInfo(source.base, source.bytes);
            if (owned->source.token == 0) return HCCL_E_PARA;
        }
        const uint32_t begin = owned->rank < attention_ranks ? attention_ranks : 0;
        const uint32_t end = owned->rank < attention_ranks ? window_count : attention_ranks;
        std::vector<HcclChannelDesc> descriptions(end - begin);
        std::vector<ChannelHandle> channels(end - begin);
        status = HcclChannelDescInit(descriptions.data(), descriptions.size());
        if (status != HCCL_SUCCESS) return status;
        for (uint32_t peer = begin; peer < end; ++peer) {
            CommLink *links = nullptr;
            uint32_t count = 0;
            status = HcclRankGraphGetLinks(owned->comm, TOPOLOGY_LAYER, owned->rank, peer, &links, &count);
            if (status != HCCL_SUCCESS) return status;
            const CommLink *selected = nullptr;
            for (uint32_t i = 0; i < count; ++i) {
                const auto protocol = links[i].linkAttr.linkProtocol;
                if (protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UBC_CTP) {
                    selected = &links[i];
                    break;
                }
            }
            if (selected == nullptr) return HCCL_E_NOT_SUPPORT;
            auto &description = descriptions[peer - begin];
            description.remoteRank = peer;
            description.channelProtocol = selected->linkAttr.linkProtocol;
            description.localEndpoint = selected->srcEndpointDesc;
            description.remoteEndpoint = selected->dstEndpointDesc;
            description.memHandles = owned->memory.data();
            description.memHandleNum = owned->memory.size();
        }
        status = HcclChannelAcquire(owned->comm, COMM_ENGINE_CCU,
            descriptions.data(), descriptions.size(), channels.data());
        if (status != HCCL_SUCCESS) return status;
        for (uint32_t peer = begin; peer < end; ++peer) {
            owned->kernels.push_back(std::make_unique<PeerKernel>(channels[peer - begin], peer, experts_per_rank));
            auto &kernel = owned->kernels.back();
            status = HcclCcuKernelRegister(owned->comm, &kernel->handle, &kernel->creator, &kernel->arg);
            if (status != HCCL_SUCCESS) return status;
        }
        status = HcclCcuKernelRegisterFinish(owned->comm);
        if (status != HCCL_SUCCESS) return status;
        return HCCL_SUCCESS;
    } catch (...) { return HCCL_E_INTERNAL; }
}

extern "C" int afd_a5_transport_launch(AfdA5Transport *transport, void *stream,
    uint64_t descriptors, uint64_t status_address)
{
    if (transport == nullptr || transport->rank < transport->attention || stream == nullptr)
        return HCCL_E_PARA;
    try {
        ThreadHandle thread = 0;
        auto status = HcclThreadAcquireWithStream(transport->comm, COMM_ENGINE_CPU_TS,
            stream, THREAD_NOTIFICATIONS, &thread);
        if (status != HCCL_SUCCESS) return status;
        const Cam::AsyncCcu::RegisteredSpan source{
            transport->source.base, transport->source.bytes, transport->source.token};
        for (const auto &kernel : transport->kernels) {
            const uint32_t peer = kernel->arg.peer;
            const auto span = transport->windows[peer];
            Cam::AsyncCcu::SegmentTaskArg task(descriptors
                    + peer * transport->experts * sizeof(Cam::AsyncCcu::SendSegment),
                source, {span.base, span.bytes, span.token}, status_address);
            status = HcclCcuKernelLaunch(transport->comm, thread, kernel->handle, &task);
            if (status != HCCL_SUCCESS) return status;
        }
        return HCCL_SUCCESS;
    } catch (...) { return HCCL_E_INTERNAL; }
}

extern "C" void afd_a5_transport_destroy(AfdA5Transport *transport) { delete transport; }
