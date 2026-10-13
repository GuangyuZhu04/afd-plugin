// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "ccu_segment_kernel.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace Cam::AsyncCcu {
namespace CcuRep = hcomm::CcuRep;

namespace {
constexpr uint32_t SEGMENT_CHANNEL_COUNT = 1;
constexpr char KERNEL_SIGNATURE[] = "afd_async_ccu_single_segment_v1";

void ValidateRegisteredSpan(const RegisteredSpan &span)
{
    if (span.base == 0 || span.bytes == 0 || span.token == 0
        || span.bytes > std::numeric_limits<uint64_t>::max() - span.base) {
        throw std::invalid_argument("CCU requires an explicit nonempty registered span and token");
    }
}
} // namespace

SegmentKernelArg::SegmentKernelArg(ChannelHandle channel, uint64_t rank)
    : destinationRank(rank)
{
    if (channel == 0) {
        throw std::invalid_argument("CCU segment kernel requires a registered channel");
    }
    channels.push_back(channel);
}

hcomm::CcuKernelSignature SegmentKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    signature.Append(KERNEL_SIGNATURE);
    signature.Append('|');
    signature.Append(destinationRank);
    signature.Append('|');
    signature.Append(channels.at(0));
    return signature;
}

SegmentTaskArg::SegmentTaskArg(uint64_t descriptor, RegisteredSpan src,
    RegisteredSpan dst, uint64_t statusAddress)
    : descriptorAddress(descriptor), source(src), destination(dst),
      planStatusAddress(statusAddress)
{
    if (descriptorAddress == 0 || descriptorAddress % alignof(SendSegment) != 0
        || sizeof(SendSegment) > std::numeric_limits<uint64_t>::max() - descriptorAddress) {
        throw std::invalid_argument("CCU segment descriptor must reference aligned device GM");
    }
    ValidateRegisteredSpan(source);
    ValidateRegisteredSpan(destination);
    if (planStatusAddress == 0 || planStatusAddress % alignof(uint64_t) != 0
        || sizeof(uint64_t) > std::numeric_limits<uint64_t>::max() - planStatusAddress) {
        throw std::invalid_argument("CCU plan status must reference aligned 8-byte device GM");
    }
}

SegmentKernel::SegmentKernel(const hcomm::CcuKernelArg &arg)
    : hcomm::CcuKernel(arg),
      expectedDestinationRank_(static_cast<const SegmentKernelArg &>(arg).destinationRank)
{
    if (arg.channels.size() != SEGMENT_CHANNEL_COUNT || arg.channels[0] == 0) {
        throw std::invalid_argument("CCU segment kernel requires exactly one registered channel");
    }
}

HcclResult SegmentKernel::Algorithm()
{
    // Load consumes the six host-known arguments in GeneArgs order. Dynamic
    // offsets and byteCount stay in device GM and are read by CCU instructions.
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

    auto sourceOffset = CreateVariable();
    auto destinationOffset = CreateVariable();
    auto byteCount = CreateVariable();
    auto destinationRank = CreateVariable();
    auto fieldOffset = CreateVariable();
    auto fieldAddress = CreateVariable();
    // The installed CCU representation has no general dynamic span comparison, so do
    // not mistake address arithmetic or token presence for bounds validation.
    // An invalid producer plan must not expose a stale descriptor from an older
    // invocation. Status is a separate uint64 GM word published after planning.
    CCU_IF(planStatus == static_cast<uint64_t>(Status::Success)) {
        LoadVariable(descriptor, sourceOffset);
        fieldOffset = offsetof(SendSegment, dstByteOffset);
        fieldAddress = descriptor + fieldOffset;
        LoadVariable(fieldAddress, destinationOffset);
        fieldOffset = offsetof(SendSegment, byteCount);
        fieldAddress = descriptor + fieldOffset;
        LoadVariable(fieldAddress, byteCount);
        fieldOffset = offsetof(SendSegment, dstRank);
        fieldAddress = descriptor + fieldOffset;
        LoadVariable(fieldAddress, destinationRank);
        // The expected rank is an immutable registration parameter, not a new
        // task argument. A wrong table-slot selection must not write its payload
        // through another peer's otherwise valid fixed channel.
        CCU_IF(destinationRank == expectedDestinationRank_) {
            CCU_IF(byteCount != 0) {
                local.addr = sourceBase;
                local.addr += sourceOffset;
                remote.addr = destinationBase;
                remote.addr += destinationOffset;
                auto completed = CreateCompletedEvent();
                HcclResult result = WriteNb(channels_[0], remote, local, byteCount, completed);
                if (result != HCCL_SUCCESS) {
                    return result;
                }
                result = WaitEvent(completed);
                if (result != HCCL_SUCCESS) {
                    return result;
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

std::vector<uint64_t> SegmentKernel::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    // The matching launch wrapper supplies this concrete task type. It contains
    // known addresses/tokens only; this method does not copy or poll device data.
    const auto &task = static_cast<const SegmentTaskArg &>(arg);
    return {task.descriptorAddress, task.source.base, task.source.token,
        task.destination.base, task.destination.token, task.planStatusAddress};
}

hcomm::KernelCreator MakeSegmentKernelCreator()
{
    return [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<SegmentKernel>(arg);
    };
}

} // namespace Cam::AsyncCcu
