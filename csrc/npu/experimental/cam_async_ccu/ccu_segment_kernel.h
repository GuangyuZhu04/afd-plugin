// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_EXPERIMENTAL_CCU_SEGMENT_KERNEL_H
#define AFD_EXPERIMENTAL_CCU_SEGMENT_KERNEL_H

#include <cstdint>
#include <memory>
#include <vector>

#include "ccu_kernel.h"

#include "combine_segments.h"

namespace Cam::AsyncCcu {

// The caller obtains these values from actual memory registration/token
// exchange. A nonzero token is a precondition, not proof of registration.
struct RegisteredSpan {
    uint64_t base;
    uint64_t bytes;
    uint64_t token;
};

class SegmentKernelArg final : public hcomm::CcuKernelArg {
public:
    SegmentKernelArg(ChannelHandle channel, uint64_t destinationRank);

    hcomm::CcuKernelSignature GetKernelSignature() const override;

    const uint64_t destinationRank;
};

class SegmentTaskArg final : public hcomm::CcuTaskArg {
public:
    // descriptorAddress points to one CCU-readable SendSegment in device GM.
    // The producer must validate both offset/count bounds and dstRank against
    // this fixed channel before publishing the descriptor. This class never
    // reads that device descriptor on the host or invents its memory tokens.
    SegmentTaskArg(uint64_t descriptorAddress, RegisteredSpan source,
        RegisteredSpan destination, uint64_t planStatusAddress);

    const uint64_t descriptorAddress;
    const RegisteredSpan source;
    const RegisteredSpan destination;
    const uint64_t planStatusAddress;
};

// Experimental payload only: one fixed peer, one dynamic contiguous segment.
// WaitEvent is emitted, but its remote visibility/completion scope still needs
// device qualification. No compact-v2 ready flag or completion notification is
// published, and this candidate is not selected by the production operators.
class SegmentKernel final : public hcomm::CcuKernel {
public:
    explicit SegmentKernel(const hcomm::CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    const uint64_t expectedDestinationRank_;
};

hcomm::KernelCreator MakeSegmentKernelCreator();

} // namespace Cam::AsyncCcu

#endif // AFD_EXPERIMENTAL_CCU_SEGMENT_KERNEL_H
