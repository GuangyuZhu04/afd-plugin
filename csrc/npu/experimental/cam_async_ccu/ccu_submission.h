// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_EXPERIMENTAL_CCU_SUBMISSION_H
#define AFD_EXPERIMENTAL_CCU_SUBMISSION_H

#include "hccl_ccu_res.h"
#include "ccu_segment_kernel.h"

namespace Cam::AsyncCcu {
// Explicit experimental host submission. The caller owns the real communication
// domain, channel, thread, registered spans, descriptor/status GM and task args.
// This object must stay alive while the domain may reference its creator/args.
// Launch enqueues a CCU task; success is not a remote-visible completion proof.
class SegmentSubmission final {
public:
    SegmentSubmission(HcclComm comm, ThreadHandle thread, ChannelHandle channel, uint64_t destinationRank);
    SegmentSubmission(const SegmentSubmission &) = delete;
    SegmentSubmission &operator=(const SegmentSubmission &) = delete;
    SegmentSubmission(SegmentSubmission &&) = delete;
    SegmentSubmission &operator=(SegmentSubmission &&) = delete;

    HcclResult Register();
    // Call only at the setup boundary after all required peer kernels have been
    // registered. It must never become a per-chunk collective or host polling.
    HcclResult FinishRegistration();
    // Requires a successful domain-wide FinishRegistration setup boundary and
    // a validated thread/producer dependency, both owned by the caller.
    HcclResult Launch(SegmentTaskArg &task);

private:
    HcclComm comm_;
    ThreadHandle thread_;
    SegmentKernelArg kernelArg_;
    hcomm::KernelCreator creator_;
    CcuKernelHandle kernelHandle_{0};
    bool registered_{false};
};
} // namespace Cam::AsyncCcu
#endif // AFD_EXPERIMENTAL_CCU_SUBMISSION_H
