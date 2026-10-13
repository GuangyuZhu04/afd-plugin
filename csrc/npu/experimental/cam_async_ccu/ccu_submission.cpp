// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "ccu_submission.h"

namespace Cam::AsyncCcu {
SegmentSubmission::SegmentSubmission(HcclComm comm, ThreadHandle thread,
    ChannelHandle channel, uint64_t destinationRank)
    : comm_(comm), thread_(thread), kernelArg_(channel, destinationRank),
      creator_(MakeSegmentKernelCreator())
{
}

HcclResult SegmentSubmission::Register()
{
    if (comm_ == nullptr) {
        return HCCL_E_PARA;
    }
    if (registered_) {
        return HCCL_SUCCESS;
    }
    const HcclResult result = HcclCcuKernelRegister(comm_, &kernelHandle_, &creator_, &kernelArg_);
    registered_ = result == HCCL_SUCCESS;
    return result;
}

HcclResult SegmentSubmission::FinishRegistration()
{
    return registered_ ? HcclCcuKernelRegisterFinish(comm_) : HCCL_E_PARA;
}

HcclResult SegmentSubmission::Launch(SegmentTaskArg &task)
{
    if (!registered_) {
        return HCCL_E_PARA;
    }
    // The caller proves producer completion and the selected thread's ordering
    // before this task can consume GM. No device metadata is read on the host.
    return HcclCcuKernelLaunch(comm_, thread_, kernelHandle_, &task);
}
} // namespace Cam::AsyncCcu
