// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_SIMT_DESCRIPTOR_BUDGET_H
#define AFD_CAM_ASYNC_SIMT_DESCRIPTOR_BUDGET_H

#include <cstdint>

#ifndef CAM_ASYNC_PIPELINE_BUDGET_INLINE
#define CAM_ASYNC_PIPELINE_BUDGET_INLINE constexpr
#define CAM_ASYNC_PIPELINE_DEFAULT_BUDGET_INLINE
#endif

namespace Cam::AsyncPipeline {
constexpr uint64_t METADATA_BUFFER_BYTES = 64 * 1024;
constexpr uint64_t DESCRIPTOR_TILE_BYTES = 8 * 1024;
constexpr uint64_t STATUS_BUFFER_BYTES = 32;
constexpr uint64_t SIMT_MAX_DATA_CACHE_BYTES = 128 * 1024;
constexpr uint64_t SIMT_COMPILER_RESERVED_BYTES = 8 * 1024;
// This upper-bound policy does not describe a measured/default cache setting.
constexpr uint64_t SIMT_RESERVED_UB_BYTES =
    SIMT_MAX_DATA_CACHE_BYTES + SIMT_COMPILER_RESERVED_BYTES;
constexpr uint64_t SIMT_PRODUCER_ARENA_BYTES =
    METADATA_BUFFER_BYTES + DESCRIPTOR_TILE_BYTES + STATUS_BUFFER_BYTES;
constexpr uint64_t SIMT_PRODUCER_MIN_PHYSICAL_UB_BYTES =
    SIMT_RESERVED_UB_BYTES + SIMT_PRODUCER_ARENA_BYTES;
// Even the no-VF failure path belongs to a compiled kernel containing VF code.
// The host must enforce this smaller launch gate before launching that path.
constexpr uint64_t SIMT_PRODUCER_MIN_STATUS_UB_BYTES =
    SIMT_RESERVED_UB_BYTES + STATUS_BUFFER_BYTES;
// P2's planner statuses occupy 0..10. This producer-only failure is nonzero,
// so the unchanged P2 CCU status guard cannot issue any payload transfer.
enum class ProducerStatus : uint64_t {
    ResourceUnavailable = 11,
};

CAM_ASYNC_PIPELINE_BUDGET_INLINE bool CanLaunchDescriptorProducer(uint64_t physicalUbBytes)
{
    return SIMT_RESERVED_UB_BYTES <= physicalUbBytes
        && STATUS_BUFFER_BYTES <= physicalUbBytes - SIMT_RESERVED_UB_BYTES;
}

CAM_ASYNC_PIPELINE_BUDGET_INLINE bool CanUseDescriptorSimt(uint64_t physicalUbBytes)
{
    return SIMT_RESERVED_UB_BYTES <= physicalUbBytes
        && SIMT_PRODUCER_ARENA_BYTES <= physicalUbBytes - SIMT_RESERVED_UB_BYTES;
}
} // namespace Cam::AsyncPipeline

#ifdef CAM_ASYNC_PIPELINE_DEFAULT_BUDGET_INLINE
#undef CAM_ASYNC_PIPELINE_DEFAULT_BUDGET_INLINE
#undef CAM_ASYNC_PIPELINE_BUDGET_INLINE
#endif
#endif // AFD_CAM_ASYNC_SIMT_DESCRIPTOR_BUDGET_H
