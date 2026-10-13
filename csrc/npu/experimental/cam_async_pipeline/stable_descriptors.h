// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_PIPELINE_STABLE_DESCRIPTORS_H
#define AFD_CAM_ASYNC_PIPELINE_STABLE_DESCRIPTORS_H

#include <cstdint>

// The A5 wrapper supplies __simt_callee__ inline and a UB-aware backend. The
// CPU probe executes these same loops with ordinary pointers. Descriptor writes
// retain the four uint64 fields of the P2 SendSegment ABI through the backend.
#ifndef CAM_ASYNC_PIPELINE_INLINE
#define CAM_ASYNC_PIPELINE_INLINE inline
#define CAM_ASYNC_PIPELINE_DEFAULT_INLINE
#endif

namespace Cam::AsyncPipeline {
constexpr uint64_t METADATA_HEADER_FIELDS = 5;

// Preconditions for both phases: a complete P2 CombineSegmentCursor preflight
// succeeded on the retained metadata; shapes, counts, target group and every
// source/destination byte span were checked. No external writer may change the
// metadata. Prefixing destroys the raw counts, so the cursor must not be used
// afterward. The original five header fields and TP MoE prefixes are retained.
// Backend provides ReadInfo(pointer, index), WriteInfo(pointer, index, value)
// and WriteDescriptor(pointer, localSlot, srcBytes, dstBytes, bytes, dstRank).

// One lane owns each complete TP row. Inclusive prefixes replace counts in
// place, avoiding another metadata-sized UB table. All rows must finish and
// become visible to the second VF before descriptor reads begin.
template <typename Backend, typename MetadataPointer>
CAM_ASYNC_PIPELINE_INLINE void PrefixTpRowsForLane(MetadataPointer metadata,
    uint64_t tpSize, uint64_t expertsPerRank, uint32_t lane, uint32_t laneCount)
{
    if (laneCount == 0 || lane >= laneCount) {
        return;
    }
    const uint64_t countsBase = METADATA_HEADER_FIELDS + tpSize;
    for (uint64_t tp = lane; tp < tpSize; tp += laneCount) {
        uint64_t prefix = 0;
        for (uint64_t expert = 0; expert < expertsPerRank; ++expert) {
            const uint64_t index = countsBase + tp * expertsPerRank + expert;
            prefix += Backend::ReadInfo(metadata, index);
            Backend::WriteInfo(metadata, index, prefix);
        }
    }
}

template <typename Backend, typename MetadataPointer>
CAM_ASYNC_PIPELINE_INLINE uint64_t PrefixBeforeExpert(MetadataPointer metadata,
    uint64_t tpSize, uint64_t expertsPerRank, uint64_t tp, uint64_t expert)
{
    if (expert == 0) {
        return 0;
    }
    return Backend::ReadInfo(metadata,
        METADATA_HEADER_FIELDS + tpSize + tp * expertsPerRank + expert - 1);
}

// One lane owns each fixed TP-major slot in this bounded output tile. The
// caller proves slotBegin + slotCount <= tpSize * expertsPerRank and output
// capacity >= slotCount, and uses the same preflight-checked rowBytes,
// payloadBase and tpGroupRank. Every output slot is written exactly once,
// including all-zero empty and out-of-chunk slots. No atomics or ticket order
// influence the expert-major source ordering or peer-specific destination.
// Recovering the source prefix costs O(TP) reads per slot; this is a deliberate
// tradeoff for bounded UB usage and independent slot ownership.
template <typename Backend, typename MetadataPointer, typename DescriptorPointer>
CAM_ASYNC_PIPELINE_INLINE void DescribeSlotsForLane(MetadataPointer metadata,
    DescriptorPointer output, uint64_t tpSize, uint64_t expertsPerRank,
    uint64_t startExpert, uint64_t endExpert, uint64_t rowBytes,
    uint64_t payloadBase, uint64_t tpGroupRank, uint64_t slotBegin,
    uint64_t slotCount, uint32_t lane, uint32_t laneCount)
{
    if (laneCount == 0 || lane >= laneCount) {
        return;
    }
    for (uint64_t localSlot = lane; localSlot < slotCount; localSlot += laneCount) {
        const uint64_t slot = slotBegin + localSlot;
        const uint64_t tp = slot / expertsPerRank;
        const uint64_t expert = slot % expertsPerRank;
        if (expert < startExpert || expert > endExpert) {
            Backend::WriteDescriptor(output, localSlot, 0, 0, 0, 0);
            continue;
        }
        const uint64_t previous = PrefixBeforeExpert<Backend>(
            metadata, tpSize, expertsPerRank, tp, expert);
        const uint64_t inclusive = Backend::ReadInfo(metadata,
            METADATA_HEADER_FIELDS + tpSize + slot);
        const uint64_t count = inclusive - previous;
        if (count == 0) {
            Backend::WriteDescriptor(output, localSlot, 0, 0, 0, 0);
            continue;
        }
        uint64_t sourceRow = 0;
        for (uint64_t peer = 0; peer < tpSize; ++peer) {
            const uint64_t peerPrevious = PrefixBeforeExpert<Backend>(
                metadata, tpSize, expertsPerRank, peer, expert);
            sourceRow += peerPrevious - PrefixBeforeExpert<Backend>(
                metadata, tpSize, expertsPerRank, peer, startExpert);
            if (peer < tp) {
                const uint64_t peerInclusive = Backend::ReadInfo(metadata,
                    METADATA_HEADER_FIELDS + tpSize + peer * expertsPerRank + expert);
                sourceRow += peerInclusive - peerPrevious;
            }
        }
        const uint64_t destinationRow = Backend::ReadInfo(
            metadata, METADATA_HEADER_FIELDS + tp) + previous;
        Backend::WriteDescriptor(output, localSlot, sourceRow * rowBytes,
            payloadBase + destinationRow * rowBytes, count * rowBytes,
            tpGroupRank + tp);
    }
}
} // namespace Cam::AsyncPipeline

#ifdef CAM_ASYNC_PIPELINE_DEFAULT_INLINE
#undef CAM_ASYNC_PIPELINE_DEFAULT_INLINE
#undef CAM_ASYNC_PIPELINE_INLINE
#endif
#endif // AFD_CAM_ASYNC_PIPELINE_STABLE_DESCRIPTORS_H
