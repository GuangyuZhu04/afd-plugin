// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_CHECKED_ARITHMETIC_H
#define AFD_CAM_ASYNC_CHECKED_ARITHMETIC_H
#include <cstdint>
#ifndef CAM_ASYNC_CHECKED_INLINE
#define CAM_ASYNC_CHECKED_INLINE inline
#define CAM_ASYNC_CHECKED_DEFAULT_INLINE
#endif

namespace Cam::AsyncArithmetic {
constexpr uint32_t LIMB_BITS = 32;
constexpr uint64_t LIMB_MASK = 0xFFFFFFFFULL;
constexpr uint64_t MAX_ARITHMETIC_VALUE = ~uint64_t{0};

// CANN 9.0/3101 can lower the usual MAX / a overflow idiom to a 128-bit
// __multi3 call which is unavailable when linking a real AIV kernel. Explicit
// 32-bit limbs retain the entire uint64 contract without that runtime helper.
CAM_ASYNC_CHECKED_INLINE bool CheckedMultiply64(uint64_t left, uint64_t right, uint64_t &result)
{
    const uint64_t leftLow = left & LIMB_MASK;
    const uint64_t rightLow = right & LIMB_MASK;
    const uint64_t leftHigh = left >> LIMB_BITS;
    const uint64_t rightHigh = right >> LIMB_BITS;
    if (leftHigh != 0 && rightHigh != 0) return false;
    const uint64_t low = leftLow * rightLow;
    // At most one cross product is nonzero after the preceding guard.
    const uint64_t cross = leftHigh * rightLow + leftLow * rightHigh;
    if (cross > LIMB_MASK) return false;
    const uint64_t high = cross << LIMB_BITS;
    if (high > MAX_ARITHMETIC_VALUE - low) return false;
    result = high + low;
    return true;
}
} // namespace Cam::AsyncArithmetic
#ifdef CAM_ASYNC_CHECKED_DEFAULT_INLINE
#undef CAM_ASYNC_CHECKED_DEFAULT_INLINE
#undef CAM_ASYNC_CHECKED_INLINE
#endif
#endif // AFD_CAM_ASYNC_CHECKED_ARITHMETIC_H
