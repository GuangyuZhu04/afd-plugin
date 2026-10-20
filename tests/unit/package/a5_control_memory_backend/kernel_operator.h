// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// CPU memory adapter for the production control state machine. This is NOT a
// CANN implementation or evidence of DMA/event/remote-visibility correctness.
#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define GM_ADDR uint8_t *

namespace AscendC {
constexpr int HCCL_GROUP_ID_0 = 0;
enum class HardEvent { MTE2_S, S_MTE2, S_MTE3, MTE3_S };
struct HcclCombineOpParam { uint64_t windowsIn[64]{}; };
struct MemoryHooks {
    std::function<void(uint64_t, uint64_t)> beforeRead;
    std::function<void(uint64_t, uint64_t)> afterWrite;
};
struct Storage { std::vector<uint8_t> bytes; MemoryHooks *hooks{nullptr}; };

template <typename T>
class LocalTensor {
public:
    LocalTensor() = default;
    explicit LocalTensor(std::shared_ptr<Storage> storage) : storage_(std::move(storage)) {}
    T GetValue(uint64_t index) const
    {
        T value;
        std::memcpy(&value, storage_->bytes.data() + index * sizeof(T), sizeof(T));
        return value;
    }
    void SetValue(uint64_t index, T value)
    {
        if ((index + 1) * sizeof(T) > storage_->bytes.size()) throw std::runtime_error("UB adapter overflow");
        std::memcpy(storage_->bytes.data() + index * sizeof(T), &value, sizeof(T));
    }
    template <typename U> LocalTensor<U> ReinterpretCast() { return LocalTensor<U>(storage_); }
    std::shared_ptr<Storage> storage_;
};
template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(T *data) { data_ = data; }
    T *data_{nullptr};
};
template <typename... Unused>
class TBuf {
public:
    template <typename T> LocalTensor<T> Get() { return LocalTensor<T>(storage_); }
    std::shared_ptr<Storage> storage_;
};
class TPipe {
public:
    explicit TPipe(MemoryHooks *hooks = nullptr) : hooks_(hooks) {}
    template <typename... Args> void InitBuffer(TBuf<Args...> &buffer, uint64_t bytes)
    {
        buffer.storage_ = std::make_shared<Storage>();
        buffer.storage_->bytes.resize(bytes);
        buffer.storage_->hooks = hooks_;
    }
    template <HardEvent Event> int FetchEventID() { return 0; }
    void Reset() {}
private:
    MemoryHooks *hooks_;
};
struct CopyParams { uint32_t blocks, bytes, sourceStride, destinationStride, reserved; };
struct Padding { bool enabled; uint32_t left, right; uint64_t value; };
template <typename T>
void DataCopyPad(LocalTensor<T> &destination, GlobalTensor<T> &source, CopyParams copy, Padding)
{
    if (copy.blocks != 1 || copy.bytes > destination.storage_->bytes.size())
        throw std::runtime_error("Unsupported control read geometry");
    if (destination.storage_->hooks && destination.storage_->hooks->beforeRead)
        destination.storage_->hooks->beforeRead(reinterpret_cast<uint64_t>(source.data_), copy.bytes);
    std::memcpy(destination.storage_->bytes.data(), source.data_, copy.bytes);
}
template <typename T>
void DataCopyPad(GlobalTensor<T> &destination, LocalTensor<T> &source, CopyParams copy)
{
    if (copy.blocks != 1 || copy.bytes > source.storage_->bytes.size())
        throw std::runtime_error("Unsupported control write geometry");
    std::memcpy(destination.data_, source.storage_->bytes.data(), copy.bytes);
    if (source.storage_->hooks && source.storage_->hooks->afterWrite)
        source.storage_->hooks->afterWrite(reinterpret_cast<uint64_t>(destination.data_), copy.bytes);
}
// Ordering is intentionally not simulated here. The test exercises selectors,
// leases, records and credit decisions; real SDK builds validate the API calls.
template <HardEvent Event> void SetFlag(int) {}
template <HardEvent Event> void WaitFlag(int) {}
template <bool> void SyncAll() {}
inline uint32_t GetBlockIdx() { return 0; }
template <int> uint8_t *GetHcclContext() { return nullptr; }
inline void Trap() { throw std::runtime_error("device trap"); }
} // namespace AscendC
