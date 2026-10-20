// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "acl/acl.h"

#include "torch_binding_cam_async.h"
#include "../runtime/cam_async_a5_transport.h"
#include "utils/op_kernel/cam_async_a5_contract.h"
#include "utils/op_kernel/cam_async_combine_segments.h"

namespace afd_plugin::cam_async {
namespace {
namespace Full = Cam::AsyncA5;
constexpr int64_t CONFIG_FIELDS = 10;
constexpr int64_t DEFAULT_RECEIVE_ROWS = 262144;
constexpr int64_t EXCHANGE_WORDS = 6;
constexpr uint64_t MIB_BYTES = 1024 * 1024;
constexpr uint64_t MAX_WINDOW_MIB = 32768;
constexpr int64_t SOURCE_CAPACITY_WORD = offsetof(Full::RuntimeArgs, sourceCapacityBytes) / sizeof(uint64_t);
constexpr int64_t MAX_METADATA_BYTES = 65536;
constexpr int64_t PLAN_AND_TRANSPORT_STATUS_WORDS = 2;

void CheckAcl(aclError result, const char *operation)
{
    TORCH_CHECK(result == ACL_SUCCESS, operation, " failed: ", result);
}
void CheckHccl(int result, const char *operation)
{
    TORCH_CHECK(result == 0, operation, " failed: ", result);
}
uint64_t Multiply(uint64_t a, uint64_t b)
{
    uint64_t result = 0;
    TORCH_CHECK(Cam::AsyncCcu::CheckedMultiply(a, b, result), "A5 capacity overflow");
    return result;
}

struct RetainedCall {
    std::vector<at::Tensor> tensors;
    c10_npu::NPUEvent completion;
};
struct DispatchReservation {
    at::Tensor ticket;
    bool pending{false};
};
} // namespace

// Owns all tensors and C ABI state. A caller must quiesce, destroy its dedicated
// HCCL process group, then release. No mutable global session registry is used.
class AsyncCamA5Context final : public torch::CustomClassHolder {
public:
    AsyncCamA5Context(at::Tensor anchor, std::string group, std::vector<int64_t> config)
        : anchor_(std::move(anchor)), group_(std::move(group))
    {
        TORCH_CHECK(config.size() == CONFIG_FIELDS, "A5 config is [rank,A,M,R,TP,max_seq,H,K,slots,quant]");
        TORCH_CHECK(anchor_.device().type() == c10::DeviceType::PrivateUse1,
            "A5 context requires an NPU anchor");
        TORCH_CHECK(anchor_.scalar_type() == at::kHalf || anchor_.scalar_type() == at::kBFloat16,
            "A5 anchor must be FP16 or BF16");
        TORCH_CHECK(!anchor_.requires_grad(), "A5 sessions support inference only");
        for (size_t i = 1; i < config.size() - 1; ++i)
            TORCH_CHECK(config[i] > 0, "A5 config dimensions must be positive");
        TORCH_CHECK(config[0] >= 0 && (config[9] == 0 || config[9] == 1), "Invalid A5 rank/quant");
        prototype_.localRank = config[0];
        prototype_.attentionRanks = config[1];
        prototype_.moeRanks = config[2];
        prototype_.expertsPerRank = config[3];
        prototype_.tpSize = config[4];
        max_sequence_ = config[5];
        prototype_.hiddenSize = config[6];
        prototype_.topk = config[7];
        prototype_.slotCount = config[8];
        quant_ = config[9];
        prototype_.domain = 1;
        TORCH_CHECK(Full::ValidateArgs(prototype_), "Invalid A5 topology or slot count");
        TORCH_CHECK(prototype_.hiddenSize % 32 == 0 && max_sequence_ <= DEFAULT_RECEIVE_ROWS
            && max_sequence_ / prototype_.tpSize <= UINT16_MAX
            && prototype_.topk <= Multiply(prototype_.moeRanks, prototype_.expertsPerRank),
            "A5 shape exceeds compact frame bounds");
        const uint64_t metadata_words = 5 + prototype_.tpSize
            + Multiply(prototype_.expertsPerRank, prototype_.tpSize);
        TORCH_CHECK(metadata_words <= MAX_METADATA_BYTES / sizeof(int64_t),
            "A5 descriptor metadata exceeds 64 KiB");
        // Match the existing allocator's float parsing and truncation.
        float factor = 1.0F;
        if (const char *value = std::getenv("BATCH_SIZE_FACTOR")) {
            try { factor = std::stof(value); } catch (const std::exception &) { factor = 1.0F; }
        }
        TORCH_CHECK(factor > 0 && factor <= 1.0F, "BATCH_SIZE_FACTOR must be in (0,1]");
        prototype_.receiveCapacity = static_cast<int64_t>(DEFAULT_RECEIVE_ROWS * factor);
        TORCH_CHECK(prototype_.receiveCapacity > 0, "A5 receive capacity is zero");
        abort_ = at::zeros({1}, anchor_.options().dtype(at::kLong));
        prototype_.abortAddress = reinterpret_cast<uint64_t>(abort_.data_ptr());
        reservations_.resize(prototype_.slotCount);
    }

    std::vector<int64_t> prepare()
    {
        TORCH_CHECK(!prepared_ && !released_, "A5 context preparation is one-shot");
        stream_ = c10_npu::getCurrentNPUStream();
        AfdA5GroupInfo info{};
        CheckHccl(afd_a5_query_group(group_.c_str(), &info), "HCCL group lookup");
        TORCH_CHECK(info.rank == prototype_.localRank
            && info.world_size == prototype_.attentionRanks + prototype_.moeRanks, "A5 group topology mismatch");
        const char *window_env = std::getenv("HCCL_BUFFSIZE");
        TORCH_CHECK(window_env != nullptr, "A5 full profile requires explicit HCCL_BUFFSIZE matching the group");
        size_t consumed = 0;
        const auto window_mib = std::stoull(window_env, &consumed);
        TORCH_CHECK(consumed == std::strlen(window_env) && window_mib > 0 && window_mib <= MAX_WINDOW_MIB
            && info.window_bytes == window_mib * MIB_BYTES, "HCCL_BUFFSIZE differs from allocated A5 window");
        prototype_.windowBytes = info.window_bytes;
        Cam::AsyncPipeline::RingLayout layout;
        TORCH_CHECK(Full::PlanLayout(info.window_bytes, info.world_size, prototype_.slotCount, 1, layout),
            "A5 ring does not fit the HCCL window");
        arguments_ = CopyArgs(prototype_);
        const int64_t bootstrap = Full::EncodeCall(Full::Mode::Initialize, prototype_.slotCount);
        if (IsAttention()) {
            auto hidden = at::empty({1, static_cast<int64_t>(prototype_.hiddenSize)}, anchor_.options());
            auto ids = at::empty({1, static_cast<int64_t>(prototype_.topk)}, anchor_.options().dtype(at::kInt));
            Dispatch(hidden, ids, arguments_, bootstrap, 0);
        } else {
            auto hidden = at::empty({1, static_cast<int64_t>(prototype_.hiddenSize)}, anchor_.options());
            auto metadata = at::empty({static_cast<int64_t>(5 + prototype_.tpSize
                + prototype_.expertsPerRank * prototype_.tpSize)}, anchor_.options().dtype(at::kLong));
            SendCombine(hidden, metadata, arguments_, bootstrap);
        }
        const auto initialized = arguments_.to(at::kCPU).contiguous();
        std::memcpy(&prototype_, initialized.data_ptr(), sizeof(prototype_));
        TORCH_CHECK(prototype_.error == 0 && prototype_.localWindow != 0, "A5 MC2 window initialization failed");
        uint64_t token = 0;
        CheckHccl(afd_a5_register_window(group_.c_str(), prototype_.localWindow, prototype_.windowBytes, &window_handle_, &token), "A5 window token");
        std::random_device random;
        const uint64_t nonce = ((uint64_t{random()} << 32) | random()) & Full::MAX_EPOCH;
        uint64_t geometry = 1469598103934665603ULL;
        for (const uint64_t value : {prototype_.attentionRanks, prototype_.moeRanks,
            prototype_.expertsPerRank, prototype_.tpSize, max_sequence_, prototype_.hiddenSize,
            prototype_.topk, prototype_.slotCount, static_cast<uint64_t>(quant_), prototype_.receiveCapacity,
            static_cast<uint64_t>(anchor_.scalar_type()), Full::FRAME_PROTOCOL_FINGERPRINT, Full::CONTEXT_VERSION})
            geometry = (geometry ^ value) * 1099511628211ULL;
        local_info_ = {prototype_.localWindow, prototype_.windowBytes, token,
            nonce == 0 ? 1 : nonce, geometry, prototype_.localRank};
        prepared_ = true;
        std::vector<int64_t> result;
        for (const uint64_t value : local_info_) {
            int64_t encoded;
            std::memcpy(&encoded, &value, sizeof(encoded));
            result.push_back(encoded);
        }
        return result;
    }

    void configure(const std::vector<std::vector<int64_t>> &resources)
    {
        TORCH_CHECK(prepared_ && !configured_ && !released_, "Prepare A5 resources before configuration");
        const uint64_t world = prototype_.attentionRanks + prototype_.moeRanks;
        TORCH_CHECK(resources.size() == world, "Incomplete A5 resource exchange");
        for (uint64_t rank = 0; rank < world; ++rank) {
            const auto &entry = resources[rank];
            TORCH_CHECK(entry.size() == EXCHANGE_WORDS
                && static_cast<uint64_t>(entry[1]) == prototype_.windowBytes
                && static_cast<uint64_t>(entry[4]) == local_info_[4]
                && static_cast<uint64_t>(entry[5]) == rank,
                "All ranks must use identical A5 geometry, dtype, capacity and rank identities");
            TORCH_CHECK(entry[0] != 0 && entry[2] != 0, "A5 peer window is not registered");
            windows_.push_back({static_cast<uint64_t>(entry[0]), static_cast<uint64_t>(entry[1]),
                static_cast<uint64_t>(entry[2])});
        }
        prototype_.domain = static_cast<uint64_t>(resources[0][3]);
        TORCH_CHECK(prototype_.domain != 0, "A5 domain identifier is zero");
        const auto stream = stream_->stream(false);
        // Initialization is collective and precedes any communication calls.
        CheckAcl(aclrtMemsetAsync(reinterpret_cast<void *>(prototype_.localWindow), prototype_.windowBytes,
            0, prototype_.windowBytes, stream), "A5 window initialization");
        CheckAcl(aclrtSynchronizeStream(stream), "A5 window initialization completion");
        if (!IsAttention()) {
            staging_ = at::empty({static_cast<int64_t>(prototype_.receiveCapacity),
                static_cast<int64_t>(prototype_.hiddenSize)}, anchor_.options());
            descriptors_ = at::empty({static_cast<int64_t>(Multiply(prototype_.attentionRanks,
                prototype_.expertsPerRank) * sizeof(Cam::AsyncCcu::SendSegment) / sizeof(int64_t))},
                anchor_.options().dtype(at::kLong));
            plan_status_ = at::ones({PLAN_AND_TRANSPORT_STATUS_WORDS}, anchor_.options().dtype(at::kLong));
            source_ = {reinterpret_cast<uint64_t>(staging_.data_ptr()), static_cast<uint64_t>(staging_.nbytes()), 0};
            prototype_.sourceCapacityBytes = source_.bytes;
            prototype_.descriptors = reinterpret_cast<uint64_t>(descriptors_.data_ptr());
            prototype_.descriptorCapacity = prototype_.attentionRanks * prototype_.expertsPerRank;
            prototype_.planStatus = reinterpret_cast<uint64_t>(plan_status_.data_ptr());
        }
        configured_ = true;
    }

    void initialize()
    {
        TORCH_CHECK(configured_ && !initialized_ && !released_, "Configure A5 frames before registration");
        AfdA5Transport *transport = nullptr;
        const int setup = afd_a5_transport_create(group_.c_str(), prototype_.attentionRanks,
            prototype_.expertsPerRank, window_handle_, windows_.data(), windows_.size(), source_, &transport);
        transport_.reset(transport, afd_a5_transport_destroy);
        CheckHccl(setup, "A5 channel and CCU kernel registration");
        arguments_ = CopyArgs(prototype_);
        stream_->synchronize();
        initialized_ = true;
    }

    at::Tensor dispatch_send(const at::Tensor &x, const at::Tensor &ids, int64_t layer)
    {
        RequireReady(true);
        auto found = reservations_.end();
        for (size_t offset = 0; offset < reservations_.size(); ++offset) {
            const size_t candidate = (next_slot_ + offset) % reservations_.size();
            if (!reservations_[candidate].pending) {
                found = reservations_.begin() + candidate;
                break;
            }
        }
        TORCH_CHECK(found != reservations_.end(), "A5 slots exhausted: enqueue a matching combine before another dispatch");
        const uint64_t slot = found - reservations_.begin();
        auto ticket = arguments_.clone();
        Dispatch(x, ids, ticket, Full::EncodeCall(Full::Mode::Dispatch, prototype_.slotCount, slot), layer);
        *found = {ticket, true};
        next_slot_ = (slot + 1) % reservations_.size();
        Retain({x, ids, ticket});
        return ticket;
    }

    std::vector<at::Tensor> dispatch_recv()
    {
        RequireReady(false);
        auto ticket = arguments_.clone();
        auto outputs = cam_async::dispatch_recv<true>(anchor_, ticket,
            Full::EncodeCall(Full::Mode::Receive, prototype_.slotCount), max_sequence_, prototype_.hiddenSize,
            prototype_.topk, prototype_.moeRanks, prototype_.attentionRanks, prototype_.expertsPerRank,
            prototype_.localRank, prototype_.attentionRanks + prototype_.moeRanks, prototype_.tpSize, quant_, group_);
        received_.push_back({ticket, outputs[2]});
        outputs.push_back(ticket);
        Retain(outputs);
        return outputs;
    }

    void combine_send(const at::Tensor &x, const at::Tensor &metadata, const at::Tensor &ticket)
    {
        RequireReady(false);
        auto found = std::find_if(received_.begin(), received_.end(), [&ticket, &metadata](const auto &entry) {
            return entry.first.is_same(ticket) && entry.second.is_same(metadata);
        });
        TORCH_CHECK(found != received_.end(), "A5 combine requires its original receive ticket and complete metadata");
        TORCH_CHECK(!x.requires_grad(), "A5 expert outputs must not require gradients");
        TORCH_CHECK(x.dim() == 2 && x.size(1) == static_cast<int64_t>(prototype_.hiddenSize)
            && x.size(0) <= static_cast<int64_t>(prototype_.receiveCapacity)
            && x.scalar_type() == anchor_.scalar_type(), "A5 expert output exceeds its registered staging arena");
        staging_.narrow(0, 0, x.size(0)).copy_(x);
        ticket.view(at::kLong).select(0, SOURCE_CAPACITY_WORD).fill_(x.nbytes());
        plan_status_.fill_(static_cast<int64_t>(Full::Error::InvalidMetadata));
        SendCombine(x, metadata, ticket, Full::EncodeCall(Full::Mode::PrepareCombine, prototype_.slotCount));
        const auto stream = stream_->stream(false);
        const auto transport = transport_;
        const uint64_t descriptors = prototype_.descriptors;
        const uint64_t status = prototype_.planStatus;
        // ACLNN uses torch-npu's task queue. Queue CCU on that same queue, so it
        // cannot overtake a descriptor producer which has not been submitted yet.
        at_npu::native::OpCommand command;
        command.Name("AfdAsyncA5CcuCombine");
        command.SetCustomHandler([transport, stream, descriptors, status]() -> int {
            const int result = afd_a5_transport_launch(transport.get(), stream, descriptors, status);
            TORCH_CHECK(result == 0, "A5 CCU launch failed: ", result);
            // This marker is queued after every CCU task on the same runtime
            // stream. A partial host submission leaves it invalid even if a
            // later queued publisher is reached during error handling.
            CheckAcl(aclrtMemsetAsync(reinterpret_cast<void *>(status + sizeof(uint64_t)),
                sizeof(uint64_t), 0, sizeof(uint64_t), stream), "A5 CCU retirement marker");
            return result;
        });
        command.Run();
        SendCombine(x, metadata, ticket, Full::EncodeCall(Full::Mode::PublishCombine, prototype_.slotCount));
        Retain({x, metadata, ticket});
        received_.erase(found);
    }

    at::Tensor combine_recv(const at::Tensor &ids, const at::Tensor &weights, const at::Tensor &ticket)
    {
        RequireReady(true);
        auto found = std::find_if(reservations_.begin(), reservations_.end(), [&ticket](const auto &entry) {
            return entry.pending && entry.ticket.is_same(ticket);
        });
        TORCH_CHECK(found != reservations_.end(), "Unknown or already combined A5 dispatch ticket");
        auto output = cam_async::combine_recv<true>(anchor_, ids, weights, ticket,
            Full::EncodeCall(Full::Mode::Combine, prototype_.slotCount), ids.size(0), prototype_.hiddenSize,
            prototype_.topk, prototype_.moeRanks, prototype_.attentionRanks, prototype_.expertsPerRank,
            prototype_.localRank, prototype_.attentionRanks + prototype_.moeRanks, group_);
        // This is a host enqueue reservation, not a fabricated completion proof.
        // Reuse executes behind CR on the same stream and waits for device credit.
        *found = {};
        Retain({ids, weights, ticket, output});
        return output;
    }

    void quiesce()
    {
        if (!initialized_) return;
        stream_->synchronize();
        retained_.clear();
        quiesced_ = true;
    }
    void abort()
    {
        TORCH_CHECK(!released_, "A5 context has already been released");
        poisoned_ = true;
        // A separate DMA stream reaches pollers even when the submission stream
        // is waiting for a peer. Never clear the flag or rewind the domain epoch.
        aclrtStream control = nullptr;
        void *host_flag = nullptr;
        CheckAcl(aclrtCreateStream(&control), "A5 abort stream");
        const auto allocated = aclrtMallocHost(&host_flag, sizeof(uint64_t));
        if (allocated != ACL_SUCCESS) {
            aclrtDestroyStream(control);
            CheckAcl(allocated, "A5 abort flag allocation");
        }
        *static_cast<uint64_t *>(host_flag) = 1;
        auto status = aclrtMemcpyAsync(abort_.data_ptr(), sizeof(uint64_t), host_flag,
            sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE, control);
        if (status == ACL_SUCCESS) status = aclrtSynchronizeStream(control);
        aclrtFreeHost(host_flag);
        aclrtDestroyStream(control);
        CheckAcl(status, "A5 abort publication");
        // CANN 9.0 exposes Suspend, not a task-retirement/abort primitive.
        // Do not mistake suspension for quiescence or return credits here.
        // Already submitted CCU tasks must drain before domain destruction.
    }
    void release_after_group_destroyed()
    {
        TORCH_CHECK(quiesced_ || !initialized_, "Quiesce A5 streams before destroying the HCCL domain");
        transport_.reset();
        initialized_ = false;
        released_ = true;
    }

private:
    bool IsAttention() const { return prototype_.localRank < prototype_.attentionRanks; }
    void RequireReady(bool attention)
    {
        TORCH_CHECK(initialized_ && !released_, "A5 context is not initialized");
        TORCH_CHECK(!poisoned_, "Aborted A5 domains cannot be reused; create a new process group");
        TORCH_CHECK(attention == IsAttention(), "A5 operator called on the wrong role");
        if (!submission_bound_) {
            // Setup is complete on its own stream. Bind the first model call,
            // which may run on a worker stream different from initialization.
            stream_ = c10_npu::getCurrentNPUStream();
            submission_bound_ = true;
        }
        TORCH_CHECK(c10_npu::getCurrentNPUStream() == *stream_,
            "A5 full profile requires one submission stream per session");
        while (!retained_.empty() && retained_.front().completion.query()) retained_.pop_front();
        quiesced_ = false;
    }
    at::Tensor CopyArgs(const Full::RuntimeArgs &args)
    {
        auto cpu = at::from_blob(const_cast<Full::RuntimeArgs *>(&args),
            {static_cast<int64_t>(sizeof(args) / sizeof(int16_t))}, at::TensorOptions().dtype(at::kHalf)).clone();
        return cpu.to(anchor_.device());
    }
    void Retain(std::vector<at::Tensor> tensors)
    {
        retained_.emplace_back();
        retained_.back().tensors = std::move(tensors);
        retained_.back().completion.record(*stream_);
    }
    void Dispatch(const at::Tensor &x, const at::Tensor &ids, const at::Tensor &ticket, int64_t mode, int64_t layer)
    {
        cam_async::dispatch_send<true>(x, ids, ticket, mode, max_sequence_, x.size(0), prototype_.hiddenSize,
            prototype_.topk, prototype_.moeRanks, prototype_.attentionRanks, prototype_.expertsPerRank,
            prototype_.localRank, prototype_.attentionRanks + prototype_.moeRanks, layer,
            prototype_.tpSize, quant_, group_);
    }
    void SendCombine(const at::Tensor &x, const at::Tensor &metadata, const at::Tensor &ticket, int64_t mode)
    {
        cam_async::combine_send<true>(x, ticket, metadata, mode, max_sequence_, prototype_.hiddenSize,
            prototype_.topk, prototype_.moeRanks, prototype_.attentionRanks, prototype_.expertsPerRank,
            prototype_.localRank, prototype_.attentionRanks + prototype_.moeRanks, prototype_.tpSize, group_);
    }
    at::Tensor anchor_, arguments_, abort_, staging_, descriptors_, plan_status_;
    const std::string group_;
    Full::RuntimeArgs prototype_{};
    uint64_t max_sequence_{0};
    int64_t quant_{0};
    std::optional<c10_npu::NPUStream> stream_;
    std::shared_ptr<AfdA5Transport> transport_;
    void *window_handle_{nullptr};
    std::array<uint64_t, EXCHANGE_WORDS> local_info_{};
    std::vector<AfdA5Span> windows_;
    AfdA5Span source_{};
    std::deque<RetainedCall> retained_;
    std::vector<DispatchReservation> reservations_;
    size_t next_slot_{0};
    std::vector<std::pair<at::Tensor, at::Tensor>> received_;
    bool prepared_{false};
    bool configured_{false};
    bool initialized_{false};
    bool quiesced_{false};
    bool released_{false};
    bool poisoned_{false};
    bool submission_bound_{false};
};
} // namespace afd_plugin::cam_async

TORCH_LIBRARY_FRAGMENT(afd_ascend, ops) {
    ops.class_<afd_plugin::cam_async::AsyncCamA5Context>("AsyncCamA5Context")
        .def(torch::init<at::Tensor, std::string, std::vector<int64_t>>())
        .def("prepare", &afd_plugin::cam_async::AsyncCamA5Context::prepare)
        .def("configure", &afd_plugin::cam_async::AsyncCamA5Context::configure)
        .def("initialize", &afd_plugin::cam_async::AsyncCamA5Context::initialize)
        .def("dispatch_send", &afd_plugin::cam_async::AsyncCamA5Context::dispatch_send)
        .def("dispatch_recv", &afd_plugin::cam_async::AsyncCamA5Context::dispatch_recv)
        .def("combine_send", &afd_plugin::cam_async::AsyncCamA5Context::combine_send)
        .def("combine_recv", &afd_plugin::cam_async::AsyncCamA5Context::combine_recv)
        .def("quiesce", &afd_plugin::cam_async::AsyncCamA5Context::quiesce)
        .def("abort", &afd_plugin::cam_async::AsyncCamA5Context::abort)
        .def("release_after_group_destroyed", &afd_plugin::cam_async::AsyncCamA5Context::release_after_group_destroyed);
}
