/* SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * Four registered-buffer schemas from UMDK cam_feature@dcc0350e.
 */
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(umdk_cam_op_lib, m) {
    // The kernel writes caller-owned control buffers in place: round_ledger,
    // afd_workspace (send state and scratch) and send_status are mutable. The
    // jetty inbox writes target the peer control_base, which is the local
    // control_buffer when the peer is this rank, so it is marked mutable too.
    m.def("afd_async_dispatch_send(Tensor x, Tensor sf, Tensor topk_idx, Tensor topk_weights, \
    Tensor token_mask, Tensor task_info, Tensor context, Tensor(a!) control_buffer, \
    Tensor(b!) round_ledger, Tensor(c!) afd_workspace, Tensor jetty_ptrs, Tensor(d!) send_status, \
    str group_ep, int num_ranks, int num_experts, int num_topk, int num_tokens_per_tile, \
    int sf_token_stride, int sf_hidden_stride, int num_sf_packs, int timeout_cycles, \
    int send_status_words, int num_vec_cores) -> Tensor(d!)");
    // The three receive buffers are returned as a fixed-size aliased tuple: the
    // kernel writes them in place and returns the same storage. execution_state
    // and afd_workspace (scratch) are also advanced in place. In FFN transaction
    // mode the kernel builds the next slice in the caller-owned active arena.
    m.def("afd_async_dispatch_recv(Tensor payload, Tensor context, Tensor(f!) active, \
    Tensor(d!) execution_state, Tensor(e!) afd_workspace, Tensor control_buffer, \
    Tensor(a!) recv_x, Tensor(b!) recv_sf, Tensor(c!) recv_src_metadata, str group_ep, \
    int num_ranks, int num_experts, \
    int num_topk, int num_source_lanes, int num_layers, int max_expanded_tokens, \
    int max_padded_tokens, int expert_alignment, int num_tokens_per_stage, int rank_idx, \
    int recv_sf_token_stride, int recv_sf_hidden_stride, int payload_bytes, \
    int control_bytes, int context_bytes, int num_hidden, int num_sf_packs, \
    int num_vec_cores) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
    // The kernel advances the caller-owned control plane in place: the send
    // path writes control_buffer (the peer inbox when the peer is this rank),
    // execution_state and the workspace scratch, and fills completion.
    m.def("afd_async_combine_send(Tensor x, Tensor src_metadata, Tensor context, Tensor(e!) active, \
    Tensor(a!) control_buffer, Tensor(b!) execution_state, Tensor(c!) afd_workspace, Tensor jetty_ptrs, \
    Tensor(d!) completion, str group_ep, int num_src_ranks, int num_ranks, int num_experts, \
    int num_topk, int num_source_lanes, int num_layers, int max_padded_tokens, \
    int num_tokens_per_tile, int rank_idx, int timeout_cycles, int completion_bytes, \
    int num_vec_cores) -> Tensor(d!)");
    // The five preallocated receive buffers are both written and returned as a
    // fixed-size aliased tuple; round_ledger, afd_workspace and the peer credit
    // stores into control_buffer are advanced in place as well.
    m.def("afd_async_combine_recv(Tensor payload, Tensor task_info, Tensor context, \
    Tensor(a!) control_buffer, Tensor(b!) round_ledger, Tensor(c!) afd_workspace, Tensor token_mask, \
    Tensor(d!) combined_x, Tensor(e!) combined_topk_idx, Tensor(f!) topk_weights, Tensor(g!) recv_status, \
    Tensor(h!) return_progress, str group_ep, int num_experts, int num_ranks, int num_topk, \
    int round_step, int start, int num_hidden, int num_token_capacity, int recv_status_words, \
    int return_progress_bytes, int timeout_cycles, int num_vec_cores) \
    -> (Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(h!))");
}
