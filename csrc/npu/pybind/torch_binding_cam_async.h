// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#pragma once
#include <torch/extension.h>
#include <string>
#include <vector>
namespace afd_plugin::cam_async {
template <bool EXECUTE_NPU>
at::Tensor dispatch_send(
    const at::Tensor &x, const at::Tensor &expert_ids,
    const at::Tensor &comm_args, const int64_t comm_id,
    const int64_t max_seq_len, const int64_t batch_size,
    const int64_t hidden_size, const int64_t top_k,
    const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t attn_rank_id,
    const int64_t world_size, const int64_t layer_index,
    const int64_t tp_size, const int64_t dynamic_quant,
    c10::string_view group_name);

template <bool EXECUTE_NPU>
std::vector<at::Tensor> dispatch_recv(
    const at::Tensor &x, const at::Tensor &comm_args, const int64_t comm_id,
    const int64_t max_seq_len, const int64_t hidden_size, const int64_t top_k,
    const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t moe_rank_id,
    const int64_t world_size, const int64_t tp_size,
    const int64_t dynamic_quant, c10::string_view group_name);

template <bool EXECUTE_NPU>
at::Tensor combine_send(
    const at::Tensor &expand_x, const at::Tensor &comm_args,
    const at::Tensor &batch_info, const int64_t comm_id,
    const int64_t max_seq_len, const int64_t hidden_size, const int64_t top_k,
    const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t moe_rank_id,
    const int64_t world_size, const int64_t tp_size,
    c10::string_view group_name);

template <bool EXECUTE_NPU>
at::Tensor combine_recv(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &expert_scales, const at::Tensor &comm_args,
    const int64_t comm_id, const int64_t batch_size, const int64_t hidden_size,
    const int64_t top_k, const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t attn_rank_id,
    const int64_t world_size, c10::string_view group_name);
}
