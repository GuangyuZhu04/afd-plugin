# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backport producer-specific SAS sizing to the pinned 0.32 eager worker."""

import torch
from vllm_ascend.attention import dsa_v1

_EXTENDED_SAS_WORDS = 2048


def copy_sparse_flash_metadata(
    target: torch.Tensor, published: torch.Tensor
) -> torch.Tensor:
    """Keep the tensor referenced by deferred tasks and copy every published word."""
    if published.dtype != torch.int32 or published.shape not in (
        (dsa_v1.DSA_METADATA_BUFFER_SIZE,),
        (_EXTENDED_SAS_WORDS,),
    ):
        raise RuntimeError(
            f"Unsupported SparseFlashMla metadata {tuple(published.shape)} {published.dtype}"
        )
    # The registered UMDK worker uses eager execution without graph capture.
    # Deferred metadata events use id(target). resize_ preserves that identity
    # and the allocation is reusable on later steps with the same producer.
    if target.shape != published.shape:
        target.resize_(published.shape)
    return target.copy_(published)


# Pinned AscendDSAMetadataBuilder._build_sas_metadata at bd42eea. Preserve
# the native cache, arguments and window computation. Only the BF16 SAS
# destination follows the producer's extent, as in Ascend 7dcdb690.
def _build_sas_metadata(
    self,
    metadata_cache: dict,
    layer_name: str,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen_q: int | torch.Tensor,
    max_seqlen_kv: int | torch.Tensor,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    ori_win_left: int,
    ori_win_right: int,
    dspark_swa: bool,
) -> torch.Tensor:
    # DSpark widens the SWA window to include the whole draft block.
    # Metadata and execution must use the same window so split-G cores
    # agree on the number of KV iterations and synchronization events.
    kv_plan = dsa_v1.get_dsa_attn_kv_plan(self.vllm_config, self.compressor_ratio)
    sas_metadata = metadata_cache.get(layer_name)
    if sas_metadata is None:
        tp_size = dsa_v1.get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk
        cmp_ratio = (
            dsa_v1._dsa_swa_only_cmp_ratio(self.compressor_ratio, self.vllm_config)
            if self.compressor_ratio <= 1
            else 4
            if self.compressor_ratio == 4
            else 128
        )
        metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(
            self.seqused_q.device
        )
        # Explicit DSpark indices give every query in a draft block the
        # same visible KV set. Plan it as non-causal attention over that
        # set, including when a short context crosses an S2 tile boundary.
        sas_seq_lens = seq_lens
        if dspark_swa:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            sas_seq_lens = torch.minimum(
                seq_lens, query_lens + self.model_config.hf_config.sliding_window
            )
        sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=self.seqused_q,
            seqused_kv=sas_seq_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(seq_lens),
            cmp_topk=index_topk if self.compressor_ratio == 4 else 0,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=0 if dspark_swa else 4,
            cmp_mask_mode=3,  # 3:causal
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="TND",
            layout_kv=dsa_v1._dsa_layout_kv(self.vllm_config),
            has_ori_kv=True,
            has_cmp_kv=self.compressor_ratio > 1,
        )
        metadata_cache[layer_name] = sas_metadata

    # ### PATCH START: use the BF16 producer's full output in the eager worker
    if kv_plan.uses_sparse_flash_mla:
        return copy_sparse_flash_metadata(self.sas_metadata_buffer, sas_metadata)
    # ### PATCH END: use the BF16 producer's full output in the eager worker
    self.sas_metadata_buffer[: dsa_v1.DSA_METADATA_BUFFER_SIZE] = sas_metadata
    return self.sas_metadata_buffer


def install_umdk_sas_metadata_adapter() -> None:
    dsa_v1.AscendDSAMetadataBuilder._build_sas_metadata = _build_sas_metadata
