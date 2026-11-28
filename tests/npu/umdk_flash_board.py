# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolate the 950 Flash Attention dependencies and HCCL before model serving."""

import argparse
import ctypes
import json
import os
from datetime import timedelta
from pathlib import Path


def capabilities():
    import torch
    from vllm_ascend.utils import enable_custom_op

    from afd_plugin.compat.npu.umdk_mxfp_stack import (
        check_umdk_mxfp_stack,
        load_umdk_attention_extension,
        load_umdk_ffn_extension,
        umdk_worker_environment,
    )

    print(json.dumps(check_umdk_mxfp_stack()), flush=True)
    vendors = umdk_worker_environment()
    symbols = (
        "aclnnHcPre",
        "aclnnHcPost",
        "aclnnQuantLightningIndexerV2",
        "aclnnVllmAscendQuantLightningIndexerV2Metadata",
        "aclnnSparseFlashMlaMetadata",
        "aclnnSparseAttnSharedkvMetadata",
        "aclnnKvQuantSparseAttnSharedkvMetadata",
        "aclnnRmsNormCast",
    )
    libs = ["libopapi.so"]
    for path in vendors["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep):
        libs.extend(
            str(lib) for lib in sorted((Path(path) / "op_api/lib").glob("*.so"))
        )
    for lib in dict.fromkeys(libs):
        try:
            dll = ctypes.CDLL(lib)
        except OSError as exc:
            print(json.dumps({"library": lib, "load_error": str(exc)}), flush=True)
            continue
        pairs = {
            s: [hasattr(dll, s + suffix) for suffix in ("", "GetWorkspaceSize")]
            for s in symbols
        }
        print(
            json.dumps({"library": lib, "execution_and_workspace": pairs}), flush=True
        )
    extension = load_umdk_attention_extension(indexed=True)
    load_umdk_ffn_extension()
    print(
        json.dumps(
            {
                "extension": extension,
                "enable_custom_op": enable_custom_op(),
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )


def attention():
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend.ops  # noqa: F401 -- initialize native ops in worker import order
    from vllm_ascend.attention import sparse_flash_mla
    from vllm_ascend.attention.dsa_v1 import (
        QLI_METADATA_BUFFER_SIZE,
        SPARSE_FLASH_MLA_METADATA_BUFFER_SIZE,
    )
    from vllm_ascend.device.device_op import DeviceOperator

    from afd_plugin.compat.npu.umdk_mxfp_stack import load_umdk_attention_extension

    torch.npu.set_device(0)
    load_umdk_attention_extension(indexed=True)
    mult, hidden = 4, 4096
    mix = mult * (mult + 2)
    residual = torch.zeros((1, mult, hidden), dtype=torch.bfloat16, device="npu")
    reduced, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
        residual,
        torch.zeros((mix, mult * hidden), dtype=torch.float32, device="npu"),
        torch.ones(3, dtype=torch.float32, device="npu"),
        torch.zeros(mix, dtype=torch.float32, device="npu"),
        mult,
        20,
        1e-6,
        1e-6,
    )
    torch.npu.synchronize()
    result = torch.ops._C_ascend.npu_hc_post(
        reduced.unsqueeze(0),
        residual.unsqueeze(0),
        post.unsqueeze(0),
        comb.unsqueeze(0),
    ).squeeze(0)
    torch.npu.synchronize()
    torch.testing.assert_close(result, residual, rtol=0, atol=0)
    print("HcPre/HcPost zero-input roundtrip passed", flush=True)
    tokens, dim, block = 4, 512, 128
    starts = torch.tensor([0, tokens], device="npu", dtype=torch.int32)
    lengths = starts[1:].contiguous()
    kv = torch.ones((1, block, 1, dim), device="npu", dtype=torch.bfloat16)
    table = torch.zeros((1, 1), device="npu", dtype=torch.int32)
    metadata = torch.empty(
        SPARSE_FLASH_MLA_METADATA_BUFFER_SIZE, dtype=torch.int32, device="npu"
    )
    for heads in (1, 2, 4, 8):
        common = dict(
            cu_seqlens_q=starts,
            seqused_kv=lengths,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=block - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )
        inputs = dict(
            **common,
            num_heads_q=heads,
            num_heads_kv=1,
            head_dim=dim,
            batch_size=1,
            max_seqlen_q=tokens,
            max_seqlen_kv=tokens,
            has_ori_kv=True,
            has_cmp_kv=False,
            cmp_mask_mode=3,
        )
        raw = sparse_flash_mla.sparse_flash_mla_metadata(**inputs)
        torch.npu.synchronize()
        print(f"heads={heads} SAS producer={tuple(raw.shape)} {raw.dtype}", flush=True)
        if raw.shape != metadata.shape or raw.dtype != metadata.dtype:
            raise RuntimeError(
                f"SparseFlashMla producer {raw.shape}/{raw.dtype} disagrees with "
                f"native 0.34 buffer {metadata.shape}/{metadata.dtype}"
            )
        metadata.copy_(raw)
        print(f"heads={heads} SAS consumer={tuple(metadata.shape)}", flush=True)
        query = torch.zeros((tokens, heads, dim), dtype=torch.bfloat16, device="npu")
        output, _ = sparse_flash_mla.sparse_flash_mla(
            query,
            **common,
            ori_kv=kv,
            ori_block_table=table,
            metadata=metadata,
            sinks=torch.zeros(heads, dtype=torch.float32, device="npu"),
            softmax_scale=dim**-0.5,
        )
        torch.npu.synchronize()
        # Zero logits, unit KV and one zero-valued sink. Causal row i sees i+1 KV rows.
        visible = torch.arange(1, tokens + 1, dtype=torch.float32)
        expected = (
            (visible / (visible + 1)).view(tokens, 1, 1).expand(tokens, heads, dim)
        )
        torch.testing.assert_close(
            output.cpu().float(), expected, rtol=0, atol=torch.finfo(torch.bfloat16).eps
        )
    qli = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=64,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        quant_mode=DeviceOperator.get_dsa_indexer_quant_mode(),
        cu_seqlens_q=starts,
        seqused_k=lengths // 4,
        cmp_residual_k=lengths % 4,
        batch_size=1,
        max_seqlen_q=tokens,
        max_seqlen_k=tokens // 4,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device=str(starts.device),
    )
    torch.npu.synchronize()
    print(f"QLI producer={tuple(qli.shape)} {qli.dtype}", flush=True)
    if qli.shape != (QLI_METADATA_BUFFER_SIZE,) or qli.dtype != torch.int32:
        raise RuntimeError(
            "QLI metadata does not match the pinned builder's own buffer"
        )


def hccl():
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group("hccl", timeout=timedelta(minutes=3))
    try:
        world = dist.get_world_size()
        for dtype in (torch.int32, torch.bfloat16):
            value = torch.tensor(
                [dist.get_rank() + 1], device=f"npu:{local}", dtype=dtype
            )
            dist.all_reduce(value)
            torch.npu.synchronize()
            if value.item() != world * (world + 1) // 2:
                raise AssertionError(
                    f"Incorrect {dtype} HcclAllreduce result on rank {local}"
                )
        print(f"HcclAllreduce passed local_rank={local} world_size={world}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capabilities", "attention", "hccl"))
    stage = parser.parse_args().stage
    {"capabilities": capabilities, "attention": attention, "hccl": hccl}[stage]()
