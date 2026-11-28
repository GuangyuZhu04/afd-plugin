# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A deferred consumer must see the whole SAS result through its saved tensor."""

import sys
from types import SimpleNamespace

import pytest

from tests.unit.fakes.async_ffn import load_host_module


@pytest.fixture
def sas_builder(monkeypatch):
    torch = pytest.importorskip("torch")
    builder_cls = type("Builder", (), {})
    native = SimpleNamespace(
        DSA_METADATA_BUFFER_SIZE=1024,
        AscendDSAMetadataBuilder=builder_cls,
        get_dsa_attn_kv_plan=lambda config, ratio: SimpleNamespace(
            uses_sparse_flash_mla=True
        ),
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.attention", SimpleNamespace(dsa_v1=native)
    )
    patch = load_host_module(
        monkeypatch, "afd_plugin.compat.patches.npu.umdk_flash_attention"
    )
    patch.install_umdk_sas_metadata_adapter()
    builder = builder_cls()
    builder.vllm_config, builder.compressor_ratio = None, 1
    builder.sas_metadata_buffer = torch.empty(1024, dtype=torch.int32)
    return builder


def cached_sas(builder, published):
    # The builder also takes this path when a sibling cache group published first.
    return builder._build_sas_metadata(
        {"swa": published}, "swa", None, None, 4, 4, None, None, 127, 0, False
    )


@pytest.mark.parametrize("words", [1024, 2048])
def test_deferred_consumer_sees_every_word_and_subsequent_updates(sas_builder, words):
    import torch

    consumer = sas_builder.sas_metadata_buffer
    published = torch.arange(words, dtype=torch.int32)
    assert cached_sas(sas_builder, published) is consumer
    torch.testing.assert_close(consumer, published)
    address = consumer.data_ptr()
    updated = published + 3000
    cached_sas(sas_builder, updated)
    assert consumer.data_ptr() == address
    torch.testing.assert_close(consumer, updated)


@pytest.mark.parametrize(
    "shape,dtype", [((1,), "int32"), ((32, 32), "int32"), ((2048,), "float32")]
)
def test_incompatible_sas_does_not_broadcast_or_cast(sas_builder, shape, dtype):
    import torch

    with pytest.raises(RuntimeError, match="Unsupported SparseFlashMla metadata"):
        cached_sas(sas_builder, torch.zeros(shape, dtype=getattr(torch, dtype)))
