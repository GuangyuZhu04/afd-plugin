# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Source provenance and pre-exec vendor paths for the pinned Flash stack."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu import umdk_mxfp_stack as stack


@pytest.mark.parametrize("indexed", [False, True])
def test_hc_only_extension_is_usable_only_without_indexed_layers(monkeypatch, indexed):
    # A schema without a device dispatch is insufficient in this worker.
    ops = SimpleNamespace(
        npu_hc_pre_v2=1, npu_hc_post=1, npu_quant_lightning_indexer_v2_metadata=1
    )
    torch = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=ops),
        _C=SimpleNamespace(
            _dispatch_has_kernel_for_dispatch_key=lambda name, key: name.startswith(
                "_C_ascend::npu_hc_"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.utils",
        SimpleNamespace(bootstrap_custom_op_env=lambda: None),
    )
    monkeypatch.setattr(
        stack, "import_module", lambda _: SimpleNamespace(__file__="vllm_ascend_C.so")
    )
    if indexed:
        with pytest.raises(
            RuntimeError,
            match="vllm_ascend_C.so.*npu_quant_lightning_indexer_v2_metadata",
        ):
            stack.load_umdk_attention_extension(indexed=True)
    else:
        assert stack.load_umdk_attention_extension(indexed=False) == "vllm_ascend_C.so"


def install_metadata(monkeypatch, commit, origin):
    monkeypatch.setattr(stack, "UMDK_MXFP_STACK", (("vllm", "test", commit),))
    monkeypatch.setattr(
        stack,
        "distribution",
        lambda _: SimpleNamespace(
            version="test", read_text=lambda _: json.dumps(origin) if origin else None
        ),
    )


@pytest.mark.parametrize("mode", ["verified", "unlisted", "wrong_digest", "no_origin"])
def test_offline_wheel_requires_digest_from_pinned_build(monkeypatch, mode):
    digest = hashlib.sha256(b"example wheel").hexdigest()
    origin = {"archive_info": {"hashes": {"sha256": digest}}}
    install_metadata(monkeypatch, "a" * 40, {} if mode == "no_origin" else origin)
    key = "AFD_UMDK_VLLM_WHEEL_SHA256"
    monkeypatch.delenv(key, raising=False)
    if mode != "unlisted":
        monkeypatch.setenv(key, "0" * 64 if mode == "wrong_digest" else digest)
    if mode == "verified":
        assert stack.check_umdk_mxfp_stack() == {"vllm": "test"}
    else:
        with pytest.raises(RuntimeError, match=key):
            stack.check_umdk_mxfp_stack()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "editable source"
    module = root / "vllm/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("value = 1\n")
    git = ["git", "-C", str(root)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "vllm"], check=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-qm",
            "source",
        ],
        check=True,
    )
    sha = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(
        stack, "find_spec", lambda _: SimpleNamespace(origin=str(module))
    )
    return root, module, sha


@pytest.mark.parametrize("mode", ["matching", "wrong_head", "dirty", "shadow"])
def test_editable_install_checks_active_clean_checkout(monkeypatch, checkout, mode):
    root, module, sha = checkout
    origin = {"url": root.as_uri(), "dir_info": {"editable": True}}
    install_metadata(monkeypatch, "0" * 40 if mode == "wrong_head" else sha, origin)
    if mode == "dirty":
        module.write_text("value = 2\n")
    elif mode == "shadow":
        monkeypatch.setattr(
            stack,
            "find_spec",
            lambda _: SimpleNamespace(origin=str(root.parent / "__init__.py")),
        )
    if mode == "matching":
        assert stack.check_umdk_mxfp_stack() == {"vllm": "test"}
    else:
        with pytest.raises(RuntimeError, match="vllm"):
            stack.check_umdk_mxfp_stack()


@pytest.mark.parametrize("missing", [None, "plugin", "native"])
def test_worker_resolves_both_vendors_before_exec(tmp_path, monkeypatch, missing):
    native = tmp_path / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    plugin = tmp_path / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    api = plugin / "op_api/lib/libafd_umdk_opapi.so"
    if missing != "native":
        (native / "op_api/lib").mkdir(parents=True)
    if missing != "plugin":
        api.parent.mkdir(parents=True)
        api.touch()
    monkeypatch.setattr(stack, "get_afd_cann_vendor_path", lambda: plugin)
    monkeypatch.setattr(stack, "get_afd_cust_opapi_path", lambda: api)
    monkeypatch.setattr(
        stack,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(tmp_path / "vllm_ascend/__init__.py")),
    )
    monkeypatch.setenv(
        "ASCEND_CUSTOM_OPP_PATH", os.pathsep.join([str(plugin), "other"])
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "cann")
    if missing:
        with pytest.raises(RuntimeError, match="950 UMDK API|custom_transformer"):
            stack.umdk_worker_environment()
    else:
        env = stack.umdk_worker_environment()
        assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [
            str(native),
            str(plugin),
            "other",
        ]
        assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [
            str(native / "op_api/lib"),
            str(api.parent),
            "cann",
        ]
        assert env["AFD_CUST_OPAPI_LIB_PATH"] == str(api)
