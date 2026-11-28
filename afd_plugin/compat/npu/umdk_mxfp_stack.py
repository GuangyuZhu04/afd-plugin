# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dependency contract for the DeepSeek-V4-Flash UMDK MXFP4 serving path."""

import json
import os
import re
import subprocess
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from afd_plugin.compat.npu.ops import (
    AFD_CUST_OPAPI_ENV,
    get_afd_cann_vendor_path,
    get_afd_cust_opapi_path,
)

UMDK_MXFP_STACK = (
    ("vllm", "0.32.0+empty.g5df63a7", "5df63a7203cdb9328d824a27b682e77c7b9928be"),
    ("vllm-ascend", "0.32.0.dev0+gbd42eea", "bd42eea01d8312c4df469fabaed45f77a12df4d3"),
)


def load_umdk_attention_extension(*, indexed: bool) -> str:
    """Load the native Attention bindings after the worker selects its device."""
    import torch
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    try:
        extension = import_module("vllm_ascend.vllm_ascend_C")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Cannot load vllm_ascend.vllm_ascend_C. Install the pinned A5 wheel "
            f"built with COMPILE_CUSTOM_KERNELS=1 and its linked vendor. {exc}"
        ) from exc
    names = ["npu_hc_pre_v2", "npu_hc_post"]
    if indexed:
        names.extend(
            (
                "npu_quant_lightning_indexer_v2_metadata",
                "npu_quant_lightning_indexer_v2",
            )
        )
    missing = [
        name
        for name in names
        if not hasattr(torch.ops._C_ascend, name)
        or not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"_C_ascend::{name}", "PrivateUse1"
        )
    ]
    if missing:
        raise RuntimeError(
            f"{extension.__file__} is missing Flash bindings: {', '.join(missing)}"
        )
    return str(extension.__file__)


def _editable_commit(name: str, url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("expected a local editable checkout")
    root = Path(url2pathname(parsed.path)).resolve(strict=True)
    module = name.replace("-", "_")
    spec = find_spec(module)
    if (
        not spec
        or not spec.origin
        or Path(spec.origin).resolve() != root / module / "__init__.py"
    ):
        raise ValueError("the imported package is outside its editable checkout")
    if not (root / ".git").exists():
        raise ValueError(f"missing Git checkout at {root}")
    git = ["git", "--no-optional-locks", "-C", str(root)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError(f"tracked source in {root} has local changes")
    return commit


def _source_commit(name: str, commit: str, origin: dict) -> str:
    if origin.get("dir_info", {}).get("editable"):
        return _editable_commit(name, origin["url"])
    if vcs := origin.get("vcs_info"):
        if vcs.get("vcs") == "git" and vcs.get("commit_id"):
            return vcs["commit_id"]
        raise ValueError("missing Git revision in installer provenance")
    # pip records the archive digest for installs by wheel path or URL. The
    # expected digest must come from the build of this specific pinned commit.
    variable = f"AFD_UMDK_{name.upper().replace('-', '_')}_WHEEL_SHA256"
    expected = os.environ.get(variable, "").lower()
    archive = origin.get("archive_info", {})
    actual = archive.get("hashes", {}).get("sha256") or archive.get(
        "hash", ""
    ).removeprefix("sha256=")
    if re.fullmatch(r"[0-9a-f]{64}", expected) and actual == expected:
        return commit
    raise ValueError(
        f"install from the pinned Git checkout, or install its wheel by path "
        f"with the trusted build digest in {variable} "
        f"(installed archive SHA-256: {actual or 'unknown'})"
    )


def check_umdk_mxfp_stack() -> dict[str, str]:
    """Check pinned worker versions and their installation provenance."""
    versions = {}
    for name, version, commit in UMDK_MXFP_STACK:
        try:
            package = distribution(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"UMDK MXFP4 requires {name}=={version} at {commit}"
            ) from exc
        if package.version != version:
            raise RuntimeError(
                f"UMDK MXFP4 requires {name}=={version} at {commit}, "
                f"found {package.version}"
            )
        try:
            origin = json.loads(package.read_text("direct_url.json") or "{}")
            installed = _source_commit(name, commit, origin)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"Cannot verify UMDK MXFP4 {name} source {commit}: {exc}"
            ) from exc
        if installed != commit:
            raise RuntimeError(
                f"UMDK MXFP4 requires {name} source {commit}, found {installed}"
            )
        versions[name] = package.version
    return versions


def umdk_worker_environment() -> dict[str, str]:
    """Resolve both installed vendors before exec, without importing CANN."""
    plugin = get_afd_cann_vendor_path()
    private_api = get_afd_cust_opapi_path()
    if private_api.name != "libafd_umdk_opapi.so" or not private_api.is_file():
        raise RuntimeError(
            f"Missing Ascend 950 UMDK API under {plugin}. Use the installed "
            "wheel from a tests-only directory or build the extension in place."
        )
    spec = find_spec("vllm_ascend")
    if not spec or not spec.origin:
        raise RuntimeError("The vllm-ascend custom_transformer vendor is required")
    native = (
        Path(spec.origin).resolve().parent
        / "_cann_ops_custom/vendors/custom_transformer"
    )
    inherited = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    candidates = [
        native,
        *(Path(p) for p in inherited if Path(p).name == "custom_transformer"),
    ]
    native = next((p for p in candidates if (p / "op_api/lib").is_dir()), None)
    if native is None:
        raise RuntimeError(
            "Missing custom_transformer vendor and op_api/lib for vllm-ascend"
        )

    def paths(key: str, *required: Path) -> str:
        inherited_paths = os.environ.get(key, "").split(os.pathsep)
        return os.pathsep.join(
            dict.fromkeys(filter(None, (*map(str, required), *inherited_paths)))
        )

    return {
        "ASCEND_CUSTOM_OPP_PATH": paths("ASCEND_CUSTOM_OPP_PATH", native, plugin),
        "LD_LIBRARY_PATH": paths(
            "LD_LIBRARY_PATH", native / "op_api/lib", private_api.parent
        ),
        AFD_CUST_OPAPI_ENV: str(private_api),
    }
