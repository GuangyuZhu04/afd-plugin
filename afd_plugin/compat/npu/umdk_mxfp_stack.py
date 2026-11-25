# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dependency contract for the DeepSeek-V4-Flash UMDK MXFP4 serving path."""

import json
from importlib.metadata import PackageNotFoundError, distribution

UMDK_MXFP_STACK = (
    ("vllm", "0.32.0+empty.g5df63a7", "5df63a7203cdb9328d824a27b682e77c7b9928be"),
    ("vllm-ascend", "0.32.0.dev0+gbd42eea", "bd42eea01d8312c4df469fabaed45f77a12df4d3"),
)


def check_umdk_mxfp_stack() -> dict[str, str]:
    """Check worker distributions, including the full Git revision when recorded."""
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
        origin = json.loads(package.read_text("direct_url.json") or "{}")
        vcs = origin.get("vcs_info", {})
        if vcs.get("vcs") == "git" and vcs.get("commit_id") != commit:
            raise RuntimeError(
                f"UMDK MXFP4 requires {name} source {commit}, "
                f"found {vcs.get('commit_id')}"
            )
        versions[name] = package.version
    return versions
