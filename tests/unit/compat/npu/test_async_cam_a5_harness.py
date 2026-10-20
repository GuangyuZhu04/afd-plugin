# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tests.npu.async_cam_a5_full import full_profile_plan, main
from tests.npu.async_cam_common import HarnessConfig, validate_config


def config() -> HarnessConfig:
    return HarnessConfig(soc_version="950", aiv_count=48, ub_budget_bytes=196608)


def test_full_profile_dry_run_keeps_device_claims_explicit(capsys):
    assert main(["--dry-run", "--aiv-count", "48", "--ub-budget-bytes", "196608"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "plan_only_no_device_validation"
    assert result["transport"] == "custom_ccu"
    assert result["wire_protocol"] == "AFDA5SLT/v1"
    assert result["precision_in_flight_rounds"] == 1
    assert result["mte_pipeline_policy"]["candidate_operators"] == ["dispatch_recv"]


def test_frame_budget_can_reject_a_legacy_acceptable_window():
    case = replace(
        config(),
        hidden_size=4096,
        lengths=(128, 128),
        capacity=256,
        max_seq_len=256,
        window_mb=4,
    )
    validate_config(case)
    with pytest.raises(ValueError, match="Each A5 slot needs"):
        full_profile_plan(case, 8)


@pytest.mark.parametrize("slots", [0, 1, 9])
def test_invalid_slot_count(slots):
    with pytest.raises(ValueError, match="between 2 and 8"):
        full_profile_plan(config(), slots)


def test_full_profile_rejects_a3_and_insufficient_descriptor_ub():
    with pytest.raises(ValueError, match="requires --soc-version 950"):
        full_profile_plan(replace(config(), soc_version="910c"), 2)
    with pytest.raises(ValueError, match="descriptor preparation needs"):
        full_profile_plan(replace(config(), hidden_size=32, ub_budget_bytes=70000), 2)
