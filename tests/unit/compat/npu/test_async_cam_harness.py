# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for the route oracle and safety preflight (no torch required)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from tests.npu.async_cam_common import (
    CONSERVATIVE_UB_BYTES,
    MAX_A5_RANKS,
    MAX_AIV_COUNT,
    SIMT_RESERVED_UB_BYTES,
    HarnessConfig,
    NativeHarness,
    chunk_plans,
    describe_config,
    f32,
    quant_dynamic_scale,
    quant_stored_scale,
    quantize_row,
    route_ids,
    unified_buffer_requirements,
    validate_config,
    validate_metadata,
    window_requirements,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


class AsyncCamPlannerTests(unittest.TestCase):
    def test_routed_rows_preserve_expert_rank_token_order_and_topk(self):
        config = HarnessConfig(capacity=8)
        plans = [chunk_plans(config, owner) for owner in range(config.moe_ranks)]
        all_rows = [
            row for owner_plans in plans for plan in owner_plans for row in plan.rows
        ]
        expected = [
            (expert, rank, token)
            for rank in range(config.attn_ranks)
            for token in range(config.lengths[rank])
            for expert in route_ids(config, rank, token)
        ]
        self.assertCountEqual(all_rows, expected)
        self.assertTrue(any(len(owner_plans) > 1 for owner_plans in plans))
        for owner_plans in plans:
            next_expert = 0
            for plan in owner_plans:
                self.assertEqual(plan.start, next_expert)
                self.assertEqual(list(plan.rows), sorted(plan.rows))
                self.assertEqual(sum(plan.counts), len(plan.rows))
                self.assertLessEqual(len(plan.rows), config.capacity)
                next_expert = plan.end + 1
            self.assertEqual(next_expert, config.experts_per_rank)

    def test_sparse_empty_owner_keeps_final_notification(self):
        config = HarnessConfig(routing="sparse", capacity=16)
        empty = chunk_plans(config, 1)
        self.assertEqual(len(empty), 1)
        self.assertEqual(empty[0].rows, ())
        self.assertEqual(
            (empty[0].start, empty[0].end), (0, config.experts_per_rank - 1)
        )
        self.assertEqual(
            empty[0].prefixes, tuple(length * config.top_k for length in config.lengths)
        )
        self.assertTrue(all(count == 0 for count in empty[0].counts))

    def test_tp_group_metadata_distinguishes_full_total_from_chunk_rows(self):
        config = HarnessConfig(
            attn_ranks=4, lengths=(7, 5, 3, 9), max_seq_len=18, capacity=8
        )
        plans = chunk_plans(config, 0)
        self.assertEqual({plan.tp_base for plan in plans}, {0, 2})
        self.assertTrue(any(plan.total_rows > len(plan.rows) for plan in plans))
        for plan in plans:
            self.assertEqual(
                len(plan.metadata(13)),
                5 + config.tp_size + config.experts_per_rank * config.tp_size,
            )
            self.assertEqual(sum(plan.table), plan.total_rows)
            validate_metadata(plan, 13, list(plan.metadata(13)), list(plan.counts))

    def test_metadata_detects_stale_layer_wrong_prefix_and_wrong_count(self):
        plan = chunk_plans(HarnessConfig(), 1)[0]
        info = list(plan.metadata(7))
        for index in (0, 1, 2, 3, 4, 5, len(info) - 1):
            malformed = info.copy()
            malformed[index] += 1
            with self.assertRaises(AssertionError):
                validate_metadata(plan, 7, malformed, list(plan.counts))
        counts = list(plan.counts)
        counts[0] += 1
        with self.assertRaises(AssertionError):
            validate_metadata(plan, 7, info, counts)

    def test_capacity_checks_tp_aggregated_whole_expert(self):
        # Each source sends 3 rows/expert, but their TP group needs 6.
        config = HarnessConfig(
            lengths=(3, 3), max_seq_len=6, routing="sparse", capacity=4
        )
        with self.assertRaisesRegex(ValueError, "whole experts must fit"):
            validate_config(config)

    def test_rejects_invalid_bounds_without_device_packages(self):
        base = HarnessConfig()
        invalid = (
            replace(base, lengths=(0, 5)),
            replace(base, tp_size=3),
            replace(base, top_k=9),
            replace(base, hidden_size=33),
            replace(base, capacity=3),
            replace(base, max_seq_len=262145),
            replace(base, lengths=(65536, 5), max_seq_len=131072),
            replace(base, timeout_seconds=float("nan")),
            replace(base, max_seq_len=10),
            replace(base, window_mb=0),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_quantization_contract_multiplies_by_127_over_amax(self):
        # An amax that is an integer multiple of 127/2 makes the multiplier an
        # exact power of two, so the expected integers are hand-checkable. A
        # reference that divided by 127 instead of multiplying by 127/amax
        # would return all zeros here, and one that used amax/127 as the
        # multiplier would be off by a factor of 127^2.
        self.assertEqual(quant_dynamic_scale(1.984375), 64.0)
        self.assertEqual(
            quantize_row([1.984375, 1.0, 0.5, -1.984375, 0.0]), [127, 64, 32, -127, 0]
        )

    def test_quantization_contract_rounds_half_to_even(self):
        # An amax of 127/64 gives the multiplier 64.0 exactly, so the odd
        # eighths below are exact ties at the .5 mark. They must round to the
        # even neighbour -- 0, 2, 2, 4 rather than the round-half-up 1, 2, 3, 4
        # -- because the kernel casts with CAST_RINT.
        row = [1.984375, 1 / 128, 3 / 128, 5 / 128, 7 / 128, -1.984375]
        self.assertEqual(quantize_row(row), [127, 0, 2, 2, 4, -127])

    def test_quantization_rounds_multiply_to_fp32_before_integer_cast(self):
        # The double product is 63.499998..., but FP32 multiplication gives
        # exactly 63.5 and the subsequent nearest-even integer cast gives 64.
        self.assertEqual(quantize_row([1.5, 0.75, -0.75]), [127, 64, -64])

    def test_stored_scale_is_reciprocal_of_rounded_multiplier(self):
        # The kernel stores 1.0f / (127.0f / amax), not amax / 127.0f. The two
        # agree when the division is exact -- any power of two, and the
        # integer-multiple anchors above -- and disagree in the last bit in
        # general, which is why the reference has to reproduce the kernel's
        # form rather than the algebraically equal one.
        for exact in (2.0, 8.0, 64.0, 1.984375):
            self.assertEqual(quant_stored_scale(exact), f32(f32(exact) / 127.0))
        amax = 0.1926826536655426
        self.assertEqual(quant_stored_scale(amax), 0.0015171861741691828)
        self.assertNotEqual(quant_stored_scale(amax), f32(f32(amax) / 127.0))

    def test_quantization_contract_rejects_undefined_rows(self):
        for row in ([0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], []):
            with self.subTest(row=row), self.assertRaises(ValueError):
                quantize_row(row)

    def test_window_bounds_cover_quantized_and_unquantized_layouts(self):
        config = HarnessConfig(hidden_size=256, max_seq_len=256)
        unquantized = window_requirements(config)
        quantized = window_requirements(replace(config, dynamic_quant=1))
        self.assertEqual(
            unquantized["attention_min_bytes"], quantized["attention_min_bytes"]
        )
        self.assertGreater(unquantized["moe_min_bytes"], quantized["moe_min_bytes"])
        self.assertEqual(unquantized["minimum_bytes"], max(unquantized.values()))

    def test_unified_buffer_bounds_reject_silent_kernel_overflow(self):
        reference = HarnessConfig(
            attn_ranks=8,
            moe_ranks=8,
            tp_size=8,
            experts_per_rank=32,
            top_k=8,
            hidden_size=6144,
            lengths=(16,) * 8,
            max_seq_len=128,
            dynamic_quant=1,
            capacity=128,
        )
        self.assertLessEqual(
            max(unified_buffer_requirements(reference).values()), CONSERVATIVE_UB_BYTES
        )
        validate_config(reference)
        with self.assertRaisesRegex(ValueError, "UB bytes"):
            validate_config(replace(reference, hidden_size=16384))
        with self.assertRaisesRegex(ValueError, "48-AIV owner"):
            validate_config(replace(HarnessConfig(), moe_ranks=49))

    def test_a5_planning_requires_explicit_hardware_limits(self):
        for overrides in (
            {},
            {"aiv_count": 8},
            {"ub_budget_bytes": CONSERVATIVE_UB_BYTES},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "950 requires explicit"),
            ):
                validate_config(HarnessConfig(soc_version="950", **overrides))

    def test_explicit_resource_plan_keeps_the_routing_contract(self):
        a3 = HarnessConfig(lengths=(256, 256), max_seq_len=512, capacity=512)
        # These are synthetic planner inputs, not claimed A5 hardware limits.
        a5 = replace(
            a3, soc_version="950", aiv_count=8, ub_budget_bytes=CONSERVATIVE_UB_BYTES
        )
        validate_config(a5)
        self.assertGreater(
            unified_buffer_requirements(a5)["dispatch_recv"],
            unified_buffer_requirements(a3)["dispatch_recv"],
        )
        self.assertEqual(window_requirements(a5), window_requirements(a3))
        for owner in range(a3.moe_ranks):
            self.assertEqual(chunk_plans(a5, owner), chunk_plans(a3, owner))
        report = describe_config(a5)
        self.assertEqual(report["assumed_aiv_count"], a5.aiv_count)
        self.assertEqual(report["unified_buffer_budget_bytes"], a5.ub_budget_bytes)
        self.assertEqual(report["status"], "plan_only_no_device_validation")

    def test_resource_preflight_enforces_declared_aiv_and_ub_limits(self):
        config = HarnessConfig(
            soc_version="950", aiv_count=8, ub_budget_bytes=CONSERVATIVE_UB_BYTES
        )
        with self.assertRaisesRegex(ValueError, "48-AIV owner"):
            validate_config(replace(config, moe_ranks=MAX_AIV_COUNT + 1))
        with self.assertRaisesRegex(ValueError, "UB bytes"):
            validate_config(replace(config, ub_budget_bytes=1024))
        for overrides in ({"aiv_count": 0}, {"ub_budget_bytes": -1}):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "must be positive"),
            ):
                validate_config(replace(config, **overrides))

    def test_source_core_policy_reports_per_rank_and_preserves_ffn_budget(self):
        a3 = HarnessConfig()
        a5 = replace(
            a3,
            soc_version="950",
            aiv_count=MAX_AIV_COUNT,
            ub_budget_bytes=CONSERVATIVE_UB_BYTES,
        )
        validate_config(a5)
        report = describe_config(a5)
        self.assertEqual(report["assumed_source_aiv_counts"], (7, 5))
        self.assertEqual(report["assumed_ffn_aiv_count"], MAX_AIV_COUNT)
        self.assertEqual(report["source_aiv_policy"]["window_statistics_rows"], 48)
        self.assertEqual(
            unified_buffer_requirements(a5), unified_buffer_requirements(a3)
        )
        self.assertEqual(window_requirements(a5), window_requirements(a3))
        self.assertEqual(
            describe_config(a3)["assumed_source_aiv_counts"], (MAX_AIV_COUNT,) * 2
        )

    def test_a5_active_aiv_count_can_be_smaller_than_moe_owner_count(self):
        config = HarnessConfig(
            soc_version="950", aiv_count=1, ub_budget_bytes=CONSERVATIVE_UB_BYTES
        )
        validate_config(config)
        self.assertEqual(config.moe_ranks, 2)
        self.assertGreater(config.moe_ranks, config.aiv_count)
        self.assertEqual(describe_config(config)["assumed_aiv_count"], 1)

    def test_planner_rejects_aiv_count_above_native_launch_cap(self):
        for soc in ("910c", "950"):
            with (
                self.subTest(soc=soc),
                self.assertRaisesRegex(ValueError, "48-AIV launch cap"),
            ):
                validate_config(
                    HarnessConfig(
                        soc_version=soc,
                        aiv_count=MAX_AIV_COUNT + 1,
                        ub_budget_bytes=CONSERVATIVE_UB_BYTES,
                    )
                )

    def test_a5_rank_preflight_matches_window_pointer_capacity(self):
        config = HarnessConfig(
            soc_version="950",
            attn_ranks=MAX_A5_RANKS - 2,
            lengths=(7,) * (MAX_A5_RANKS - 2),
            aiv_count=MAX_AIV_COUNT,
            ub_budget_bytes=CONSERVATIVE_UB_BYTES,
        )
        self.assertEqual(config.world_size, MAX_A5_RANKS)
        validate_config(config)
        with self.assertRaisesRegex(ValueError, "950 world-size"):
            validate_config(
                replace(config, attn_ranks=MAX_A5_RANKS, lengths=(7,) * MAX_A5_RANKS)
            )

    def test_a5_pipeline_policy_preserves_baseline_ub_acceptance(self):
        a3 = HarnessConfig()
        a5 = replace(
            a3,
            soc_version="950",
            aiv_count=MAX_AIV_COUNT,
            ub_budget_bytes=max(unified_buffer_requirements(a3).values()),
        )
        validate_config(a5)
        self.assertEqual(
            unified_buffer_requirements(a5), unified_buffer_requirements(a3)
        )
        self.assertEqual(
            describe_config(a5)["mte_pipeline_policy"]["candidate_operators"],
            ("dispatch_recv", "combine_send"),
        )
        self.assertEqual(
            describe_config(a3)["mte_pipeline_policy"]["candidate_operators"], ()
        )

    def test_simt_policy_keeps_scalar_fallback_and_baseline_ub_contract(self):
        baseline = HarnessConfig()
        config = replace(
            baseline,
            soc_version="950",
            aiv_count=MAX_AIV_COUNT,
            ub_budget_bytes=max(unified_buffer_requirements(baseline).values()),
        )
        # A shape fitting only the original minimum stays accepted. The optional
        # SIMT arena is an additional native selector condition, not a preflight
        # requirement or evidence that a VF was selected/executed on hardware.
        validate_config(config)
        self.assertEqual(
            unified_buffer_requirements(config), unified_buffer_requirements(baseline)
        )
        policy = describe_config(config)["simt_routing_policy"]
        self.assertEqual(policy["candidate_operator"], "dispatch_send")
        self.assertEqual(policy["reserved_ub_bytes"], SIMT_RESERVED_UB_BYTES)
        self.assertIn("scalar", policy["insufficient_resources"])
        self.assertIsNone(
            describe_config(baseline)["simt_routing_policy"]["candidate_operator"]
        )

    def test_a5_cli_dry_run_reports_explicit_resources_without_qualification(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "tests/npu/async_cam_precision.py"),
                "--dry-run",
                "--soc-version",
                "950",
                "--aiv-count",
                "8",
                "--ub-budget-bytes",
                str(CONSERVATIVE_UB_BYTES),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["config"]["soc_version"], "950")
        self.assertEqual(report["assumed_aiv_count"], 8)
        self.assertEqual(report["status"], "plan_only_no_device_validation")

    def test_cli_help_and_dry_run_use_only_standard_library(self):
        script = REPOSITORY_ROOT / "tests/npu/async_cam_precision.py"
        for arguments in (("--help",), ("--dry-run", "--capacity", "8")):
            result = subprocess.run(
                [sys.executable, "-S", str(script), *arguments],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            if "--dry-run" in arguments:
                self.assertEqual(
                    json.loads(result.stdout)["status"],
                    "plan_only_no_device_validation",
                )

    def test_unmeasured_send_retains_inputs_without_waiting_for_peer(self):
        harness = NativeHarness(HarnessConfig())
        synchronize = Mock()
        harness.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=synchronize))
        harness.operations = {"dispatch_send": Mock(return_value="placeholder")}
        harness._round_keepalive = []
        before = Mock()
        result = harness._invoke("dispatch_send", ("input",), False, before, {}, {})
        self.assertEqual(result, "placeholder")
        self.assertEqual(harness._round_keepalive, [(("input",), "placeholder")])
        synchronize.assert_not_called()
        before.assert_called_once_with("dispatch_send")

    def test_timing_readiness_precedes_fence_and_releases_prior_buffers(self):
        harness = NativeHarness(HarnessConfig())
        order = []
        start = Mock()
        start.record.side_effect = lambda: order.append("start")
        start.elapsed_time.return_value = 0.125
        end = Mock()
        end.record.side_effect = lambda: order.append("end")
        npu = SimpleNamespace(
            synchronize=lambda: order.append("sync"),
            Event=Mock(side_effect=(start, end)),
        )
        harness.torch = SimpleNamespace(npu=npu)
        harness.operations = {"combine_recv": lambda: order.append("op")}
        harness._round_keepalive = [((), "prior_send")]
        event_ms: dict[str, list[float]] = {"combine_recv": []}
        host_ms: dict[str, list[float]] = {"combine_recv": []}
        harness._invoke(
            "combine_recv",
            (),
            True,
            lambda name: order.append("ready"),
            event_ms,
            host_ms,
        )
        self.assertEqual(order, ["ready", "sync", "start", "op", "end", "sync"])
        self.assertEqual(harness._round_keepalive, [])
        self.assertEqual(event_ms["combine_recv"], [0.125])
        self.assertEqual(len(host_ms["combine_recv"]), 1)


if __name__ == "__main__":
    unittest.main()
