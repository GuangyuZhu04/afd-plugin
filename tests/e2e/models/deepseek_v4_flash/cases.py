# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: DSV4 single-deployment declarations, separate from dedicated/multi-pod
#              runs.
# Create: 2026-11-13
# Note: Each case declares its topology and acceptance checks.
# History: 2026-11-13 create
"""DSV4 single-deployment declarations, separate from dedicated/multi-pod runs.

Importing IDs is safe. Checkpoint quantization is resolved only by build_case
in the runner child. Existing launch-profile helpers remain model-owned and
are also used by the unmigrated dedicated and multi-pod executors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from ipaddress import ip_address
from pathlib import Path
from shutil import copyfile

from tests.e2e.case import (
    CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY,
    CLEANUP_PROFILE_STRICT,
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    EXECUTION_EAGER,
    EXECUTION_FULL_DECODE_GRAPH,
    NPU_ASYNC_TERMINATION_TIMEOUT_S,
    NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S,
    NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S,
    Case,
    CaseEnvironment,
    DboConfig,
    Deployment,
    Gsm8kSettings,
    Gsm8kTask,
    HashInputCheck,
    Mrv2Check,
    Phase,
    PullMegaCheck,
    Role,
)
from tests.e2e.models.deepseek_v4_flash import config, gpu_config
from tests.e2e.models.deepseek_v4_flash.completions import CompletionTask

GPU_BASIC_IDS = gpu_config.DSV4_GPU_BASIC_SCENARIOS
GPU_DBO_IDS = gpu_config.DSV4_GPU_DBO_SCENARIOS
NPU_ASYNC_IDS = tuple(config.DSV4_ASYNC_SHAPES)
NPU_SYNC_IDS = config.DSV4_SYNC_CAMP2P_SCENARIOS
GSM8K_SAMPLES = 7
GSM8K_THRESHOLD = 0.27
DBO_CONCURRENCY = 12
DBO_MINIMUM_SAMPLES = 2 * DBO_CONCURRENCY
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 8


def _budget(arguments: list[str], name: str) -> int | None:
    """Transfer a model profile's budget to its deployment declaration."""
    if name not in arguments:
        return None
    index = arguments.index(name)
    value = int(arguments[index + 1])
    del arguments[index : index + 2]
    return value


def _profile_arguments(
    case_id: str, environment: CaseEnvironment
) -> argparse.Namespace:
    args = argparse.Namespace(
        scenario=case_id,
        model=environment.model,
        device_backend=environment.device_backend,
        afd_host=environment.afd_host,
        common_vllm_arg=[],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        completion_output_path=environment.output_path / "completions.json",
        gsm8k_output_path=environment.output_path / "gsm8k",
        use_decode_bench_connector=False,
    )
    if environment.device_backend == "gpu":
        gpu_config.configure_scenario(args)
    elif case_id in NPU_ASYNC_IDS:
        config.configure_scenario(args)
    else:
        config.configure_sync_camp2p_scenario(args)
    return args


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend == "gpu":
        return _gpu_case(case_id, environment)
    if environment.device_backend == "npu" and case_id in (
        *NPU_ASYNC_IDS,
        *NPU_SYNC_IDS,
    ):
        return _npu_case(case_id, environment)
    raise ValueError(f"unsupported DSV4 {environment.device_backend} case {case_id!r}")


def _gpu_case(case_id: str, environment: CaseEnvironment) -> Case:
    baseline, graph, dbo, attention_ranks, ffn_ranks = (
        gpu_config.DSV4_GPU_SCENARIO_SETTINGS[case_id]
    )
    args = _profile_arguments(case_id, environment)
    max_num_seqs = _budget(args.common_vllm_arg, "--max-num-seqs")
    max_num_batched_tokens = _budget(args.common_vllm_arg, "--max-num-batched-tokens")
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    roles = ("baseline",) if baseline else ("attention", "ffn")
    checks = (
        (Mrv2Check(roles, graph=graph, dbo=dbo),)
        if not baseline or graph or dbo
        else ()
    )
    phases = []
    for index, name in enumerate(gpu_config.DSV4_GPU_HASH_WINDOWS):
        hash_checks = (
            ()
            if baseline
            else (
                HashInputCheck(
                    graph=graph,
                    dbo=dbo,
                    compare_with=gpu_config.DSV4_GPU_HASH_WINDOWS[index - 1]
                    if index
                    else None,
                ),
            )
        )
        phases.append(
            Phase(
                name,
                CompletionTask(
                    first_request_index=index * config.DSV4_CONCURRENT_REQUESTS,
                    answer_repetitions=gpu_config.DSV4_GPU_ANSWER_REPETITIONS,
                ),
                (*checks, *hash_checks),
            )
        )
    phases.append(
        Phase(
            "gsm8k",
            Gsm8kTask(
                Gsm8kSettings(
                    sample_limit=DBO_MINIMUM_SAMPLES if dbo else GSM8K_SAMPLES,
                    threshold=GSM8K_THRESHOLD,
                    num_concurrent=DBO_CONCURRENCY if dbo else None,
                    minimum_samples=DBO_MINIMUM_SAMPLES if dbo else 1,
                )
            ),
            checks,
        )
    )
    return Case(
        id=case_id,
        model="DeepSeek-V4-Flash",
        served_model_name_prefix="dsv4-flash-gpu",
        deployment=Deployment(
            backend="gpu",
            attention=Role(attention_ranks, execution=execution),
            ffn=None if baseline else Role(ffn_ranks, execution=execution),
            connector=None if baseline else "P2pNcclAFDConnector",
            dbo=DboConfig(DBO_DECODE_THRESHOLD, DBO_PREFILL_THRESHOLD) if dbo else None,
            cudagraph_capture_size=gpu_config.DSV4_GPU_CAPTURE_SIZE if graph else None,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        common_vllm_args=tuple(args.common_vllm_arg),
        phases=tuple(phases),
    )


def _npu_case(case_id: str, environment: CaseEnvironment) -> Case:
    mega = case_id == config.DSV4_ASYNC_CAM_MEGA_SCENARIO
    asynchronous = case_id in NPU_ASYNC_IDS
    shape = None if asynchronous else config.DSV4_SYNC_SHAPES[case_id]
    args = _profile_arguments(case_id, environment)
    max_num_seqs = _budget(args.common_vllm_arg, "--max-num-seqs")
    max_num_batched_tokens = _budget(args.common_vllm_arg, "--max-num-batched-tokens")
    graph = shape is not None and config.sync_use_graph(shape)
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    topology: config.DSV4AsyncShape | config.DSV4SyncShape = (
        config.DSV4_ASYNC_SHAPES[case_id] if shape is None else shape
    )
    attention_ranks = topology.attention_ranks
    ffn_ranks = topology.ffn_ranks
    attention_tp = topology.attention_tp_size
    ffn_tp = 1 if shape is None else shape.ffn_tp_size
    attention_dbo = "--enable-dbo" in args.attention_vllm_arg
    if attention_dbo:
        # The deployment owns the DBO flag. Its A3 profile uses native thresholds.
        args.attention_vllm_arg.remove("--enable-dbo")
    connector_extra_config = (
        json.loads(args.afd_connector_extra_config[0])
        if args.afd_connector_extra_config
        else {}
    )
    worker_environment: tuple[tuple[str, str], ...] = ()
    ffn_config: dict[str, bool | int] = dict(
        config.additional_config("ffn", async_cam=asynchronous)
    )
    if case_id == config.DSV4_ASYNC_CAM_A5_SCENARIO:
        address = ip_address(environment.afd_host)
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError(
                "A5 async CAM requires --afd-host to advertise the HCCL interface IP"
            )
        # UMDK owns the registered data windows. This setting sizes the HCCL domain.
        # Load ordinary packed weights and scales for the local MXFP4 MLP.
        ffn_config["enable_fused_mc2"] = 0
        worker_environment = (
            ("VLLM_LOGGING_LEVEL", "INFO"),
            ("BATCH_SIZE_FACTOR", "1.0"),
            ("HCCL_BUFFSIZE", str(connector_extra_config["hccl_buffer_size"])),
            ("HCCL_OP_EXPANSION_MODE", "AIV"),
            ("HCCL_IF_IP", environment.afd_host),
        )
    checks: tuple[Mrv2Check | PullMegaCheck, ...] = ()
    if mega:
        connector_extra_config["ffn_mega_moe"] = True
        checks = (PullMegaCheck(),)
    elif graph:
        checks = (Mrv2Check(("attention", "ffn"), graph=True),)
    compilation = None if shape is None else config.sync_compilation_config(shape)
    case_type = UmdkFlashCase if case_id == config.DSV4_ASYNC_CAM_A5_SCENARIO else Case
    return case_type(
        id=case_id,
        model="DeepSeek-V4-Flash",
        served_model_name_prefix="dsv4-flash" if asynchronous else "dsv4-flash-sync",
        deployment=Deployment(
            backend="npu",
            attention=Role(
                attention_ranks // attention_tp,
                attention_tp,
                execution,
                dbo=DboConfig() if attention_dbo else None,
                vllm_args=tuple(args.attention_vllm_arg),
                additional_config=tuple(
                    config.additional_config(
                        "attention", async_cam=asynchronous
                    ).items()
                ),
            ),
            ffn=Role(
                ffn_ranks // ffn_tp,
                ffn_tp,
                execution,
                vllm_args=tuple(args.ffn_vllm_arg),
                additional_config=tuple(ffn_config.items()),
            ),
            connector=args.afd_connector,
            compute_gate_on_attention=args.compute_gate_on_attention,
            connector_extra_config=tuple(connector_extra_config.items()),
            pin_attention_address=args.attention_data_parallel_address is not None,
            cleanup_profile=CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY
            if case_id == config.DSV4_ASYNC_CAM_SCENARIO
            else CLEANUP_PROFILE_STRICT,
            shutdown_timeout_s=NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S
            if asynchronous
            else DEFAULT_SHUTDOWN_TIMEOUT_S,
            worker_shutdown_timeout_s=NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S
            if asynchronous
            else None,
            termination_timeout_s=NPU_ASYNC_TERMINATION_TIMEOUT_S
            if asynchronous
            else config.DSV4_PROCESS_TERMINATION_TIMEOUT_S,
            cudagraph_capture_size=(
                shape.cudagraph_capture_size if shape is not None and graph else None
            ),
            compilation_config=tuple(
                (key, tuple(value) if isinstance(value, list) else value)
                for key, value in compilation.items()
            )
            if compilation is not None
            else (),
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        common_vllm_args=tuple(args.common_vllm_arg),
        worker_environment=worker_environment,
        phases=(
            Phase(
                "evaluation",
                CompletionTask(
                    check_answer=True if shape is None else shape.check_answer,
                    umdk_rank_counts=(attention_ranks, ffn_ranks)
                    if case_id == config.DSV4_ASYNC_CAM_A5_SCENARIO
                    else None,
                ),
                checks,
            ),
        ),
    )


class UmdkFlashCase(Case):
    def prepare_worker_environment(
        self, environment: CaseEnvironment
    ) -> dict[str, str]:
        # Resolve the installed packages only on the launch host. --describe
        # remains usable on a CPU-only machine without vLLM or the plugin wheel.
        from afd_plugin.compat.npu.ops import (
            AFD_CUST_OPAPI_ENV,
            get_afd_cann_vendor_path,
        )
        from afd_plugin.compat.npu.umdk_mxfp_stack import (
            UMDK_MXFP_STACK,
            check_umdk_mxfp_stack,
            umdk_worker_environment,
        )

        vendors = umdk_worker_environment()
        versions = check_umdk_mxfp_stack()
        versions.update(
            {pkg: version(pkg) for pkg in ("vllm-afd-plugin", "torch", "torch-npu")}
        )
        out = environment.output_path.resolve()
        logs = {}
        for key, default in (
            ("ASCEND_WORK_PATH", out),
            ("ASCEND_PROCESS_LOG_PATH", out / "plog"),
            ("ASCEND_DUMP_PATH", out / "extra-info"),
        ):
            path = Path(os.environ.get(key) or default).resolve()
            path.mkdir(parents=True, exist_ok=True)
            logs[key] = str(path)

        kernels = get_afd_cann_vendor_path() / "op_impl/ai_core/tbe/kernel/ascend950"
        saved = out / "packaged-kernels"
        saved.mkdir(parents=True, exist_ok=True)
        digests = {}
        for op in ("dispatch_send", "dispatch_recv", "combine_send", "combine_recv"):
            objects = sorted((kernels / ("afd_async_" + op)).glob("*.o"))
            if not objects:
                raise RuntimeError(
                    f"950 wheel has no device object for {op} under {kernels}"
                )
            for obj in objects:
                for source in (obj, obj.with_suffix(".json")):
                    dest = saved / source.name
                    copyfile(source, dest)
                    digests[dest.name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        (out / "umdk-runtime.json").write_text(
            json.dumps(
                {
                    "packages": versions,
                    "source_commits": {name: sha for name, _, sha in UMDK_MXFP_STACK},
                    "packaged_kernel_sha256": digests,
                    "plugin_opapi": vendors[AFD_CUST_OPAPI_ENV],
                    "vendor_search_path": vendors["ASCEND_CUSTOM_OPP_PATH"],
                    "devices": environment.devices,
                    "cann_log_paths": logs,
                },
                indent=2,
            )
            + "\n"
        )
        return {**dict(self.worker_environment), **vendors, **logs}


if __name__ == "__main__":
    from tests.e2e.runner import case_main

    raise SystemExit(
        case_main(
            build_case, (*GPU_BASIC_IDS, *GPU_DBO_IDS, *NPU_ASYNC_IDS, *NPU_SYNC_IDS)
        )
    )
