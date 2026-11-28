#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_ascend950.sh [preflight|attention|hccl|roundtrip|model|all]
Default: all. Source AFD_WORK/stack-env.sh first, or set AFD_WORK.
AFD_RESULTS selects persistent storage. AFD_E2E_DEVICES is a comma-separated
list of allocated physical dies (attention: >=1, hccl: >=2, roundtrip: 4 or 8,
model/all: 8). Model additionally needs AFD_NPU_E2E_MODEL, HCCL_IF_IP and
HCCL_SOCKET_IFNAME. Each invocation creates a fresh run directory.
EOF
}
stage=${1:-all}
case "$stage" in --help|-h) usage; exit 0;; preflight|attention|hccl|roundtrip|model|all) ;; *) usage >&2; exit 2;; esac
[[ $# -le 1 ]] || exit 2
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
if [[ -z ${AFD_BOARD_COMMIT:-} ]]; then
    source "${AFD_WORK:?activate stack-env.sh or set AFD_WORK}/stack-env.sh"
fi
set -u
commit=$(git -C "$repo" rev-parse HEAD)
[[ $AFD_BOARD_COMMIT == "$commit" ]] || { echo "Test checkout $commit differs from installed $AFD_BOARD_COMMIT" >&2; exit 2; }
: "${AFD_RESULTS:?set AFD_RESULTS to a persistent results directory}"
mkdir -p "$AFD_RESULTS"
results=$(cd "$AFD_RESULTS" && pwd -P)
run=$(mktemp -d "$results/flash-${stage}.XXXXXXXX")
printf 'Results: %s\n' "$run"
exec > >(tee "$run/console.log") 2>&1
phase=setup
trap 'code=$?; printf "%s\t%d\n" "$phase" "$code" > "$run/exit-status.tsv"' EXIT
mkdir "$run/checkout"
git -C "$repo" archive "$commit" tests | tar -x -C "$run/checkout"
cd "$run/checkout"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_PROCESS_LOG_PATH="$run/plog" ASCEND_DUMP_PATH="$run/extra-info" ASCEND_WORK_PATH="$run"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_DUMP_PATH"
ids=${AFD_E2E_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-}}
printf 'commit=%s\npython=%s\ndevices=%s\nstage=%s\n' "$commit" "$(command -v python)" "$ids" "$stage" > "$run/manifest.txt"
python - "$commit" >> "$run/manifest.txt" <<'PY'
import sys
from importlib.metadata import distribution
from pathlib import Path
import afd_plugin
from afd_plugin.compat.npu.ops import get_afd_cust_opapi_path
pkg = distribution("vllm-afd-plugin")
installed = Path(pkg.locate_file("afd_plugin/__init__.py")).resolve()
if Path(afd_plugin.__file__).resolve() != installed or not pkg.version.endswith("+g" + sys.argv[1][:7]):
    raise RuntimeError(f"Loaded {afd_plugin.__file__} ({pkg.version}); expected the wheel for {sys.argv[1]}")
print("plugin", pkg.version, installed, pkg.read_text("direct_url.json"))
print("plugin_opapi", get_afd_cust_opapi_path())
PY
for info in "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/version.cfg" /usr/local/Ascend/driver/version.info; do
    if [[ -r $info ]]; then cat "$info" >> "$run/manifest.txt"; fi
done
python -m pip freeze > "$run/packages.txt"
if [[ $stage != preflight ]]; then
    [[ $ids =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Set AFD_E2E_DEVICES to allocated device IDs" >&2; exit 2; }
    IFS=, read -r -a cards <<< "$ids"
    unique=$(printf '%s\n' "${cards[@]}" | sort -nu | wc -l)
    [[ $unique -eq ${#cards[@]} ]] || { echo "Duplicate device IDs: $ids" >&2; exit 2; }
    case "$stage" in
        all|model) [[ ${#cards[@]} == 8 ]] || { echo "4A4F needs eight devices" >&2; exit 2; };;
        roundtrip) [[ ${#cards[@]} == 4 || ${#cards[@]} == 8 ]] || { echo "Roundtrip needs four or eight devices" >&2; exit 2; };;
        hccl) [[ ${#cards[@]} -ge 2 ]] || { echo "HCCL needs at least two devices" >&2; exit 2; };;
    esac
    export ASCEND_RT_VISIBLE_DEVICES="$ids"
    if command -v npu-smi >/dev/null; then npu-smi info > "$run/npu-smi.txt" 2>&1 || true; fi
fi

run_phase() {
    phase=$1
    shift
    printf '\nRunning %s\n' "$phase"
    "$@" 2>&1 | tee "$run/$phase.log"
}
run_ranks() {
    local name=$1 devices=$2 workers=$3
    shift 3
    run_phase "$name" env ASCEND_RT_VISIBLE_DEVICES="$devices" \
        python -m torch.distributed.run --standalone --nproc-per-node="$workers" \
        --max-restarts=0 --tee 3 --log-dir "$run/$name-ranks" "$@"
}
if [[ $stage == preflight || $stage == all ]]; then
    run_phase meta python -m pytest -o addopts= -q tests/npu/test_umdk_meta_dispatch.py \
        tests/npu/test_umdk_meta_combine.py tests/npu/test_umdk_experts.py
    run_phase capabilities python tests/npu/umdk_flash_board.py capabilities
fi
if [[ $stage == attention || $stage == all ]]; then
    run_phase attention env ASCEND_RT_VISIBLE_DEVICES="${cards[0]}" python tests/npu/umdk_flash_board.py attention
fi
if [[ $stage == hccl || $stage == all ]]; then
    run_ranks hccl-pair "${cards[0]},${cards[1]}" 2 tests/npu/umdk_flash_board.py hccl
    if [[ ${#cards[@]} -gt 2 ]]; then
        run_ranks hccl-group "$ids" "${#cards[@]}" tests/npu/umdk_flash_board.py hccl
    fi
fi
if [[ $stage == roundtrip || $stage == all ]]; then
    four=$(IFS=,; echo "${cards[*]:0:4}")
    for rows in 1 65; do
        run_ranks "small-$rows" "$four" 4 tests/npu/umdk_roundtrip.py --profile small --rows "$rows"
    done
    if [[ ${#cards[@]} == 8 ]]; then
        for rows in 1 64 4096; do
            run_ranks "flash-$rows" "$ids" 8 tests/npu/umdk_roundtrip.py --profile dsv4-flash --rows "$rows"
        done
    fi
fi
if [[ $stage == model || $stage == all ]]; then
    phase=model-setup
    : "${AFD_NPU_E2E_MODEL:?set the absolute FP4 checkpoint path}" "${HCCL_IF_IP:?}" "${HCCL_SOCKET_IFNAME:?}"
    [[ -f $AFD_NPU_E2E_MODEL/config.json ]] || { echo "Missing checkpoint config.json" >&2; exit 2; }
    export AFD_E2E_BACKEND=npu AFD_E2E_DEVICES="$ids"
    export AFD_NPU_E2E_VLLM_BIN=${AFD_NPU_E2E_VLLM_BIN:-$(command -v vllm)}
    export AFD_NPU_E2E_STARTUP_TIMEOUT=${AFD_NPU_E2E_STARTUP_TIMEOUT:-2400}
    export AFD_WORKER_LOG_DIR="$run/workers"
    mkdir "$AFD_WORKER_LOG_DIR"
    python -m tests.npu.umdk_worker_logs "$run/logging.json"
    export VLLM_LOGGING_CONFIG_PATH="$run/logging.json"
    scenario=afd-dsv4-flash-async-cam-dp2tp2-ep4
    run_phase model python -m pytest -o addopts= -q -s --basetemp "$run/model" \
        "tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[$scenario]"
fi
phase=passed
