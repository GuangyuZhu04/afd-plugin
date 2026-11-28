#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -eo pipefail

if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
Usage: AFD_WORK=/persistent/afd bash setup_ascend950.sh [source|wheels|installed]
Run inside a compatible A5 aarch64 CANN build/serving image. The default is source.
source      build pinned vLLM (empty), vLLM-Ascend (with native ops) and DeepEP
wheels      install VLLM_WHEEL, VLLM_ASCEND_WHEEL and DEEPEP_WHEEL with trusted digests
installed   reuse the image's pinned vLLM, vLLM-Ascend and AFDBuffer
Every mode installs this branch's plugin into a venv under AFD_WORK.
Optional local sources: VLLM_SOURCE, VLLM_ASCEND_SOURCE, DEEPEP_SOURCE.
Optional prebuilt plugin: AFD_PLUGIN_WHEEL and AFD_PLUGIN_WHEEL_SHA256.
See ascend950_umdk.md for inputs, wheel digests and the board test commands.
EOF
    exit 0
fi
mode=${1:-source}
case "$mode" in source|wheels|installed) ;; *) echo "Expected source, wheels or installed" >&2; exit 2;; esac
[[ $# -le 1 ]] || exit 2
: "${AFD_WORK:?set AFD_WORK to a persistent directory outside the checkout}"
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
mkdir -p "$AFD_WORK"
work=$(cd "$AFD_WORK" && pwd -P)
case "$work/" in "$repo/"*) echo "AFD_WORK must be outside $repo" >&2; exit 2;; esac
cann=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
[[ -r $cann ]] || { echo "Missing CANN environment: $cann" >&2; exit 1; }
source "$cann"
set -u
python_bin=${AFD_PYTHON:-python3}
"$python_bin" -c 'import sys,platform; assert sys.version_info[:2] == (3,12) and platform.machine() == "aarch64", (sys.version,platform.machine())'
# A fresh environment per build also keeps the last working setup usable on failure.
build=$(mktemp -d "$work/build.XXXXXXXX")
exec > >(tee "$build/setup.log") 2>&1
"$python_bin" -m venv --system-site-packages "$build/venv"
source "$build/venv/bin/activate"
cd "$build"
mkdir -p wheels
if [[ -n ${AFD_WHEELHOUSE:-} ]]; then
    export PIP_NO_INDEX=1 PIP_FIND_LINKS="$AFD_WHEELHOUSE"
fi
python -m pip install 'setuptools>=77' 'setuptools-scm>=8' wheel 'pybind11>=2.12,<3' 'cmake>=3.26,<4' ninja pytest
if [[ -n ${AFD_RUNTIME_REQUIREMENTS:-} ]]; then
    python -m pip install -r "$AFD_RUNTIME_REQUIREMENTS"
fi
# A toolchain image supplies the vendor-matched torch stack and serving dependencies.
python -c 'import torch,torch_npu; print("torch",torch.__version__,"torch_npu",torch_npu.__version__)'
vllm_sha=5df63a7203cdb9328d824a27b682e77c7b9928be
ascend_sha=bd42eea01d8312c4df469fabaed45f77a12df4d3
deep_sha=25b5153eb2549107c13c8d462e325ac28cea8dea
plugin_sha=$(git -C "$repo" rev-parse HEAD)
export MAX_JOBS=${MAX_JOBS:-4} AFD_BUILD_JOBS=${AFD_BUILD_JOBS:-4}

checkout() {
    local url=$1 sha=$2 dest=$3
    git init -q "$dest"
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch --depth 1 origin "$sha"
    git -C "$dest" checkout -q --detach FETCH_HEAD
    [[ $(git -C "$dest" rev-parse HEAD) == "$sha" ]]
}

install_checked() {
    local file=$1 expected=$2 actual
    [[ -f $file && $expected =~ ^[0-9a-fA-F]{64}$ ]] || { echo "Missing wheel or trusted SHA-256: $file" >&2; return 1; }
    actual=$(sha256sum "$file")
    [[ ${actual%% *} == "${expected,,}" ]] || { echo "Wheel digest mismatch: $file" >&2; return 1; }
    python -m pip install --no-deps --force-reinstall "$file"
}

build_wheel() {
    local name=$1 src=$2
    mkdir "wheels/$name"
    python -m pip wheel --no-deps --no-build-isolation "$build/$src" -w "wheels/$name"
    local files=("wheels/$name"/*.whl)
    [[ ${#files[@]} == 1 && -f ${files[0]} ]]
    built_wheel="$build/${files[0]}"
}

case "$mode" in
source)
    checkout "${VLLM_SOURCE:-https://github.com/vllm-project/vllm.git}" "$vllm_sha" vllm-src
    VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.32.0+empty.g5df63a7 build_wheel vllm vllm-src
    export AFD_UMDK_VLLM_WHEEL_SHA256=$(sha256sum "$built_wheel" | cut -d' ' -f1)
    install_checked "$built_wheel" "$AFD_UMDK_VLLM_WHEEL_SHA256"
    checkout "${VLLM_ASCEND_SOURCE:-https://github.com/vllm-project/vllm-ascend.git}" "$ascend_sha" ascend-src
    git -C ascend-src submodule update --init --recursive
    SOC_VERSION="${AFD_ASCEND_SOC:-ascend950dt_9582}" COMPILE_CUSTOM_KERNELS=1 \
        SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0.dev0+gbd42eea build_wheel ascend ascend-src
    export AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256=$(sha256sum "$built_wheel" | cut -d' ' -f1)
    install_checked "$built_wheel" "$AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256"
    checkout "${DEEPEP_SOURCE:-https://github.com/puxiaokun3721/DeepEP-Ascend.git}" "$deep_sha" deepep-src
    git -C deepep-src submodule update --init --recursive third-party/deep_jit
    [[ $(git -C deepep-src/third-party/deep_jit rev-parse HEAD) == 65f513952b8771408a19202f65e36398522db6db ]]
    build_wheel deepep deepep-src
    python -m pip install --no-deps --force-reinstall "$built_wheel"
    ;;
wheels)
    install_checked "${VLLM_WHEEL:?}" "${AFD_UMDK_VLLM_WHEEL_SHA256:?}"
    install_checked "${VLLM_ASCEND_WHEEL:?}" "${AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256:?}"
    install_checked "${DEEPEP_WHEEL:?}" "${DEEPEP_WHEEL_SHA256:?}"
    ;;
esac
if [[ -n ${AFD_PLUGIN_WHEEL:-} ]]; then
    install_checked "$AFD_PLUGIN_WHEEL" "${AFD_PLUGIN_WHEEL_SHA256:?}"
    plugin_wheel=$AFD_PLUGIN_WHEEL
else
    mkdir plugin-src
    git -C "$repo" archive "$plugin_sha" | tar -x -C plugin-src
    SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1 SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+g${plugin_sha:0:7}" \
        build_wheel plugin plugin-src
    plugin_wheel=$built_wheel
    python -m pip install --no-deps --force-reinstall "$plugin_wheel"
fi
sha256sum "$plugin_wheel" > "$build/plugin-wheel.sha256"
wheel_rev=$(python - <<'PY'
from importlib.metadata import version
import re
v = version("vllm-afd-plugin")
m = re.fullmatch(r"0\.0\.0\+g([0-9a-f]{7,40})", v)
if m is None:
    raise SystemExit(f"Expected the board release wheel with a Git revision, found {v}")
print(m[1])
PY
)
wheel_commit=$(git -C "$repo" rev-parse --verify "$wheel_rev^{commit}")
if ! git -C "$repo" diff --quiet "$wheel_commit" "$plugin_sha" -- \
    afd_plugin csrc umdk_cam_op_lib.py setup.py setup.cfg pyproject.toml MANIFEST.in LICENSE .gitattributes; then
    echo "Installed wheel $wheel_rev and checkout $plugin_sha have different package sources" >&2
    exit 1
fi
printf '#!/usr/bin/env bash\nexec %q -m vllm.entrypoints.cli.main "$@"\n' "$build/venv/bin/python" > "$build/venv/bin/afd-board-vllm"
chmod +x "$build/venv/bin/afd-board-vllm"
# Record variables before starting a fresh Python process: the dynamic loader uses
# LD_LIBRARY_PATH at exec. Do not import the extension to discover its own vendor.
env_file="$build/stack-env.sh"
{
    printf 'source %q\nsource %q\n' "$cann" "$build/venv/bin/activate"
    printf 'export AFD_BOARD_COMMIT=%q SOC_VERSION=950 HCCL_OP_EXPANSION_MODE=AIV\n' "$plugin_sha"
    printf 'export AFD_BOARD_WHEEL_COMMIT=%q\n' "$wheel_commit"
    printf 'export AFD_NPU_E2E_VLLM_BIN=%q\n' "$build/venv/bin/afd-board-vllm"
    for key in AFD_UMDK_VLLM_WHEEL_SHA256 AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256; do
        [[ -z ${!key:-} ]] || printf 'export %s=%q\n' "$key" "${!key}"
    done
    python - <<'PY'
from pathlib import Path
from shlex import quote
from afd_plugin.compat.npu.umdk_mxfp_stack import check_umdk_mxfp_stack, umdk_worker_environment
import sys
print(check_umdk_mxfp_stack(), file=sys.stderr)
for key, value in umdk_worker_environment().items():
    print(f"export {key}={quote(value)}")
PY
} > "$env_file"
set +u
source "$env_file"
set -u
python - "$wheel_rev" <<'PY'
import sys
from importlib.metadata import version
from importlib.util import find_spec
import deep_ep
from afd_plugin.compat.npu.ops import ensure_afd_async_ops_available, uses_umdk_async
assert version("vllm-afd-plugin").endswith("+g" + sys.argv[1]), "plugin wheel revision differs from checkout"
assert hasattr(deep_ep, "AFDBuffer"), "DeepEP wheel has no AFDBuffer"
assert find_spec("vllm_ascend.vllm_ascend_C"), "Pinned vllm-ascend wheel requires COMPILE_CUSTOM_KERNELS=1"
ensure_afd_async_ops_available()
assert uses_umdk_async(), "installed plugin has no 950 registered-buffer operators"
print("Flash adapter", find_spec("afd_plugin.compat.patches.npu.umdk_flash_attention").origin)
PY
# Publish the activation file only after the verified install succeeds.
link=$(mktemp "$work/.stack-env.XXXXXXXX")
printf 'source %q\n' "$env_file" > "$link"
mv -f "$link" "$work/stack-env.sh"
printf 'Source %s/stack-env.sh before running the board tests.\n' "$work"
