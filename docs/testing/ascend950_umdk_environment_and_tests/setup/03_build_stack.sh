#!/bin/bash
# Build and install the three-layer pinned stack inside the container.
#
# Layer notes, each learned from a failed run:
#
#   vLLM        The image already carries a checkout at the pinned commit; only
#               the dist-info version needs restamping or the gate sees another
#               string.
#
#   vllm-ascend The image build is 3260 commits off the pin. Two traps:
#                 - setuptools-scm cannot parse the branch tag name, so pin the
#                   version with SETUPTOOLS_SCM_PRETEND_VERSION;
#                 - COMPILE_CUSTOM_KERNELS=0 skips the extension and yields a
#                   py3-none-any wheel with no vllm_ascend_C.so. That extension
#                   carries the _C_ascend operators, so without it the model
#                   aborts with AttributeError: npu_hc_pre_v2. This script
#                   grafts the prebuilt .so into the wheel instead of paying for
#                   a full CMake rebuild.
#
#   DeepEP      Needs GCC 14 for C++20 <format> plus elfutils headers. Never
#               override _GLIBCXX_USE_CXX11_ABI: the torch wheel is ABI=1.
#
#   afd-plugin  SOC_VERSION=950 selects the arch35 operator sources.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"
afd_require_container || exit 1

DIST=/work/artifacts/dist

afd_log_step 'stamping the vLLM checkout'
afd_diexec "
  SP=/usr/local/python3.12.16/lib/python3.12/site-packages
  VD=\$(ls -d \$SP/vllm-*.dist-info 2>/dev/null | head -1)
  sed -i 's/^Version: .*/Version: $AFD_VLLM_VERSION/' \"\$VD/METADATA\"
  echo \"  now: \$(grep ^Version: \"\$VD/METADATA\")\"
"

afd_log_step 'building the vllm-ascend wheel'
afd_diexec '
  set -uo pipefail
  PY=/usr/local/python3.12.16/bin/python3
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
  mkdir -p /work/artifacts/dist
  cd /work/src/vllm-ascend-pin || exit 1
  rm -rf .git
  SOC_VERSION=ascend950dt_9582 COMPILE_CUSTOM_KERNELS=0     CMAKE_ARGS=-DCOMPILE_CUSTOM_KERNELS=0     SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0.dev0+gbd42eea     $PY -m pip wheel --no-deps --no-build-isolation . -w /work/artifacts/dist     > /work/logs/build-vllm-ascend.log 2>&1
  echo "  rc=$?"'

afd_log_step 'grafting the prebuilt extension into the wheel'
# Without this, _C_ascend is an empty namespace and the model aborts on
# AttributeError: npu_hc_pre_v2 before it reaches any operator.
afd_diexec '
  set -euo pipefail
  W=$(ls /work/artifacts/dist/vllm_ascend-*.whl | head -1)
  SO=/work/src/vllm-ascend-pin/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
  ENTRY=vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
  if [ -f "$SO" ]; then
    TMP=$(mktemp -d)
    ( cd "$TMP" && unzip -q "$W" )
    cp "$SO" "$TMP/$ENTRY"
    ( cd "$TMP" && zip -q "$W" "$ENTRY" )
    rm -rf "$TMP"
    echo "  embedded $(basename "$SO")"
  else
    echo "  WARNING: prebuilt extension missing; _C_ascend will be empty" >&2
  fi'

afd_log_step 'building the afd-plugin 950 UMDK wheel'
afd_diexec '
  set -uo pipefail
  PY=/usr/local/python3.12.16/bin/python3
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
  cd /work/src/afd-plugin || exit 1
  SOC_VERSION=950 AFD_BUILD_ASCEND_OPS=1     SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev100+gcbd8e2a9a     $PY -m pip wheel --no-build-isolation --no-deps . -w /work/artifacts/dist     > /work/logs/build-afd-plugin.log 2>&1
  echo "  rc=$?"'

afd_log_step 'building the DeepEP wheel (GCC 14)'
afd_diexec '
  set -uo pipefail
  PY=/usr/local/python3.12.16/bin/python3
  cd /work/src/deepep || exit 1
  set +u; source /opt/openEuler/gcc-toolset-14/enable; set -u
  $PY -m pip wheel --no-build-isolation --no-deps . -w /work/artifacts/dist     > /work/logs/build-deepep.log 2>&1
  echo "  rc=$?"'

afd_log_step 'installing the built wheels'
afd_diexec "
  set -uo pipefail
  PY=/usr/local/python3.12.16/bin/python3
  VA=\$(ls $DIST/vllm_ascend-*.whl | head -1)
  SHA=\$(sha256sum \"\$VA\" | cut -d' ' -f1)
  mkdir -p /work/artifacts
  # The version gate accepts a wheel install only with a trusted digest, so
  # record it where the test scripts can read it.
  echo \"export AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256=\$SHA\" > /work/artifacts/stack-env.sh
  \$PY -m pip install -q --no-index --no-deps --force-reinstall \"\$VA\" 2>&1 | tail -1
  \$PY -m pip install -q --no-index --no-deps --force-reinstall $DIST/deep_ep-*.whl 2>&1 | tail -1
  \$PY -m pip install -q --no-index --no-deps --force-reinstall $DIST/vllm_afd_plugin-*.whl 2>&1 | tail -1
  echo \"  wheel sha256=\$SHA\"
"

afd_log_step 'verifying the extension landed'
afd_diexec "
  SP=/usr/local/python3.12.16/lib/python3.12/site-packages
  N=\$(ls \$SP/vllm_ascend/*.so 2>/dev/null | wc -l)
  echo \"  vllm_ascend_C.so present: \$N\"
  [ \"\$N\" -gt 0 ] || { echo '  ERROR: extension missing' >&2; exit 1; }
"

echo
echo 'stack installed'
echo 'next: setup/04_preflight.sh'
