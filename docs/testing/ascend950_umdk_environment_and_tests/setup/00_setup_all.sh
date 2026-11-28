#!/bin/bash
# Build the whole environment from nothing.
#
# Use this after a container is recycled: the wheels live on the persistent
# mount, so a rebuild only recreates the container and reinstalls them. Pass
# --clean to force a full rebuild of the wheels as well.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"

CLEAN=0
[ "${1:-}" = "--clean" ] && CLEAN=1

afd_log_step 'container'
bash "$HERE/01_create_container.sh"

afd_log_step 'OS build dependencies'
bash "$HERE/02_install_build_deps.sh"

if [ "$CLEAN" = 1 ]; then
    afd_log_step 'discarding cached wheels'
    rm -f "$AFD_WORK"/artifacts/dist/*.whl
fi

if ls "$AFD_WORK"/artifacts/dist/vllm_afd_plugin-*.whl >/dev/null 2>&1 \
   && ls "$AFD_WORK"/artifacts/dist/deep_ep-*.whl >/dev/null 2>&1 \
   && ls "$AFD_WORK"/artifacts/dist/vllm_ascend-*.whl >/dev/null 2>&1; then
    afd_log_step 'wheels already built; reinstalling them'
    afd_diexec '
      set -uo pipefail
      PY=/usr/local/python3.12.16/bin/python3
      VA=\$(ls /work/artifacts/dist/vllm_ascend-*.whl | head -1)
      SHA=\$(sha256sum \"\$VA\" | cut -d' ' -f1)
      mkdir -p /work/artifacts
      echo \"export AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256=\$SHA\" > /work/artifacts/stack-env.sh
      \$PY -m pip install -q --no-index --no-deps --force-reinstall \"\$VA\" 2>&1 | tail -1
      \$PY -m pip install -q --no-index --no-deps --force-reinstall /work/artifacts/dist/deep_ep-*.whl 2>&1 | tail -1
      \$PY -m pip install -q --no-index --no-deps --force-reinstall /work/artifacts/dist/vllm_afd_plugin-*.whl 2>&1 | tail -1'
else
    afd_log_step 'building the stack'
    bash "$HERE/03_build_stack.sh"
fi

afd_log_step 'preflight'
bash "$HERE/04_preflight.sh"

echo
echo 'environment ready'
