#!/bin/bash
# Run the DeepSeek-V4-Flash async-CAM acceptance case on eight dies.
#
# The case starts four Attention and four FFN services, issues ten overlapping
# chat requests, and requires strict cleanup. It needs all eight dies and about
# four minutes of wall time: roughly three to bring the services up, then the
# requests either answer or fail immediately.
#
# Before this, run 10_roundtrip.sh dsv4-flash. The model case fails on the same
# transport defects as the roundtrip but takes far longer to say so.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"
afd_require_container || exit 1

SCENARIO=${SCENARIO:-afd-dsv4-flash-async-cam-dp2tp2-ep4}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2400}
CASE_TIMEOUT=${CASE_TIMEOUT:-3600}

afd_require_free_npus 8 || exit 1

RUN=$(afd_run_dir model-e2e)
afd_log_step "model acceptance: $SCENARIO"
echo "  run dir: $RUN"
echo "  devices: $DEVICES"

mkdir -p "$RUN/plog" "$RUN/extra-info"

# Reference the pinned stack versions the worker checks, then hand the whole
# environment to the case in one process so nothing is inherited halfway.
# RUN is a host path; the container reaches the same tree at /work.
CRUN=$(afd_ctr_path "$RUN")

afd_diexec "
  set -uo pipefail
  PY=/usr/local/python3.12.16/bin/python3
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
  [ -f /work/artifacts/stack-env.sh ] && . /work/artifacts/stack-env.sh
  SP=/usr/local/python3.12.16/lib/python3.12/site-packages
  A=\$SP/afd_plugin/_cann_ops_custom/vendors/afd-plugin
  V=\${VLLM_ASCEND_VENDOR:-/mnt/xds/sfs/_cann_ops_custom_copy/vendors/custom_transformer}
  export ASCEND_CUSTOM_OPP_PATH=\"\$A:\$V\"
  export LD_LIBRARY_PATH=\"\$A/op_api/lib:\$V/op_api/lib:\${LD_LIBRARY_PATH:-}\"
  export AFD_CUST_OPAPI_LIB_PATH=\"\$A/op_api/lib/libafd_umdk_opapi.so\"
  export SOC_VERSION=950 HCCL_OP_EXPANSION_MODE=AIV
  export HCCL_CONNECT_TIMEOUT=3600 HCCL_EXEC_TIMEOUT=2000
  export ASCEND_PROCESS_LOG_PATH=\"$CRUN/plog\"
  export ASCEND_WORK_PATH=\"$CRUN\"
  export ASCEND_DUMP_PATH=\"$CRUN/extra-info\"

  # The case reads its deployment from these.
  export AFD_E2E_BACKEND=npu
  export AFD_E2E_DEVICES=\"$DEVICES\"
  export AFD_NPU_E2E_MODEL=\"$AFD_MODEL\"
  export AFD_NPU_E2E_VLLM_BIN=\$(command -v vllm)
  export HCCL_IF_IP=\"$AFD_HCCL_IF_IP\"
  export HCCL_SOCKET_IFNAME=\"$AFD_HCCL_IFNAME\"
  export AFD_NPU_DSV4_E2E_API_PORT=\${API_PORT:-19280}
  export AFD_NPU_DSV4_E2E_AFD_PORT=\${AFD_PORT:-6455}
  export AFD_NPU_E2E_STARTUP_TIMEOUT=$STARTUP_TIMEOUT

  # Give every run its own tests tree: a stale working directory breaks child
  # processes that call os.getcwd() after it is removed.
  TESTS_DIR=/tmp/model-tests-\$\$
  rm -rf \"\$TESTS_DIR\" && mkdir -p \"\$TESTS_DIR\"
  git -C /work/src/afd-plugin archive HEAD tests | tar -x -C \"\$TESTS_DIR\"
  cd \"\$TESTS_DIR\" || exit 1

  timeout $CASE_TIMEOUT \$PY -m pytest -q -s --basetemp \"$CRUN/pytest\" \\
    \"tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[$SCENARIO]\" \\
    > \"$CRUN/e2e.log\" 2>&1
  echo \"  exit=\$?\"
"

afd_log_step 'verdict'
CASE=$(find "$RUN" -name run.json 2>/dev/null | head -1)
if [ -n "$CASE" ]; then
    python3 - "$CASE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print('  execution         :', (d.get('execution') or {}).get('status'))
print('  cleanup           :', (d.get('cleanup') or {}).get('status'))
print('  acceptance_passed :', d.get('acceptance_passed'))
err = (d.get('execution') or {}).get('error') or ''
if err:
    print('  first error       :', err.splitlines()[0][:160])
PY
else
    echo '  no run.json; the case did not reach its reporting step'
fi

echo
echo "evidence: $RUN"
