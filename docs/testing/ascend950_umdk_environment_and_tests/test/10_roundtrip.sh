#!/bin/bash
# Run the operator-level roundtrip: four registered-buffer operators with an
# exact BF16 oracle, no model and no checkpoint.
#
# This is the cheapest device check. It exercises the transport the Flash case
# uses, and one launch takes about a minute against the model case's thirty.
#
# Usage: 10_roundtrip.sh [small|dsv4-flash] [rows] [devices] [workers]
#   small       H=256, K=2, E=4,  4 ranks (2A2F)
#   dsv4-flash  H=4096, K=6, E=256, 8 ranks (4A4F), the model geometry
#
# dsv4-flash requires exactly eight ranks: the profile is 4A4F and the harness
# rejects a world size that is not sources plus receivers.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"
afd_require_container || exit 1

PROFILE=${1:-small}
ROWS=${2:-}
case "$PROFILE" in
    small)      DEVICES=${3:-0,1,2,3};            WORKERS=${4:-4} ;;
    dsv4-flash) DEVICES=${3:-0,1,2,3,4,5,6,7};    WORKERS=${4:-8} ;;
    *) echo "unknown profile $PROFILE (use small or dsv4-flash)" >&2; exit 2 ;;
esac

afd_require_free_npus "$WORKERS" || exit 1

RUN=$(afd_run_dir "roundtrip-$PROFILE")
afd_log_step "roundtrip $PROFILE rows=${ROWS:-default} devices=$DEVICES"
echo "  run dir: $RUN"

mkdir -p "$RUN/plog" "$RUN/extra-info"

# Build the runner arguments on the host: PROFILE and ROWS are this script's
# variables, and the container has no such names. Escaping them would hand the
# container an unset variable, which set -u turns into an immediate abort.
ARGS="--profile $PROFILE"
[ -n "$ROWS" ] && ARGS="$ARGS --rows $ROWS"
# RUN is a host path; the container reaches the same tree at /work.
CRUN=$(afd_ctr_path "$RUN")

afd_diexec "
  set -uo pipefail
  PY=/usr/local/python3.12.16/bin/python3
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
  SP=/usr/local/python3.12.16/lib/python3.12/site-packages
  A=\$SP/afd_plugin/_cann_ops_custom/vendors/afd-plugin
  # vllm-ascend is a wheel install here, so its own _cann_ops_custom tree does
  # not exist. The plugin resolves the vendor from ASCEND_CUSTOM_OPP_PATH, so
  # point it at a tree that actually carries op_api/lib.
  V=\${VLLM_ASCEND_VENDOR:-/mnt/xds/sfs/_cann_ops_custom_copy/vendors/custom_transformer}
  export ASCEND_CUSTOM_OPP_PATH=\"\$A:\$V\"
  export LD_LIBRARY_PATH=\"\$A/op_api/lib:\$V/op_api/lib:\${LD_LIBRARY_PATH:-}\"
  export AFD_CUST_OPAPI_LIB_PATH=\"\$A/op_api/lib/libafd_umdk_opapi.so\"
  export SOC_VERSION=950 HCCL_OP_EXPANSION_MODE=AIV
  export HCCL_CONNECT_TIMEOUT=3600 HCCL_EXEC_TIMEOUT=2000
  export ASCEND_RT_VISIBLE_DEVICES=\"$DEVICES\"
  # Keep CANN process logs and device dumps on the persistent mount; a lost
  # dump is what makes the next failure unattributable.
  export ASCEND_PROCESS_LOG_PATH=\"$CRUN/plog\"
  export ASCEND_WORK_PATH=\"$CRUN\"
  export ASCEND_DUMP_PATH=\"$CRUN/extra-info\"
  # Run from a tests-only copy so the installed wheel supplies the plugin.
  rm -rf /tmp/rt-tests && mkdir -p /tmp/rt-tests
  git -C /work/src/afd-plugin archive HEAD tests | tar -x -C /tmp/rt-tests
  cd /tmp/rt-tests || exit 1
  timeout 1800 torchrun --standalone --nproc-per-node=$WORKERS \\
    --log-dir \"$CRUN/ranks\" --tee 3 \\
    tests/npu/umdk_roundtrip.py $ARGS > \"$CRUN/roundtrip.log\" 2>&1
  echo \"  exit=$?\"
"

afd_log_step 'verdict'
# Judge the run on the marker the harness prints only after the exact
# comparison passes, not on the exit status of torchrun alone: a rank that
# aborts still exits non-zero but so does a healthy run that the shell mangles.
LOG=$RUN/roundtrip.log
if grep -aq 'numerical roundtrip passed' "$LOG" 2>/dev/null; then
    echo '  result: PASS (BF16 oracle matched on every rank)'
    rc=0
else
    echo '  result: FAIL'
    grep -aoE 'CombineRecv failed|DispatchRecv failed|50703[0-9]|invalid GM|timeout or trap|RuntimeError[^"]{0,90}' \
        "$LOG" 2>/dev/null | sort | uniq -c | head -8
    rc=1
fi
echo
echo "evidence: $RUN"
exit $rc
