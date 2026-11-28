#!/bin/bash
# Launch the DeepSeek-V4-Flash async-CAM E2E acceptance case on Ascend 950.
# The case starts 4 Attention + 4 FFN services, issues ten overlapping chat
# requests, and requires strict cleanup. Needs all eight dies free.
set -uo pipefail
PY=/usr/local/python3.12.16/bin/python3
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1

SP=/usr/local/python3.12.16/lib/python3.12/site-packages
A=$SP/afd_plugin/_cann_ops_custom/vendors/afd-plugin
# vllm-ascend is a wheel install here, so it has no _cann_ops_custom tree.
# The plugin resolves the vendor from ASCEND_CUSTOM_OPP_PATH entries named
# custom_transformer that actually contain op_api/lib; point at a valid one.
V=${VLLM_ASCEND_VENDOR:-/mnt/xds/sfs/_cann_ops_custom_copy/vendors/custom_transformer}
export ASCEND_CUSTOM_OPP_PATH="$A:$V"
export LD_LIBRARY_PATH="$A/op_api/lib:$V/op_api/lib:${LD_LIBRARY_PATH:-}"
export AFD_CUST_OPAPI_LIB_PATH="$A/op_api/lib/libafd_umdk_opapi.so"
export SOC_VERSION=950
export HCCL_OP_EXPANSION_MODE=AIV HCCL_CONNECT_TIMEOUT=3600 HCCL_EXEC_TIMEOUT=2000
# vllm-ascend was installed from a locally built wheel, so the plugin's
# provenance gate cannot match a Git commit. Supply the trusted build digest
# recorded when this stack was built (artifacts/stack-env.sh).
if [ -f /work/artifacts/stack-env.sh ]; then
  . /work/artifacts/stack-env.sh
fi

SCENARIO=${SCENARIO:-afd-dsv4-flash-async-cam-dp2tp2-ep4}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
NAME=${NAME:-model-e2e-$(date +%m%d%H%M%S)}
RUN=/work/runs/$NAME
mkdir -p "$RUN/plog" "$RUN/extra-info"
export ASCEND_PROCESS_LOG_PATH="$RUN/plog"
export ASCEND_WORK_PATH="$RUN"
export ASCEND_DUMP_PATH="$RUN/extra-info"

# The A5 case advertises the host interface to both roles.
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES="$DEVICES"
export AFD_NPU_E2E_MODEL=/mnt/xds/sfs/DeepSeek-V4-Flash
export AFD_NPU_E2E_VLLM_BIN=$(command -v vllm)
export HCCL_IF_IP=${HCCL_IF_IP:-192.168.16.111}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp39s0f3}
export AFD_NPU_DSV4_E2E_API_PORT=${API_PORT:-19280}
export AFD_NPU_DSV4_E2E_AFD_PORT=${AFD_PORT:-6455}
export AFD_NPU_E2E_STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}

# Run from a tests-only copy so the installed wheel supplies the plugin, and
# take the tests from the same revision as the tree under test.
# A stale working directory breaks child processes that call os.getcwd()
# after it is removed, so give every run its own freshly created tree.
TESTS_DIR=/tmp/model-tests-$$
rm -rf "$TESTS_DIR" && mkdir -p "$TESTS_DIR"
git -C /work/src/afd-plugin archive HEAD tests | tar -x -C "$TESTS_DIR"
cd "$TESTS_DIR" || exit 1

echo "run=$RUN scenario=$SCENARIO devices=$DEVICES"
timeout ${CASE_TIMEOUT:-3600} $PY -m pytest -q -s --basetemp "$RUN/pytest" \
  "tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[$SCENARIO]" \
  > "$RUN/e2e.log" 2>&1
RC=$?
echo "E2E_RC=$RC"

# Collect the verdict from run.json when the case produced one.
find "$RUN" -name run.json 2>/dev/null | head -1 | while read f; do
  $PY - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
def g(k1,k2):
    v=d.get(k1) or {}
    return v.get(k2) if isinstance(v,dict) else v
print("execution =", g("execution","status"))
print("cleanup   =", g("cleanup","status"))
print("acceptance_passed =", d.get("acceptance_passed"))
PY
done

grep -aoE 'acceptance_passed=True|acceptance_passed=False|[0-9]+ passed|[0-9]+ failed|50703[0-9]|CombineRecv failed|invalid GM'   "$RUN/e2e.log" | sort | uniq -c | head -10
echo MODEL_E2E_DONE
