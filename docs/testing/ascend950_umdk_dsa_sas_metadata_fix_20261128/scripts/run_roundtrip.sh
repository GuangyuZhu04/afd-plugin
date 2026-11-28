#!/bin/bash
# Exercise the plugin's registered-buffer transport on real devices.
#
# The env contract follows docs/npu/ascend950_umdk_bringup.md: run from a
# tests-only copy so the installed wheel (with its vendored OPP) is imported
# instead of the working tree, and keep every artefact on the host mount.
set -uo pipefail
PY=/usr/local/python3.12.16/bin/python3
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1

SP=/usr/local/python3.12.16/lib/python3.12/site-packages
A=$SP/afd_plugin/_cann_ops_custom/vendors/afd-plugin
V=$SP/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$A:$V"
export LD_LIBRARY_PATH="$A/op_api/lib:$V/op_api/lib:${LD_LIBRARY_PATH:-}"
export AFD_CUST_OPAPI_LIB_PATH="$A/op_api/lib/libafd_umdk_opapi.so"
export SOC_VERSION=950
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=2000

PROFILE=${PROFILE:-small}
ROWS=${ROWS:-}
DEVICES=${DEVICES:-0,1,2,3}
NWORKERS=${NWORKERS:-4}
RUN_NAME=${RUN_NAME:-roundtrip-$PROFILE-$(date +%m%d%H%M%S)}
RUN=/work/runs/$RUN_NAME

rm -rf "$RUN" && mkdir -p "$RUN/plog" "$RUN/extra-info"
# Device dumps and CANN process logs must land on the host mount: the previous
# bisect lost its dump because it lived in the container filesystem.
export ASCEND_PROCESS_LOG_PATH="$RUN/plog"
export ASCEND_WORK_PATH="$RUN"
export ASCEND_DUMP_PATH="$RUN/extra-info"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

# tests-only harness: same revision as the wheel, sources off PYTHONPATH
rm -rf /tmp/tests-only && mkdir -p /tmp/tests-only
cp -R /work/src/afd-plugin/tests /tmp/tests-only/
cp -R /work/src/afd-plugin/tests/npu/deepep_pipeline_cam.patch /tmp/tests-only/tests/npu/ 2>/dev/null || true
cd /tmp/tests-only || exit 1

ARGS="--profile $PROFILE"
[ -n "$ROWS" ] && ARGS="$ARGS --rows $ROWS"

echo "run=$RUN profile=$PROFILE rows=${ROWS:-default} devices=$DEVICES"
timeout 1800 torchrun --standalone --nproc-per-node=$NWORKERS \
  --log-dir "$RUN/ranks" --tee 3 \
  tests/npu/umdk_roundtrip.py $ARGS > "$RUN/roundtrip.log" 2>&1
echo "ROUNDTRIP_RC=$?"
grep -aoE 'UMDK Ascend 950 [a-z0-9-]+ numerical roundtrip passed|50703[0-9]|timeout or trap|invalid GM|RuntimeError[^"]{0,90}' \
  "$RUN/roundtrip.log" | sort | uniq -c | head -10
echo ROUNDTRIP_DONE