#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -euo pipefail

CANN_ROOT=${1:?Usage: build.sh CANN_ROOT OUTPUT_DIR}
OUTPUT_DIR=${2:?Usage: build.sh CANN_ROOT OUTPUT_DIR}
MODULE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SDK_BASE="$CANN_ROOT/aarch64-linux"
mkdir -p "$OUTPUT_DIR"

cmake -S "$MODULE_DIR" -B "$OUTPUT_DIR/host" \
  -DAFD_CANN_ROOT="$CANN_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build "$OUTPUT_DIR/host" -j 2

DEVICE_FLAGS=(-std=c++17 -xcce --cce-aicore-arch=dav-c310-vec \
  --cce-aicore-only -D__DAV_C310__ \
  -I"$SDK_BASE/asc/include" -I"$SDK_BASE/asc/include/basic_api" \
  -I"$SDK_BASE/asc/include/simt_api/cpp" -I"$SDK_BASE/asc/impl/basic_api" \
  -I"$SDK_BASE/asc/impl/basic_api/dav_c310" -I"$SDK_BASE/asc" \
  -I"$SDK_BASE/asc/impl" -I"$SDK_BASE/ascendc/include/highlevel_api" \
  -I"$SDK_BASE/tikcpp/tikcfw" -I"$SDK_BASE/tikcpp/tikcfw/interface" \
  -I"$SDK_BASE/tikcpp/tikcfw/impl" -I"$MODULE_DIR")

"$CANN_ROOT/bin/bisheng" "${DEVICE_FLAGS[@]}" -c "$MODULE_DIR/prepare_segments.cpp" \
  -o "$OUTPUT_DIR/prepare_segments.o" > "$OUTPUT_DIR/producer.log" 2>&1
"$CANN_ROOT/bin/bisheng" "${DEVICE_FLAGS[@]}" -c "$MODULE_DIR/probes/hccl_ccu_client.cpp" \
  -o "$OUTPUT_DIR/hccl_ccu_client.o" > "$OUTPUT_DIR/client.log" 2>&1

# A negative capability probe must fail for the expected missing CCU member.
# A different compiler/include error must never count as capability evidence.
if "$CANN_ROOT/bin/bisheng" "${DEVICE_FLAGS[@]}" -c "$MODULE_DIR/probes/negative_batch_write.cpp" \
    -o "$OUTPUT_DIR/negative_batch_write.o" > "$OUTPUT_DIR/batch_write_negative.log" 2>&1; then
  echo "CCU BatchWrite unexpectedly compiled; re-audit the matching SDK before integration."
  exit 1
fi
python3 - "$OUTPUT_DIR/batch_write_negative.log" <<'CCU_CAPABILITY'
import pathlib
import re
import sys

diagnostic = pathlib.Path(sys.argv[1]).read_text()
if not re.search(
    r"""(?:no member named|has no member named)\s+['"‘’“”]BatchWrite['"‘’“”]|['"‘’“”]BatchWrite['"‘’“”]\s+is not a member""",
    diagnostic,
):
    raise SystemExit(diagnostic)
print("CANN 9.0 / 3101 CCU BatchWrite absence confirmed")
CCU_CAPABILITY

echo "CCU candidate compiled and linked; no communication was launched."
