#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -euo pipefail
CANN_ROOT=${1:?Usage: build.sh CANN_ROOT OUTPUT_DIR}
OUTPUT_DIR=${2:?Usage: build.sh CANN_ROOT OUTPUT_DIR}
MODULE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SDK_BASE="$CANN_ROOT/aarch64-linux"
mkdir -p "$OUTPUT_DIR"
# Revalidate the matching host CCU builder and its capability probes.
bash "$MODULE_DIR/../cam_async_ccu/build.sh" "$CANN_ROOT" "$OUTPUT_DIR/ccu"

DEVICE_FLAGS=(-std=c++17 -xcce --cce-aicore-arch=dav-c310-vec \
  --cce-aicore-only -D__DAV_C310__ \
  -I"$SDK_BASE/asc/include" -I"$SDK_BASE/asc/include/basic_api" \
  -I"$SDK_BASE/asc/include/simt_api/cpp" -I"$SDK_BASE/asc/impl/basic_api" \
  -I"$SDK_BASE/asc/impl/basic_api/dav_c310" -I"$SDK_BASE/asc" \
  -I"$SDK_BASE/asc/impl" -I"$SDK_BASE/ascendc/include/highlevel_api" \
  -I"$SDK_BASE/tikcpp/tikcfw" -I"$SDK_BASE/tikcpp/tikcfw/interface" \
  -I"$SDK_BASE/tikcpp/tikcfw/impl" -I"$MODULE_DIR")


for source in simt_descriptor_producer slot_record_codec; do
  "$CANN_ROOT/bin/bisheng" "${DEVICE_FLAGS[@]}" -c "$MODULE_DIR/$source.cpp" \
    -o "$OUTPUT_DIR/$source.o" > "$OUTPUT_DIR/$source.log" 2>&1
  readelf -Ws "$OUTPUT_DIR/$source.o" > "$OUTPUT_DIR/$source.symbols.txt"
done
echo "P3 SIMT and P4 immutable-record codec compiled; no communication was launched."
