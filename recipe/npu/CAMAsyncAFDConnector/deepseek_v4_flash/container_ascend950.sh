#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo 'AFD_IMAGE=<A5 image> AFD_WORK=<host dir> AFD_E2E_DEVICES=0,1,2,3,4,5,6,7 bash container_ascend950.sh [docker run options]'
    exit 0
fi
: "${AFD_IMAGE:?use the qualified A5 aarch64 CANN development/serving image}"
: "${AFD_WORK:?select a persistent host directory}"
: "${AFD_E2E_DEVICES:?select the allocated physical device IDs}"
[[ $AFD_E2E_DEVICES =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid device list" >&2; exit 2; }
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
mkdir -p "$AFD_WORK"
work=$(cd "$AFD_WORK" && pwd -P)
args=(--rm -it --network host --ipc host --workdir /workspace/afd-plugin
      -v "$repo:/workspace/afd-plugin:ro" -v "$work:/work/afd"
      -e AFD_WORK=/work/afd -e AFD_RESULTS=/work/afd/results
      -e "AFD_E2E_DEVICES=$AFD_E2E_DEVICES" -e "ASCEND_RT_VISIBLE_DEVICES=$AFD_E2E_DEVICES"
      -e HCCL_IF_IP -e HCCL_SOCKET_IFNAME
      -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/afd-plugin)
IFS=, read -r -a cards <<< "$AFD_E2E_DEVICES"
for id in "${cards[@]}"; do
    dev="/dev/davinci$id"
    [[ -c $dev ]] || { echo "Missing allocated device $dev" >&2; exit 1; }
    args+=(--device "$dev")
done
for dev in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc /dev/ummu /dev/ubcore /dev/uburma/* /dev/infiniband/*; do
    if [[ -c $dev ]]; then args+=(--device "$dev"); fi
done
for path in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/dcmi /usr/local/bin/npu-smi /etc/hccn.conf /etc/hixlep; do
    if [[ -e $path ]]; then args+=(-v "$path:$path:ro"); fi
done
if [[ -n ${AFD_NPU_E2E_MODEL:-} ]]; then
    [[ -f $AFD_NPU_E2E_MODEL/config.json ]] || { echo "Missing checkpoint config.json" >&2; exit 2; }
    args+=(-v "$AFD_NPU_E2E_MODEL:/models/DeepSeek-V4-Flash:ro" -e AFD_NPU_E2E_MODEL=/models/DeepSeek-V4-Flash)
fi
exec docker run "${args[@]}" "$@" "$AFD_IMAGE" bash
