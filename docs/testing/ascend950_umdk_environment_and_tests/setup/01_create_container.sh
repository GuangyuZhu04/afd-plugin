#!/bin/bash
# Create the workspace container.
#
# Choices worth knowing:
#   --network host  HCCL and URMA publish on host addresses; port mapping would
#                   break the peer rendezvous the AFD services rely on.
#   --init          PID 1 must reap children. Without it a dnf run leaves
#                   zombies and later dnf invocations block forever on the lock.
#   /work bind      every log, wheel and run artefact stays on the host, so a
#                   recycled container costs only site-packages state.
#   device list     omitting /dev/uburma or an URMA provider library is what
#                   surfaces much later as device error 507035.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"

NAME=${1:-afd-e2e-$(date +%m%d%H%M)}

afd_log_step "preflight: image and host paths"
docker image inspect "$AFD_IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: image $AFD_IMAGE not found on $AFD_DOCKER_HOST" >&2
    echo "       list it with: docker images" >&2
    exit 1
}
for p in "$AFD_URMA_ROOT/urma-core" "$AFD_URMA_ROOT/urma-providers" "$AFD_MODEL"; do
    [ -e "$p" ] || { echo "ERROR: missing host path $p" >&2; exit 1; }
done

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container $NAME already exists; starting it"
    docker start "$NAME" >/dev/null
else
    afd_log_step "creating container $NAME"
    DEV=()
    for d in /dev/davinci_manager /dev/hisi_hdc /dev/ummu /dev/ummu/tid \
             /dev/ubcore /dev/ubcore/ubcore \
             /dev/davinci0 /dev/davinci1 /dev/davinci2 /dev/davinci3 \
             /dev/davinci4 /dev/davinci5 /dev/davinci6 /dev/davinci7; do
        [ -e "$d" ] && DEV+=("--device=$d")
    done
    for d in /dev/uburma/*; do [ -e "$d" ] && DEV+=("--device=$d"); done

    docker run -d --name "$NAME" \
        --network host --privileged --init --shm-size=64g \
        "${DEV[@]}" \
        -v "$AFD_URMA_ROOT/urma-core/liburma.so.0.0.3:/usr/lib64/liburma.so.0.0.3:ro" \
        -v "$AFD_URMA_ROOT/urma-core/liburma_common.so.0.0.1:/usr/lib64/liburma_common.so.0.0.1:ro" \
        -v "$AFD_URMA_ROOT/urma-providers/liburma-udma.so:/usr/lib64/urma/liburma-udma.so:ro" \
        -v "$AFD_URMA_ROOT/urma-providers/liburma_ubagg.so.0.0.3:/usr/lib64/urma/liburma_ubagg.so.0.0.3:ro" \
        -v /etc/hixlep:/etc/hixlep:ro \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /usr/local/dcmi:/usr/local/dcmi:ro \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
        -v /var/log/npu:/usr/slog \
        -v "$AFD_WORK":/work:rw \
        -v /mnt/xds/sfs:/mnt/xds/sfs:ro \
        "$AFD_IMAGE" sleep infinity >/dev/null
    sleep 6
fi

mkdir -p "$AFD_WORK"/{logs,runs,artifacts,src,patches}
printf '%s' "$NAME" > "$AFD_WORK/.container-name"

afd_log_step "verify devices and mounts"
afd_diexec 'echo "  davinci: $(ls /dev/davinci* 2>/dev/null | wc -l)"; echo "  uburma:  $(ls /dev/uburma 2>/dev/null | wc -l)"; echo "  urma libs: $(ls /usr/lib64/liburma*.so* /usr/lib64/urma/*.so* 2>/dev/null | wc -l)"; echo "  model:   $([ -f /mnt/xds/sfs/DeepSeek-V4-Flash/config.json ] && echo present || echo MISSING)"; echo "  python:  $(python3 -V 2>&1)"; echo "  cann:    $(ls -d /usr/local/Ascend/cann-* 2>/dev/null | tail -1)"'

echo
echo "container ready: $NAME"
echo "next: setup/02_install_build_deps.sh"
