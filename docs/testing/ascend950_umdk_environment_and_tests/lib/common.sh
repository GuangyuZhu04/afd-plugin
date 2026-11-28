#!/bin/bash
# Shared configuration for every script in this tree.
#
# Source it, do not execute it. Defaults are discovered from the host where
# possible so the same tree works in a different checkout; anything that
# cannot be discovered is an explicit override with a loud failure.
#
# Overrides (all optional unless marked required):
#   AFD_WORK          host path mounted at /work: holds src/, runs/, artifacts/
#   AFD_REPO          afd-plugin checkout                     (auto)
#   AFD_IMAGE         image holding the CANN runtime
#   AFD_DOCKER_HOST   docker endpoint holding that image
#   AFD_URMA_ROOT     host dir with urma-core/ and urma-providers/
#   AFD_MODEL         DeepSeek-V4-Flash checkpoint directory
#   AFD_PROXY         HTTP proxy for package and git access   (empty = direct)
#   AFD_HCCL_IF_IP    this host rendezvous address            (auto)
#   AFD_HCCL_IFNAME   interface carrying that address         (auto)
set -uo pipefail

# Resolve the checkout that carries this tree: this file lives at
# <repo>/docs/testing/<tree>/lib/common.sh
_afd_self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_afd_testing=$(cd "$_afd_self/../.." && pwd)
AFD_REPO=${AFD_REPO:-$(cd "$_afd_testing/../.." && pwd)}
# The workspace mounted at /work holds the source trees under src/, so it is
# the parent of the checkout, not the checkout itself.
#
# Derivation alone is not safe. A checkout shallow enough to have no grandparent
# (for example /tmp/clone-verify, or any directory directly under /) resolves
# AFD_WORK to "/". The scripts then mount the host root at /work, write their
# run directories into /, and report success. Require the derived path to be a
# real directory that is neither the filesystem root nor the checkout itself.
_afd_derived_work=$(dirname "$(dirname "$AFD_REPO")")
if [ -z "${AFD_WORK:-}" ]; then
    case "$_afd_derived_work" in
        /|"$AFD_REPO")
            echo "ERROR: cannot derive the workspace from \$AFD_REPO=$AFD_REPO" >&2
            echo "       AFD_WORK defaults to the parent of the checkout, which here is" >&2
            echo "       '$_afd_derived_work'. Set it explicitly:" >&2
            echo "         export AFD_WORK=/path/to/workspace" >&2
            echo "       The workspace must hold src/ with the three source trees, and" >&2
            echo "       it is what the container sees at /work." >&2
            # Sourced file: exit terminates the script that sourced it, which
            # is the intent. A "return 1" would only give up on this file and
            # let the caller continue with the bad path.
            exit 1
            ;;
    esac
    AFD_WORK=$_afd_derived_work
fi

# The image is site-specific: it is one team's build of vLLM-Ascend for this
# SoC. It cannot be defaulted to something that exists everywhere, so requiring
# it is honest; a fallback name would fail later, inside the container, with a
# message about a missing module.
AFD_IMAGE=${AFD_IMAGE:-}
AFD_IMAGE_HINT=${AFD_IMAGE_HINT:-vllm-ascend-a5:032-megamoe-20261125}

# Leave the docker endpoint alone unless asked. `docker` already resolves the
# local default socket, and overriding it with a site-specific path turned
# "no such socket" into "image not found" on every other machine.
if [ -n "${AFD_DOCKER_HOST:-}" ]; then
    export DOCKER_HOST=$AFD_DOCKER_HOST
fi

AFD_URMA_ROOT=${AFD_URMA_ROOT:-/mnt/xds/sfs/950-DT/images/dependency/libs}
AFD_MODEL=${AFD_MODEL:-/mnt/xds/sfs/DeepSeek-V4-Flash}
AFD_PROXY=${AFD_PROXY:-}

# Discover the address and interface carrying the default route instead of
# pinning one host's NIC name.
if [ -z "${AFD_HCCL_IF_IP:-}" ] || [ -z "${AFD_HCCL_IFNAME:-}" ]; then
    _afd_route=$(ip -4 route get 1.1.1.1 2>/dev/null | head -1)
    AFD_HCCL_IF_IP=${AFD_HCCL_IF_IP:-$(printf "%s" "$_afd_route" | sed -n "s/.* src \([0-9.]*\).*/\1/p")}
    AFD_HCCL_IFNAME=${AFD_HCCL_IFNAME:-$(printf "%s" "$_afd_route" | sed -n "s/.* dev \([^ ]*\).*/\1/p")}
fi

# Pinned stack: the branch version gate compares these exact strings.
AFD_VLLM_VERSION=${AFD_VLLM_VERSION:-0.32.0+empty.g5df63a7}
AFD_VLLM_COMMIT=${AFD_VLLM_COMMIT:-5df63a7203cdb9328d824a27b682e77c7b9928be}
AFD_VLLM_ASCEND_VERSION=${AFD_VLLM_ASCEND_VERSION:-0.32.0.dev0+gbd42eea}
AFD_VLLM_ASCEND_COMMIT=${AFD_VLLM_ASCEND_COMMIT:-bd42eea01d8312c4df469fabaed45f77a12df4d3}

afd_container() {
    local name
    name=$(cat "$AFD_WORK/.container-name" 2>/dev/null) || return 1
    [ -n "$name" ] || return 1
    docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true || return 1
    printf "%s" "$name"
}

afd_require_container() {
    afd_container >/dev/null && return 0
    echo "ERROR: no running container recorded in $AFD_WORK/.container-name" >&2
    echo "       create one with setup/01_create_container.sh" >&2
    return 1
}

afd_diexec() {
    local name
    name=$(afd_container) || { afd_require_container; return 1; }
    docker exec "$name" bash -lc "$*"
}

# Count dies with no process attached. This host is shared, so every device
# run checks rather than assumes.
afd_free_npus() {
    local i n free=0
    for i in 0 1 2 3 4 5 6 7; do
        n=$(npu-smi info -t proc-mem -i "$i" 2>/dev/null | grep -c 'Process id' || true)
        [ "${n:-1}" -eq 0 ] && free=$((free + 1))
    done
    printf "%s" "$free"
}

afd_require_free_npus() {
    local need=${1:-8} free
    free=$(afd_free_npus)
    [ "$free" -ge "$need" ] && return 0
    echo "ERROR: need $need free dies, found $free" >&2
    echo "       this host is shared; wait for the current tenant to finish" >&2
    return 1
}

afd_log_step() { printf '\n=== %s ===\n' "$*" >&2; }

afd_run_dir() { printf "%s/runs/%s-%s" "$AFD_WORK" "$1" "$(date +%m%d%H%M%S)"; }

# Translate a host path under AFD_WORK to the path the container sees.
# Build artefacts are created by the host but consumed inside the container,
# and the two sides disagree on the prefix: AFD_WORK is bind-mounted at /work.
# Passing the host path straight through produces a redirect to a directory
# that does not exist, which bash reports against the redirect target rather
# than the missing parent.
afd_ctr_path() {
    local p=$1
    case "$p" in
        "$AFD_WORK")   printf "/work" ;;
        "$AFD_WORK"/*) printf "/work%s" "${p#"$AFD_WORK"}" ;;
        *)             printf "%s" "$p" ;;
    esac
}
