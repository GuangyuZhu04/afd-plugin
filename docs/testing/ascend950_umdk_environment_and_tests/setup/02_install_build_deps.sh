#!/bin/bash
# Install the OS packages the pinned stack needs to compile.
#
# Two things bite here and are handled explicitly:
#   -- the proxy intermittently returns 503 on the rolling 'update' repo, which
#      aborts a metadata refresh even when the base repo is reachable, so the
#      retry disables that repo;
#   -- openEuler SP4 has no gcc-toolset-14 metapackage, only the split packages.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"
afd_require_container || exit 1

PROXY=${AFD_PROXY:-http://192.168.16.146:3128}

if afd_diexec 'test -f /usr/include/elfutils/libdwfl.h && test -x /opt/openEuler/gcc-toolset-14/enable' 2>/dev/null; then
    echo "build dependencies already present"
    exit 0
fi

afd_log_step "installing build dependencies (through $PROXY)"
afd_diexec "
    export http_proxy=$PROXY https_proxy=$PROXY
    for attempt in 1 2 3; do
        dnf install -y --disablerepo=update \\
            elfutils-devel elfutils-libelf-devel jansson \\
            gcc-toolset-14-gcc gcc-toolset-14-gcc-c++ gcc-toolset-14-binutils \\
            gcc-toolset-14-libstdc++-devel gcc-toolset-14-libgcc \\
            > /work/logs/dnf-builddeps.log 2>&1 && break
        rm -f /var/cache/dnf/metadata_lock.pid
        sleep 10
    done
    tail -3 /work/logs/dnf-builddeps.log
"

afd_log_step "verify"
afd_diexec 'ls /usr/include/elfutils/libdwfl.h >/dev/null && echo "  libdwfl: OK" || { echo "  libdwfl: MISSING"; exit 1; }; source /opt/openEuler/gcc-toolset-14/enable && echo "  gcc: $(gcc --version | head -1)"'

echo
echo "build dependencies ready"
echo "next: setup/03_build_stack.sh"
