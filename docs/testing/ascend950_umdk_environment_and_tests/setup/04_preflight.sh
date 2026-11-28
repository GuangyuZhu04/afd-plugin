#!/bin/bash
# Verify the stack before touching a device.
#
# Order matters: each step fails in practice before the next one can run.
#   1. the pinned version gate, which rejects a wheel installed without a
#      trusted digest;
#   2. the four arch35 operators, which register only once
#      afd_plugin._C_ascend is imported -- importing afd_plugin alone is not
#      enough, and the failure reads as a missing schema;
#   3. the DeepEP registered-buffer owner, without which no session builds;
#   4. the vllm-ascend extension, without which _C_ascend is an empty
#      namespace and the model aborts on AttributeError.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"
afd_require_container || exit 1

afd_log_step 'preflight'
afd_diexec "
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
  # The version gate needs the digest of the wheel this environment installed.
  [ -f /work/artifacts/stack-env.sh ] && . /work/artifacts/stack-env.sh
  # Run outside the source tree so the installed wheel is imported rather than
  # the working copy, which has no vendored OPP.
  cd /tmp || exit 1
  /usr/local/python3.12.16/bin/python3 - <<'PY'
import os
import sys


def fail(msg):
    print('FAIL:', msg)
    sys.exit(1)


print('-- distributions --')
import importlib.metadata as md
for name in ('vllm', 'vllm-ascend', 'vllm-afd-plugin', 'deep-ep'):
    try:
        print(f'  {name:16s} {md.version(name)}')
    except Exception:
        print(f'  {name:16s} MISSING')

print('-- pinned stack gate --')
from afd_plugin.compat.npu.umdk_mxfp_stack import check_umdk_mxfp_stack
try:
    print('  ', check_umdk_mxfp_stack())
except Exception as exc:
    fail(f'version gate: {exc}')

print('-- operators --')
import torch, torch_npu  # noqa: F401
import afd_plugin._C_ascend  # registers the arch35 schemas
from afd_plugin.compat.npu.ops import ensure_afd_async_ops_available, uses_umdk_async
ensure_afd_async_ops_available()
print('   ensure_afd_async_ops_available: OK, uses_umdk_async =', uses_umdk_async())
for op, arity in (('afd_async_dispatch_send_arch35', 23),
                  ('afd_async_dispatch_recv_arch35', 28),
                  ('afd_async_combine_send_arch35', 22),
                  ('afd_async_combine_recv_arch35', 24)):
    schema = getattr(torch.ops.umdk_cam_op_lib, op).default._schema
    n = len(schema.arguments)
    if n != arity:
        fail(f'{op} has {n} args, expected {arity}')
    print(f'   {op}: {n} args OK')

print('-- registered-buffer owner --')
import deep_ep
if not hasattr(deep_ep, 'AFDBuffer'):
    fail('deep_ep has no AFDBuffer')
print('   AFDBuffer: present')

print('-- vllm-ascend extension --')
import vllm_ascend
pkg = os.path.dirname(vllm_ascend.__file__)
ext = [f for f in os.listdir(pkg) if f.startswith('vllm_ascend_C') and f.endswith('.so')]
if not ext:
    fail('vllm_ascend_C.so missing; _C_ascend would be empty')
print('   extension:', ext[0])

print()
print('PREFLIGHT_OK')
PY
"

# The checks above end in an exit status, and without this the script reported
# success after printing a traceback: the Python body exits non-zero, docker
# exec forwards it, and nothing looked. A preflight that cannot fail is worse
# than none, because the next script trusts it.
RC=$?
if [ "$RC" -ne 0 ]; then
    echo
    echo "PREFLIGHT_FAILED (exit $RC)" >&2
    echo "  the checks above reported a problem; fix it before running a test" >&2
    exit "$RC"
fi

echo
echo 'environment verified'
