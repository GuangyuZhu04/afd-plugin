#!/bin/bash
# Preflight: prove the plugin wheel exposes the four arch35 UMDK operators,
# that the pinned version gate passes, and that DeepEP exposes AFDBuffer.
# Runs on CPU-only terms (no device forward), so it is safe while NPUs are busy.
set -uo pipefail
PY=/usr/local/python3.12.16/bin/python3
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1

# The plugin loads its vendor OPP at import time; export both vendors so the
# operator loader does not cache a partial path (recorded as pitfall P14).
SP=/usr/local/python3.12.16/lib/python3.12/site-packages
A=$SP/afd_plugin/_cann_ops_custom/vendors/afd-plugin
V=$SP/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$A:$V"
export LD_LIBRARY_PATH="$A/op_api/lib:$V/op_api/lib:${LD_LIBRARY_PATH:-}"
export AFD_CUST_OPAPI_LIB_PATH="$A/op_api/lib/libafd_umdk_opapi.so"
export SOC_VERSION=950

# Run outside the source checkout so the installed wheel is imported, not the
# working tree (pitfall P8).
cd /tmp || exit 1

$PY - <<'PY'
import importlib.metadata as m
print('--- distribution versions ---')
for name in ('vllm', 'vllm-ascend', 'vllm-afd-plugin', 'deepep'):
    try:
        print(f'{name:18s} {m.version(name)}')
    except Exception:
        print(f'{name:18s} MISSING')

print('--- pinned stack gate ---')
from afd_plugin.compat.npu.umdk_mxfp_stack import check_umdk_mxfp_stack
print(check_umdk_mxfp_stack())

print('--- operator registration ---')
import torch, torch_npu, afd_plugin._C_ascend  # noqa: F401
from afd_plugin.compat.npu.ops import (
    ensure_afd_async_ops_available,
    get_afd_cust_opapi_path,
    uses_umdk_async,
)
ensure_afd_async_ops_available()
print('ensure_afd_async_ops_available: OK')
print('uses_umdk_async:', uses_umdk_async())
print('cust opapi:', get_afd_cust_opapi_path())

for op in (
    'afd_async_dispatch_send_arch35',
    'afd_async_dispatch_recv_arch35',
    'afd_async_combine_send_arch35',
    'afd_async_combine_recv_arch35',
):
    schema = getattr(torch.ops.umdk_cam_op_lib, op).default._schema
    print(f'{op}: {schema}')

print('--- DeepEP registered-buffer owner ---')
import deep_ep
print('deep_ep version:', getattr(deep_ep, '__version__', 'n/a'))
print('AFDBuffer present:', hasattr(deep_ep, 'AFDBuffer'))
print('PREFLIGHT_OK')
PY
echo "PREFLIGHT_RC=$?"