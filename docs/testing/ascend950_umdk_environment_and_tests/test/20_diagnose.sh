#!/bin/bash
# Summarise why a run failed, from its own artefacts.
#
# The launcher prints only a summary; the operator-level cause sits in the
# per-role service logs inside the pytest directory. This separates a device
# trap from a returned failure status, which is the first question when a run
# fails.
#
# Usage: 20_diagnose.sh <run-dir>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../lib/common.sh"

RUN=${1:-}
if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then
    echo "usage: $0 <run-dir>" >&2
    echo "recent runs:" >&2
    ls -dt "$AFD_WORK"/runs/*/ 2>/dev/null | head -5 >&2
    exit 2
fi

exec python3 - "$RUN" <<'PY'
import glob
import json
import os
import re
import sys

run = sys.argv[1]


def read(path):
    return open(path, errors="replace").read().splitlines()


print("rundir:", os.path.basename(run))
print()

verdict = glob.glob(os.path.join(run, "**", "run.json"), recursive=True)
if verdict:
    d = json.load(open(verdict[0]))
    print("verdict:")
    print("  execution        :", (d.get("execution") or {}).get("status"))
    print("  cleanup          :", (d.get("cleanup") or {}).get("status"))
    print("  acceptance_passed:", d.get("acceptance_passed"))
    err = (d.get("execution") or {}).get("error") or ""
    if err:
        print("  first error      :", err.splitlines()[0][:200])
else:
    print("verdict: no run.json; the case did not reach its reporting step")
print()

print("device faults:")
plogs = glob.glob(os.path.join(run, "**", "plog", "**", "*.log"), recursive=True)
sigs = {}
for p in plogs:
    for line in read(p):
        for key in ("error code = 95", "aivec error", "507035", "invalid GM"):
            if key in line:
                sigs[key] = sigs.get(key, 0) + 1
if sigs:
    for k in sorted(sigs):
        print("  %5dx %s" % (sigs[k], k))
else:
    print("  none (a returned status, not a device trap)")
print()

print("operator errors:")
seen = set()
pat = re.compile(r"(\w*(?:Error|Exception)): (.*)")
for name in ("attention.log", "ffn.log", "roundtrip.log", "reference.log"):
    for p in glob.glob(os.path.join(run, "**", name), recursive=True):
        for line in read(p):
            m = pat.search(line)
            if not m or "Traceback" in line:
                continue
            msg = ("%s: %s" % (m.group(1), m.group(2))).strip()[:190]
            if msg[:110] in seen:
                continue
            seen.add(msg[:110])
            print(" ", msg)
            if len(seen) >= 10:
                break
if not seen:
    print("  none")
PY
