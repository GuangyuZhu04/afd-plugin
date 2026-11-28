# Ascend 950 AFD async transport: environment and tests

Scripts and documentation for building an Ascend 950 environment and testing
the AFD asynchronous transport operators on DeepSeek-V4-Flash.

Everything here runs on the **host** and drives a container, because on a shared
host the container is recycled and a host-side script survives that.

## Start here

The scripts expect the checkout at `<workspace>/src/afd-plugin` and treat the
grandparent as the workspace. If yours is anywhere else, export it first:

```bash
export AFD_WORK=/path/to/workspace    # holds src/, runs/, artifacts/
```

Without that, a clone elsewhere is refused rather than guessed: see
[Set the workspace first](docs/environment.md#set-the-workspace-first).

```bash
cd docs/testing/ascend950_umdk_environment_and_tests

bash setup/00_setup_all.sh      # build the environment
bash setup/04_preflight.sh      # verify it without touching a device

bash test/10_roundtrip.sh small        # operator-level, 4 dies, ~1 min
bash test/10_roundtrip.sh dsv4-flash   # operator-level, 8 dies, model geometry
bash test/11_model_e2e.sh              # full services, ten requests, ~4 min
bash test/20_diagnose.sh <run-dir>     # explain a finished run
```

## Documentation

| Document | Covers |
| --- | --- |
| [docs/environment.md](docs/environment.md) | prerequisites, setup order, the three build traps, the version gate, failure modes |
| [docs/tests.md](docs/tests.md) | each test, its pass condition, how to read a failure, status codes, evidence layout |

## Layout

```text
lib/common.sh   paths, container helpers, free-die checks, pinned versions
setup/          container, build dependencies, stack build, preflight
test/           roundtrip, model acceptance, run diagnosis
docs/           the two documents above
```

## Configuration

Scripts read these from the environment and discover the rest. Nothing is
hard-coded to one host except the defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AFD_WORK` | parent of this checkout | persistent workspace, mounted at `/work` |
| `AFD_REPO` | this checkout | afd-plugin source |
| `AFD_IMAGE` | `vllm-ascend-a5:032-megamoe-20261125` | image with the CANN runtime |
| `AFD_DOCKER_HOST` | `unix:///var/run/docker-network-lab.sock` | daemon holding that image |
| `AFD_URMA_ROOT` | `/mnt/xds/sfs/950-DT/images/dependency/libs` | URMA core and provider libraries |
| `AFD_MODEL` | `/mnt/xds/sfs/DeepSeek-V4-Flash` | checkpoint |
| `AFD_PROXY` | empty | HTTP proxy for package installs |
| `AFD_HCCL_IF_IP` | from the default route | rendezvous address |
| `AFD_HCCL_IFNAME` | from the default route | interface carrying it |

## What these tests established

The operator layer works. `507035`, the MTE invalid-GM trap that blocked the
transport for several rounds, is gone: the roundtrip passes at both geometries
and the DeepEP reference pipeline completes.

The model serves. Two consecutive runs of `test/11_model_e2e.sh` reported
`acceptance_passed=True` with ten correct answers out of ten, both roles
started, all eight UMDK buffers registered, and cleanup passed.

That closed the last blocker, a metadata size contract mismatch fixed by the
SAS buffer patch in
[the sibling fix directory](../ascend950_umdk_dsa_sas_metadata_fix_20261128/README.md).
The two runs and their full logs are in
[the end-to-end record](../ascend950_umdk_e2e_pass_20261128/README.md).

One failure seen in earlier runs, `Thread export failed` / `HCCL_E_RUNTIME`,
did not appear in either passing run. It correlates with other tenants starting
or stopping on this shared host, so read it as contention rather than a defect
until a run reproduces it in isolation.

The history behind those statements is in
[the roundtrip record](../ascend950_umdk_dsv4_flash_logs/README.md) and
[the CANN operator gap note](../ascend950_umdk_cann_operator_gap_20261128/README.md).
