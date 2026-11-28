# Environment setup

Build a working Ascend 950 environment for the AFD async-transport operators
and the DeepSeek-V4-Flash acceptance case.

Scripts run on the **host**, not inside the container, and drive the container
through `docker exec`. That keeps them usable after the container is recycled,
which happens on a shared host.

## What you need before starting

| Item | Why |
| --- | --- |
| Eight Ascend 950 dies | the case is 4 Attention + 4 FFN |
| An image with the CANN runtime and a vLLM checkout | tested against `vllm-ascend-a5:032-megamoe-20261125` |
| The URMA runtime and provider libraries on the host | missing ones surface much later as device error 507035 |
| `/etc/hixlep` | endpoint configuration the transport reads |
| The DeepSeek-V4-Flash checkpoint | about 149 GB |
| A docker daemon whose root has room | the images are ~15 GB each |
| A source tree with `vllm-ascend-pin/` and `deepep/` pre-staged | see below |

### Source trees the build expects

The build script assumes these already exist under `$AFD_WORK/src/`:

```text
src/afd-plugin/        this checkout (SOC_VERSION=950 wheel source)
src/vllm-ascend-pin/   vllm-ascend at bd42eea, with its prebuilt
                       vllm_ascend/vllm_ascend_C*.so in place
src/deepep/            DeepEP-Ascend at 25b5153e plus its DeepJIT submodule
```

They are not fetched automatically because two of them need a pinned commit
and a compiler toolchain; see *Obtaining the sources* below.

## Order of operations

```bash
cd docs/testing/ascend950_umdk_environment_and_tests
bash setup/00_setup_all.sh          # container, deps, wheels, install, preflight
```

That runs four steps in sequence; run them individually to see where time
goes:

| Script | Does | Takes |
| --- | --- | --- |
| `setup/01_create_container.sh` | creates the container with the device, mount and network set | seconds |
| `setup/02_install_build_deps.sh` | installs GCC 14 and elfutils headers | 5–15 min, proxy-bound |
| `setup/03_build_stack.sh` | builds and installs all three wheels | 5–10 min |
| `setup/04_preflight.sh` | verifies gate, operators, buffer owner, extension | seconds |

`00_setup_all.sh --clean` discards cached wheels and rebuilds from source.

## What each choice is for

| Choice | Reason |
| --- | --- |
| `--network host` | HCCL and URMA publish on host addresses; port mapping breaks the peer rendezvous |
| `--init` | PID 1 must reap. Without it a `dnf` run leaves zombies and later `dnf` calls block on the lock forever |
| `--privileged` | the transport needs the UB character devices |
| `-v $AFD_WORK:/work` | every log, wheel and run artefact survives container recycling |
| full `/dev/davinci*` list | a missing die fails the 8-rank case at link time |
| `/dev/uburma/*` and `/dev/ubcore` | absent ones surface as `507035` inside the operator, not at setup |
| URMA `.so` files at their SONAMEs | the loader resolves by path, not by `LD_LIBRARY_PATH` |
| `/etc/hixlep` | endpoint directory the URMA provider reads |

## The three build traps

These cost the most time, so they are handled explicitly rather than left to
whoever hits them next.

**1. `COMPILE_CUSTOM_KERNELS=0` produces a wheel with no extension.**

Building vllm-ascend that way skips CMake and yields a `py3-none-any` wheel.
The `vllm_ascend_C.so` extension carries the `_C_ascend` operators, so without
it the model aborts on `AttributeError: npu_hc_pre_v2` before reaching any
operator. `03_build_stack.sh` grafts the prebuilt `.so` into the wheel instead
of paying for a full rebuild.

**2. `setuptools-scm` cannot parse the branch tag.**

The repository tags branches as `ascend950-umdk-dsv4-flash-893f765`, which the
version regex rejects. Every build pins the version with
`SETUPTOOLS_SCM_PRETEND_VERSION`.

**3. openEuler SP4 has no `gcc-toolset-14` metapackage.**

Only the split packages exist: `gcc-toolset-14-gcc`, `-gcc-c++`, `-binutils`,
`-libstdc++-devel`, `-libgcc`. DeepEP needs GCC 14 for C++20 `<format>`. Do not
override `_GLIBCXX_USE_CXX11_ABI`: the torch wheel is ABI=1 and overriding it
produces undefined-symbol failures at import.

## The version gate

The plugin refuses to start unless the two framework versions match exactly:

```text
vllm        0.32.0+empty.g5df63a7   at 5df63a72
vllm-ascend 0.32.0.dev0+gbd42eea    at bd42eea
```

For an **editable** vllm-ascend install the gate additionally requires a real
git checkout whose `HEAD` is that commit with no tracked changes. For a
**wheel** install it requires the digest of the pinned build in
`AFD_UMDK_VLLM_ASCEND_WHEEL_SHA256`.

`03_build_stack.sh` writes that digest to `artifacts/stack-env.sh`; the test
scripts source it. If the gate fails with *Cannot verify ... source*, that file
is missing or the wheel was reinstalled from a different build.

## Obtaining the sources

```bash
# afd-plugin: this repository, at the branch under test
git clone https://github.com/GuangyuZhu04/afd-plugin.git src/afd-plugin
git -C src/afd-plugin checkout codex/ascend950-umdk-dsv4-flash

# vllm-ascend: pinned commit, then keep its prebuilt extension
git clone https://github.com/vllm-project/vllm-ascend.git src/vllm-ascend-pin
git -C src/vllm-ascend-pin checkout bd42eea01d8312c4df469fabaed45f77a12df4d3

# DeepEP-Ascend: the AFD fork, with its DeepJIT submodule
git clone https://github.com/puxiaokun3721/DeepEP-Ascend.git src/deepep
git -C src/deepep checkout 25b5153eb2549107c13c8d462e325ac28cea8dea
git -C src/deepep submodule update --init --recursive
```

The prebuilt `vllm_ascend_C*.so` comes from the image itself; it is easier to
copy it out of the image than to rebuild CMake against CANN.

## Verifying the result

```bash
bash setup/04_preflight.sh
```

Expect `PREFLIGHT_OK` plus the four operator arities `23/28/22/24`. If the
arities differ, the installed wheel does not match the source tree.

## When something fails

| Symptom | Cause |
| --- | --- |
| `dnf` hangs on a lock | container started without `--init`, so zombies hold the lock |
| `AttributeError: npu_hc_pre_v2` | wheel built without the extension; see trap 1 |
| `Cannot verify UMDK MXFP4 ... source` | wheel digest not set; source `artifacts/stack-env.sh` |
| `HCCL function error create_config` | dies busy; wait for the tenant to finish |
| `507035` inside the operator | a device or URMA library mount is missing |
| image not found | the image lives on the second daemon; set `AFD_DOCKER_HOST` |

Every script reads `AFD_WORK`, `AFD_IMAGE`, `AFD_DOCKER_HOST`, `AFD_URMA_ROOT`,
`AFD_MODEL` and `AFD_PROXY` from the environment, and discovers the HCCL
address from the default route. On a different host, set those instead of
editing the scripts.
