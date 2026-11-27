# Ascend 950 arch35 DispatchSend 507035: bisection and eliminated causes

This note records a systematic attempt to find the cause of the `507035`
device error that
[the roundtrip result](ascend950_umdk_arch35_roundtrip_result.md) records.
**It does not contain a fix.** It records what was measured, which hypotheses
the measurements eliminated, and the one observation still missing.

The sender traps on its first launch in every configuration tried, in both the
arch35 runtime kernel and the original `impls/afd/dispatch_send.h` body, which
share `bind_directory()` and the layout headers. The receiver, which validates
its directory before dereferencing it, does not trap.

## Recorded failure

```text
RuntimeError: UMDK DispatchSend synchronization failed on rank 1, layer 2,
              rounds=1: ... device error type 3, error code is 507035
aivec error, core id is 0, error code = 95
pc start: 0x120040819000, current: 0x12004081ab20
errcode:(95) The MTE instruction accesses an invalid GM address or the
             cross-device memory access timeout. subErrType: 0x4
```

Disassembling the dumped kernel object (not stripped, so the object the device
faulted on can be read directly) puts the trap at relative `0x1b20`:

```text
1b18: LD_IO.b32  X3, [X0], #128, #0
1b1c: LD_IO.b64  X2, [X30], #-1816, #0
1b20: LD_IO.b32  X4, [X0], #168, #0      <- trap
1b24: LD_IO.b64  X6, [X0], #304, #0
```

The third operand is a pointer-increment stride, not a memory offset:
neighbouring code reads `LD_IO.b64 X2, [X4], #24` and then stores through
`X2`. So the illegal address is the value in `X0`; `#168` only advances the
register afterwards. The recurring `168` across builds comes from the same
constant pool entry, not from the same faulting field.

## Bisection

Each row is a single-variable change; every row still traps.

| # | Change | Result |
| --- | --- | --- |
| 1 | dispatch only, combine never called | traps, so the fault is in DispatchSend |
| 2 | reference pipeline without `--local-comm-rounds` | traps, so local rounds are not required |
| 3 | reference pipeline without `--ffn-transaction` | traps, so the transaction mode is not required |
| 4 | `--batch 1 --window 1` | traps, so payload volume is not a factor |
| 5 | `_AIV_CORES = 1` | traps, so launch width is not a factor |
| 6 | fresh container, same wheel | traps, so container state is not a factor |

The DeepEP reference pipeline traps as well, at relative `0x1b20` from its own
kernel entry. Two schedulers, two geometries and two call paths that share only
the operator binary fail on the same instruction, which places the defect in the
operator rather than in the plugin's session setup.

## What was measured and eliminated

### Resident geometry is correct

The exception dump records the 28 argument slots. Decoding them against the
operator schema gives the values the launch actually used:

```text
num_ranks = 4   num_experts = 256   num_topk = 6
num_vec_cores = 32   num_tokens_per_tile = 256
timeout_cycles = 120000000000
```

All match `session.cfg` printed by the session itself, and the kernel object in
the dump is byte-identical to the installed one (sha256 `2290fc09…`), so the
launch did not use a stale or mismatched binary.

### The context blob the device read is well formed

The dump's `exception_info` is text metadata, but the context blob can be
recovered from the same capture and decoded field by field. For the Flash
geometry (`num_source_lanes=4`, `num_ranks=4`, `num_layers=43`):

```text
abi_version=2   num_ranks=4   num_src_ranks=4   num_source_lanes=4
num_experts=256 num_topk=6    num_hidden_bytes=8192   num_sf_packs=0
num_vec_cores=32  num_token_capacity=4096  num_tokens_per_tile=256
num_layers=43   src_rank_idx=0  num_slots=1
return_progress_mode=0  channel_stride=32  comm_round_mode=1

lanes_offset     @72  = 256      (kernel requires sizeof(AFDContextLayout)=256)
peers_offset     @80  = 384      (= 256 + 32*4)
addresses_offset @88  = 512      (= 384 + 32*4)
layers_offset    @96  = 1536     (= 512 + 64*16)
context_bytes    @104 = 2016     (equals the context tensor size)
payload_bytes    @112 = 192 MiB
control_bytes    @120 = 1024
```

Compiling the kernel's own `AFDContextLayout` and printing `offsetof` for each
field reproduces these offsets exactly, and the DeepEP host header that packs the
blob is byte-identical to the kernel header after normalising the `alignas`
argument. The declared `num_bytes = 625472` also equals the `afd_workspace`
tensor size, so the two sides agree end to end.

### Directory indexing is in range

The blob's peer directory decodes as:

```text
peer[0] domain_rank=4 peer_idx=96   control_bytes=512
peer[1] domain_rank=5 peer_idx=128  control_bytes=512
peer[2] domain_rank=6 peer_idx=160  control_bytes=512
peer[3] domain_rank=7 peer_idx=192  control_bytes=512
```

The host computes `peer_idx = (r < rank_idx ? r : r - 1) * channel_stride` with
`channel_stride = 32`, and the kernel indexes `peer_idx + vec_core_idx`. The
largest index is `192 + 31 = 223`. The device jetty table holds 224 entries
(`7 non-self ranks * 32`), so the last valid slot is reachable and nothing is
out of range. `addresses[l * num_ranks + f]` is packed and indexed with the
same expression on both sides, and `control_bytes = 512` covers the largest
control offset the kernel writes.

### Buffer sizes vary by geometry, so fixed-offset probes are unsafe

| Profile | context tensor | afd_workspace |
| --- | ---: | ---: |
| `small` | 832 B | 3008 B |
| `dsv4-flash` | 2016 B | 625472 B |

The same kernel binary runs with blobs two orders of magnitude apart. Any probe
that writes a diagnostic at a fixed high workspace offset is an out-of-bounds
write in the smaller profile. An earlier probe of this investigation did exactly
that before the sizes were measured.

## Asymmetry between send and recv

```text
afd_async_dispatch_recv_rt.h : 8 recv_range() checks plus a full offset-chain check
afd_async_dispatch_send_rt.h : none
```

The receiver's `check_context()` verifies that `lanes_offset` is
`sizeof(AFDContextLayout)`, that each following offset equals the previous one
plus its array size, and that every segment lies inside `context_bytes`. It then
rejects a mismatched layout with `BAD_ABI` before forming any pointer. The sender
has no equivalent, so a layout it does not expect becomes a pointer instead of an
error code.

Adding the equivalent `check_directory()` to the sender compiles and rejects
bad layouts early, but **it does not fix this fault**: with the check installed
the kernel still traps `507035`, now at relative `0x608`. The change was
therefore reverted rather than presented as a fix.

## The observation still missing

The trap reads through `X0`, so the value of `X0` at trap time identifies the
offending argument. Two attempts to capture it failed:

1. Writing the eight `GM_ADDR` arguments into a fixed workspace offset was
   compiled and launched. The trap moved to relative `0xaac4`, earlier than the
   original, and the device dump stores metadata only, not tensor payload, so the
   values could not be read back.
2. Reading the values from the existing dump is not possible for the same reason.

Getting this needs either a register read at the trap (`msdebug`/`ascendgdb`)
or a `-g` kernel build whose `llvm-dwarfdump --debug-line` table maps
`0x1b20` straight to a source line.

## Status

No fix. `dispatch_send` and the plugin source tree are unchanged; the installed
kernel is the original `2290fc09…`. The list below is what a future attempt does
not need to re-measure:

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| host/kernel struct mismatch | compiled `offsetof` matches the dumped blob | eliminated |
| 32/64-bit field misalignment | offsets sit at `@72`, u64-aligned as the kernel reads them | eliminated |
| context tensor too small | needs 1708, holds 2016 | eliminated |
| `context+168` out of range | 168 is inside a 2016-byte tensor | eliminated |
| `peer_idx` out of range | 96/128/160/192, max index 223 of 224 | eliminated |
| jetty table undersized | 224 = 7 * 32 covers the largest index | eliminated |
| address-array index order | identical expression on both sides | eliminated |
| `control_bytes` too small | 512 against a 480-byte maximum write | eliminated |
| JIT and CAM layout drift | header diff is identical | eliminated |
| AIV launch width | `_AIV_CORES = 1` still traps | eliminated |
| batch or window size | `--batch 1 --window 1` still traps | eliminated |
| local rounds or transaction mode | removing either still traps | eliminated |
| stale kernel binary | dump object sha256 matches the installed one | eliminated |
