# DeepSeek-V4-Flash end-to-end pass

The AFD asynchronous transport now serves the model. Two consecutive runs of
`test/11_model_e2e.sh` completed with the acceptance case reporting success and
ten correct answers.

## Result

Both runs, same host, same build, back to back:

| Run | execution | cleanup | acceptance_passed | answers | requests |
| --- | --- | --- | --- | --- | --- |
| `run1` (model-e2e-1128175346) | passed | passed | true | 10/10 | 10 |
| `run2` (model-e2e-1128180326) | passed | passed | true | 10/10 | 10 |

```text
$ grep -a "Concurrent completions" run1/e2e.log
Concurrent completions: 10/10 passed
```

The case checks each answer, not only that the request was served
(`answer_checked: true`). All ten are arithmetic whose answers are known:

```text
Request 1: Compute 12 + 7. Reply with just the number. -> 19
Request 2: Compute 13 + 7. Reply with just the number. -> 20
...
Request 10: Compute 21 + 7. Reply with just the number. -> 28
```

Both roles came up and every UMDK buffer registered:

```json
"umdk": {"attention": [0, 1, 2, 3], "ffn": [4, 5, 6, 7], "mxfp4": [4, 5, 6, 7]}
```

Four Attention ranks in DP2/TP2 and four FFN ranks in EP4/TP1, eight dies total.

## What was blocking, and what cleared it

| Failure | Status now |
| --- | --- |
| `507035` MTE invalid GM (rel PC 0x1b20) | fixed upstream by `f4b1f20`; 0 occurrences |
| `metadata has incorrect shape (1024) ... must be 2048` | fixed by the SAS metadata buffer patch; 0 occurrences |
| `Thread export failed` / `HCCL_E_RUNTIME` (ret 15) | 0 occurrences in either run |

The last one is worth stating precisely, because earlier notes treated it as a
possible defect. It is not reproducible here: it appeared in three runs taken
while another tenant was starting or stopping on the shared host, and in these
two clean runs it does not appear at all. Treat it as environment noise from
contention, not as a defect in the transport.

Its traceback is misleading in a way that cost time, so the mapping is recorded
in `../ascend950_umdk_environment_and_tests/docs/tests.md`: the exception is
raised at a `torch.tensor` copy, which is only the sync point that surfaced a
failure in an earlier asynchronous `HcclAllreduce`.

The metadata patch is in
[`../ascend950_umdk_dsa_sas_metadata_fix_20261128/`](../ascend950_umdk_dsa_sas_metadata_fix_20261128/README.md).
`DSA_METADATA_BUFFER_SIZE` stays at 1024; the SparseAttentionSharedKV metadata
needs its own 2048-byte buffer, which is what the patch adds.

## Evidence

```text
run1/  run2/
  run.json             verdict, deployment, phase results
  completions.json     the ten requests and their answers
  umdk-runtime.json    pinned versions, kernel digests, vendor search path
  e2e.log              both role logs and the launcher output
```

`e2e.log` is kept whole rather than excerpted: the failure modes this project
has chased were visible only as one line in eight megabytes of output.

## Reproducing

```bash
cd ../ascend950_umdk_environment_and_tests
export AFD_IMAGE=vllm-ascend-a5:032-megamoe-20261125
export AFD_WORK=/path/to/workspace
bash setup/00_setup_all.sh
bash setup/04_preflight.sh          # expect PREFLIGHT_OK
bash test/10_roundtrip.sh dsv4-flash    # expect PASS
bash test/11_model_e2e.sh               # expect acceptance_passed=True
```

The roundtrip first is deliberate: it takes two minutes and exercises the same
transport, so it distinguishes a transport fault from a model-path fault before
the four-minute model case runs.

## Scope of this claim

This shows the asynchronous transport integrates correctly and that the model
serves correct answers for these ten requests. It does not measure throughput,
long-context behaviour, or accuracy beyond ten short arithmetic prompts. The
case is a smoke test by design.
