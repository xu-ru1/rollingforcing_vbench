# RollingForcing step-position cache: Git delivery and server runbook

This is the R8-frozen six-method experiment packaged for a different server. The
new launcher calls the validated `inference.py`, cache runtime, FLOP recorder,
480-frame crop, and VBench filename-mapping modules. It does not change the
cache policy, thresholds, scheduler, or inference length.

## Frozen contract

| Item | Value |
| --- | --- |
| Methods | `rf_vanilla`, `rf_fixed_slow`, `rf_front_slow`, `rf_fixed_fast`, `rf_front_fast`, `rf_u_shape_fast` |
| Cache thresholds | `0`, `0.26`, `0.24`, `0.40`, `0.54`, `0.58`, respectively |
| Seed | `0`, reset before each prompt |
| Generation | 126 latent frames, 3 per block, 42 complete blocks |
| Raw video | 501 decoded RGB frames, 832×480, 16 fps |
| VBench video | Exact decoded frames `[0,480)`, losslessly re-encoded, 30 seconds |
| Prompt suite | Official VBench-1 metadata: 946 records, 944 unique generation strings |
| Latency/PFLOPs | Full 126-latent inference, including clean-cache DiT forwards; not inferred from 480-frame videos |

The checked-in six configs and prompt index were copied byte-for-byte from the
successful R8 preflight return bundle. Their SHA256 values are frozen in
`configs/formal_vbench/delivery_protocol.json`. The two duplicate official
prompt strings map to one video each. This is the **fixed single-seed VBench protocol**:
official prompts and 16 dimensions, one video per unique prompt at seed 0,
944 videos per method. It is not the official multi-sample leaderboard protocol.
`.gitattributes` preserves LF line endings for the frozen hash-checked files
when they move through Git.
`inference.py` retains the lab-validated R9 checkpoint-loader cleanup hash.
All 15 delivery source files now have frozen SHA256 entries in the protocol and
`check` rejects any mismatch. Ten core files match R7 immutable source
hashes. `utils/wan_wrapper.py` is explicitly marked as a delivery-only
`WAN_MODEL_ROOT` path-resolution amendment; the crop script has a delivery-only
artifact-safety amendment that leaves the `[0,480)` interval unchanged.
The commit/materialize scripts have delivery-frozen hashes because the available
R7 source manifest did not list them; this provenance distinction is recorded
per file in the protocol.

## Repository and environment

The local source directory used to prepare this delivery has no `.git` metadata.
Place these files in the intended Git repository or initialize a new repository
before committing. Do not copy `outputs/`, `debug-gpt/`, checkpoints, Wan weights,
or generated videos into Git. The `.gitignore` excludes these; it retains the
configs, prompt index/list, scripts, source, and reference metadata.

On the destination server, replace only the bracketed paths:

```bash
git clone <GIT_URL> <REPO_DIR>
cd <REPO_DIR>

# First select an existing Python environment compatible with the lab CUDA stack.
python --version
python -m pip --version
nvidia-smi
python -c 'import torch; print("torch",torch.__version__,"CUDA",torch.version.cuda)'
python -c 'import flash_attn; print("flash-attn",flash_attn.__version__)'
# Missing imports may be recorded here; install only missing inference dependencies
# after reviewing the selected environment. Do not run the full requirements.txt
# as an initial step or upgrade torch/CUDA/flash-attn/diffusers/transformers.

export CHECKPOINT="<CHECKPOINT_DIR>/rolling_forcing_dmd.pt"
export WAN_MODEL_ROOT="<MODEL_DIR>/Wan2.1-T2V-1.3B"
export OUTPUT_ROOT="<OUTPUT_DIR>/rf_step_cache_delivery"
export GPU_ID=<GPU_ID>

python scripts/run_step_cache_delivery.py check
bash scripts/record_step_cache_delivery_env.sh "$OUTPUT_ROOT/environment"
git rev-parse HEAD
```

Record the selected Python path, NVIDIA driver, CUDA runtime, torch and
flash-attn versions before installing anything. A missing optional import is
not a reason to reinstall the whole environment: `requirements.txt` includes
TensorRT, PyCUDA and ONNX tooling outside the core inference path. Keep it as a
reference and install only confirmed missing packages compatible with the lab
stack. Then run the static check and environment recorder shown above.

`WAN_MODEL_ROOT` is the directory containing `Wan2.1_VAE.pth`,
`models_t5_umt5-xxl-enc-bf16.pth`, and `google/umt5-xxl/`. It also contains the
Wan diffusion model files read by `from_pretrained`. The checkpoint path is the
EMA RollingForcing checkpoint. The launcher checks these paths before GPU use.
Use `--verify-checkpoint` once before formal generation to compare its SHA256
with the lab R8 freeze; this reads the entire checkpoint and may take time.

The lab environment's full `pip freeze`, VBench commit, CUDA/driver version, and
GPU compatibility were not included in this local source tree. Capture them on
the lab server with `record_step_cache_delivery_env.sh` and reproduce or review
them on the destination. Do not upgrade torch, CUDA, diffusers, transformers,
flash-attn, xformers, or VBench simply to satisfy an installer.

## One-prompt migration smoke

Run on one idle GPU. It uses the **frozen `rf_front_fast` config** but asks the
CLI for 12 latent frames to keep the migration probe small. Formal generation
remains 126 latent frames.

```bash
python scripts/run_step_cache_delivery.py smoke \
  --method rf_front_fast --checkpoint "$CHECKPOINT" \
  --wan-model-root "$WAN_MODEL_ROOT" --output-root "$OUTPUT_ROOT" \
  --gpu "$GPU_ID" --verify-checkpoint
```

Expected: `[Inference] Loaded checkpoint weights: generator_ema`, then
`[delivery] SMOKE_OK`. The output is
`$OUTPUT_ROOT/smoke/rf_front_fast/videos/0-0_ema.mp4`, with 45 decoded frames.
`video_check.json`, `audit_hashes.jsonl`, `runtime_summary.json`,
`run_manifest.json`, `inference.log`, and `status.json` accompany it. A missing
`omegaconf`/`flash_attn`/`torch` import is an environment issue; `checkpoint
missing` or `Wan ... missing` is an asset/path issue; a stack trace from
`step_cache_runtime.py`, `step_cache_sparse.py`, or `causal_model.py` after model
load requires checking source/config parity. A failed decode or missing MP4 is
reported by `video_check.log`.

An alternate one-prompt file can be passed with `--prompt-file <PROMPT_FILE>`;
it must contain exactly one nonempty line. To smoke Vanilla separately, use a
different output root and `--method rf_vanilla`.

## Formal-length cache migration gate

After the unchanged 12-latent smoke, run a separate frozen-policy gate on one
GPU. It uses R7's first formal-smoke prompt (`a cat sitting on the grass`),
`rf_front_fast`, seed 0, 126 latent, 3 latent/block and 42 complete blocks.
The R7 per-sample DiT FLOPs for this prompt were lower than Vanilla and the
three-prompt runtime summary recorded reuse. The destination run must still
prove reuse itself; no threshold is adjusted if it does not.

```bash
python scripts/run_step_cache_delivery.py migration_gate \
  --method rf_front_fast --checkpoint "$CHECKPOINT" \
  --wan-model-root "$WAN_MODEL_ROOT" --output-root "$OUTPUT_ROOT" \
  --gpu "$GPU_ID" --verify-checkpoint
```

Success prints `[delivery] MIGRATION_GATE_OK` only after the MP4 decodes to
501 frames and `runtime_summary.json` reports sparse execution with
`operator_totals.reuse_block_events > 0`. The audit log, manifest, runtime
summary, video check, and `status.json` remain under
`$OUTPUT_ROOT/migration_gate/rf_front_fast/`. Until this GPU command succeeds
on the destination server, migration-gate status is **NEED_GPU_VERIFY**.

## Formal video generation

Run one method/shard per GPU session. The launcher preserves the R8 prompt
order and frozen configs. One successful shard is skipped on repeat; an
incomplete shard is left intact for inspection. Failure writes `failure.json`
with the first incomplete global prompt index. A shard creates no `status: ok`
until raw commit, 501→480 pixel-exact crop, and VBench-standard filename
materialization all succeed.

```bash
python scripts/run_step_cache_delivery.py generate \
  --method rf_vanilla --start 0 --count 128 \
  --checkpoint "$CHECKPOINT" --wan-model-root "$WAN_MODEL_ROOT" \
  --output-root "$OUTPUT_ROOT" --gpu "$GPU_ID"
```

The eight shard ranges are `0:128`, `128:128`, `256:128`, `384:128`,
`512:128`, `640:128`, `768:128`, `896:48`. Repeat each range for the six
methods. For example, after a successful smoke:

```bash
for METHOD in rf_vanilla rf_fixed_slow rf_front_slow rf_fixed_fast rf_front_fast rf_u_shape_fast; do
  for START in 0 128 256 384 512 640 768 896; do
    COUNT=128
    [ "$START" -eq 896 ] && COUNT=48
    python scripts/run_step_cache_delivery.py generate \
      --method "$METHOD" --start "$START" --count "$COUNT" \
      --checkpoint "$CHECKPOINT" --wan-model-root "$WAN_MODEL_ROOT" \
      --output-root "$OUTPUT_ROOT" --gpu "$GPU_ID" || exit 1
  done
done
```

The serial loop avoids concurrent model loads and account-memory spikes. Each
method should finish with 944 files under
`$OUTPUT_ROOT/videos/<method>/vbench30_standard/`. The `raw/` and
`vbench30_indexed/` directories retain traceable intermediate videos; shard
reports retain the prompt-to-video mapping and pixel hashes. The generation
process does not provide the paper latency. It disables the FLOP hook so
generation, latency, and PFLOPs stay separate.

```bash
python scripts/summarize_step_cache_delivery.py --output-root "$OUTPUT_ROOT"
```

Inspect `$OUTPUT_ROOT/reports/delivery_summary.json` before VBench scoring.
`formal_generation_complete` must be `true`.

## VBench-1 scoring

Provide a separate official VBench checkout and its own working Python if
needed. The checkout's `vbench/VBench_full_info.json` must have the frozen
SHA256. Score one method at a time, after all 944 videos for that method exist:

```bash
python scripts/run_step_cache_delivery_score.py \
  --method rf_vanilla --output-root "$OUTPUT_ROOT" \
  --vbench-root <VBENCH_DIR> --vbench-python <VBENCH_PYTHON> --gpu "$GPU_ID"
```

The scoring entry invokes the official `evaluate.py` on 15 dimensions, the
official `static_filter.py` for temporal flickering, then official `evaluate.py`
for the final dimension. It records the static-filter retained count and raw
logs/results under `$OUTPUT_ROOT/vbench_results/<method>/`. Official VBench
requires its own pretrained metric assets. The exact VBench checkout commit
and those assets are **NEED_SERVER_CHECK**; no VBench source or weights are
vendored here. The R8 pre-VBench manifest explicitly had no checkout commit,
so the frozen protocol says **NEED_SERVER_CHECK** rather than inventing one.
At score time the actual VBench git commit, metadata SHA256 and selected Python
executable/version/package environment are written to the score manifest,
`vbench_environment.json`, and `vbench_pip_freeze.txt`. If a lab commit is recovered and entered into the
frozen protocol, scoring verifies equality before evaluation. The scorer
refuses incomplete or extra input videos and will
not overwrite an existing score directory. The [VBench README](https://github.com/Vchitect/VBench/blob/master/README.md)
documents the standard mode and temporal-flickering filter.

## Separate latency and DiT PFLOPs runs

Measure latency on one relatively idle GPU. The default ten-line R6D prompt
file contains one warmup followed by nine timed prompts. `--profile` reports
CUDA-timed full rolling diffusion including clean-cache updates. The report
discards the warmup and retains all nine raw timings and CV. FLOP hooks are
disabled during this run.

```bash
python scripts/run_step_cache_delivery.py latency \
  --method rf_front_fast --checkpoint "$CHECKPOINT" \
  --wan-model-root "$WAN_MODEL_ROOT" --output-root "$OUTPUT_ROOT" \
  --gpu "$GPU_ID"
```

For PFLOPs, the default three calibration prompts run at 126 latent frames.
The DiT recorder counts both `main_denoise` and `clean_cache_update`. The raw
`flops.json` contains operation totals, branches, and per-prompt counts; the
smaller `pflops_report.json` is only a convenience summary.

```bash
python scripts/run_step_cache_delivery.py pflops \
  --method rf_front_fast --checkpoint "$CHECKPOINT" \
  --wan-model-root "$WAN_MODEL_ROOT" --output-root "$OUTPUT_ROOT" \
  --gpu "$GPU_ID"
```

Repeat both commands with each method and the same prompt files. For repeated
independent timing sessions, use a new `OUTPUT_ROOT`; existing run directories
are never overwritten. `--prompt-file <PROMPT_FILE>` is available for smoke,
latency, and PFLOPs; formal VBench generation always uses the frozen official
prompt index. `--config <CONFIG>` accepts a relocated byte-identical frozen
config, but rejects changed content.

## Output and return bundle

```text
<OUTPUT_ROOT>/
  environment/                         pip freeze, GPU/CUDA, ffmpeg, Git
  freeze/                              copied immutable protocol, configs, prompt index
  smoke/<method>/                      MP4, decode check, audit, runtime, logs
  migration_gate/rf_front_fast/         126-latent MP4, reuse proof, decode check
  shards/<method>/<start>_<count>/     prompts, index, manifest, audit, logs, raw/crop/materialize reports, status
  videos/<method>/raw/                 501-frame outputs
  videos/<method>/vbench30_indexed/    exact 480-frame indexed outputs
  videos/<method>/vbench30_standard/   exact 480-frame official names
  latency/<method>/                    inference.log and latency_report.json
  pflops/<method>/                     flops.json and pflops_report.json
  vbench_results/<method>/             15-dimension result, static filter, temporal result, logs
  reports/delivery_summary.json        six-method coverage/performance index
```

After smoke, return the small smoke directory including its MP4:

```bash
tar -czf <SMOKE_RETURN_TAR> -C "$OUTPUT_ROOT" smoke environment
```

After generation/scoring, return reports and **raw** logs (especially
`pflops/<method>/flops.json`), not just table numbers:

```bash
tar --exclude='*.mp4' -czf <REPORT_RETURN_TAR> -C "$OUTPUT_ROOT" \
  freeze shards latency pflops vbench_results reports environment
```

Keep the 480-frame videos on the scoring server. If they need transfer, send
`videos/<method>/vbench30_standard/` separately; it is large. Also report the
Git commit hash and the VBench checkout commit. Never treat batch-generation
wall time as the clean-GPU latency result.
