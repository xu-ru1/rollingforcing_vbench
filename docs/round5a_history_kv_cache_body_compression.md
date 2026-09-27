# Round 5A: History KV Cache-Body Compression Prototype

## Scope

Round 5A implements a conservative inference-only prototype for history KV cache-body compression. The implementation uses a `logical_read_path` mode: it reads the working/history range from `kv_cache`, selects a uniform/linspace subset of history tokens, and lets the following self-attention see `sink + compressed_history + current`.

The current `kv_cache` is a fixed preallocated tensor per layer:

```text
k: [batch_size, 1560 * 24, 12, 128]
v: [batch_size, 1560 * 24, 12, 128]
```

Because the buffers are preallocated, Round 5A may reduce the visible history length used by later attention, but it is not expected to reduce PyTorch peak allocated/reserved memory immediately. Real memory reduction likely needs dynamic cache allocation or smaller cache buffers in a later round.

## What Round 5A Does

- Adds a separate `cache_body_compress_*` config family, default off.
- Applies only when explicitly enabled with `cache_body_compress_enabled: true` and `cache_body_compress_real_enabled: true`.
- Uses uniform/linspace token selection only.
- Protects the sink/anchor block.
- Protects current denoising KV.
- Defaults to denoise `anchor_working_current` only.
- Records `[FlowCache][cache_body_compression]`, `[FlowCache][cache_body_verify]`, and `[FlowCache][cache_body_summary]`.
- Uses fallback to the original uncompressed cache read path on any error.

## What Round 5A Does Not Do

- No importance scoring.
- No redundancy scoring.
- No output reuse.
- No L1rel threshold.
- No `current_only` compression.
- No default `clean_cache_update` compression.
- No attention sink compression.
- No current denoising KV compression.
- No training logic change.
- No checkpoint change.
- No cache eviction change.
- No cross-attention cache change.
- No in-place `kv_cache` compaction.

## Why No Scoring

This round is meant to validate the cache-body path itself. Importance or redundancy scoring would mix cache-path correctness with a new token-quality policy, making failures harder to interpret. Uniform/linspace selection is enough to confirm whether reducing visible history tokens is wired correctly.

## Why No Output Reuse Or L1rel

Output reuse and L1rel threshold recomputation are separate FlowCache features. They change execution semantics beyond KV visibility and would obscure whether history KV cache-body compression alone is safe.

## Round 4 Limitation

Round 4 compressed `input_key/input_value` after the attention input had already been assembled. It was useful for quality and stability checks, but it did not shrink the `kv_cache` body, did not reduce cache allocation, did not change cache eviction, and therefore did not lower peak memory.

Round 5A moves the new compression decision earlier: after reading `working_cache_key/working_cache_v` from `kv_cache`, before assembling `anchor + working + current` for attention. This is still logical compaction, but it is closer to a cache-body read path than Round 4's post-assembly attention-input compression.

## Mode

Round 5A uses:

```text
mode: logical_read_path
```

It does not use:

```text
mode: in_place_compaction
```

In-place compaction was intentionally deferred because it would need careful updates to `global_end_index`, `local_end_index`, positional assumptions, and eviction behavior.

## Protection Strategy

For denoise `anchor_working_current`:

```text
original visible = protected_sink + history + protected_current
compressed visible = protected_sink + compressed_history + protected_current
```

The sink/anchor block and current KV are kept intact. Only the middle working/history region is selected with uniform/linspace indices.

`current_only` is always recorded as skipped and must have `applied=false`.

`clean_cache_update` is disabled by default because that pass refreshes the clean cache and is more sensitive to cache bookkeeping. It can only be attempted if explicitly enabled, but the Round 5A validation config keeps it off.

## Fallback

Any shape mismatch, invalid keep count, non-finite tensor in debug verification, or attention-path exception falls back to the original uncompressed visible KV. Fallback events are recorded with `fallback=true`, `applied=false`, and `saved_visible_tokens=0`.

## Server Validation Commands

Server path and environment:

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

Syntax check:

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

Create the Round 5A config:

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round5a.yaml

cat >> configs/rolling_forcing_dmd_flowcache_round5a.yaml <<'EOF'

flowcache:
  enabled: true
  debug: false

  metadata_enabled: true
  metadata_output_path: logs/round5a_cache_body.jsonl
  log_kv_ranges: false
  log_summary: true

  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false

  kv_compress_enabled: false
  kv_compress_dry_run: false
  kv_compress_real_enabled: false

  cache_body_compress_enabled: true
  cache_body_compress_real_enabled: true
  cache_body_compress_target_ratio: 0.5
  cache_body_compress_strategy: uniform
  cache_body_compress_min_candidate_tokens: 4680
  cache_body_compress_apply_to_denoise: true
  cache_body_compress_apply_to_clean_cache_update: false
  cache_body_compress_apply_to_current_only: false
  cache_body_compress_protect_sink: true
  cache_body_compress_protect_current: true
  cache_body_compress_max_windows: null
  cache_body_compress_max_layers: null
  cache_body_compress_debug_verify: true

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0

  output_reuse_enabled: false
  l1rel_threshold: 0.0
EOF
```

Disabled baseline 21-frame smoke:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5a_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5a_disabled_smoke.log
```

Round 5A 21-frame smoke:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5a.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5a_cache_body_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5a_cache_body_smoke.log
```

If both 21-frame smoke runs pass, collect the 21-frame logs first. The 81-frame three-prompt command can be run later:

```bash
cat > logs/round5a_three_prompts.txt <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
A futuristic city street at night, neon lights, slow camera pan.
A small dog running through a flower field, bright daylight, smooth motion.
EOF

CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5a.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round5a_three_prompts.txt \
  --output_folder videos/round5a_cache_body_81_three_prompts \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5a_cache_body_81_three_prompts.log
```

## Log Checks

```bash
grep -n "\[FlowCache\]" logs/round5a_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[cache_body_compression\]" logs/round5a_cache_body_smoke.log | head -n 120
grep -n "\[FlowCache\]\[cache_body_verify\]" logs/round5a_cache_body_smoke.log | head -n 120
grep -n "\[FlowCache\]\[cache_body_summary\]" logs/round5a_cache_body_smoke.log
grep -n "\[FlowCache\]\[warning\]" logs/round5a_cache_body_smoke.log | head -n 80
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round5a_*.log

head -n 60 logs/round5a_cache_body.jsonl
wc -l logs/round5a_cache_body.jsonl

ls -lh videos/round5a_disabled_smoke
ls -lh videos/round5a_cache_body_smoke
```

JSONL validation:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("logs/round5a_cache_body.jsonl")
records = []
tensor_like_lines = []

with path.open("r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        text = line.strip()
        if not text:
            continue
        if "tensor(" in text or "device=" in text or "grad_fn=" in text:
            tensor_like_lines.append(line_no)
        records.append(json.loads(text))

events = Counter(record.get("event") for record in records)
body = [record for record in records if record.get("event") == "cache_body_compression"]
summary = [record for record in records if record.get("event") == "cache_body_summary"]
applied = [record for record in body if record.get("applied")]

applied_by_branch = Counter(record.get("branch") for record in applied)
current_only_applied = sum(1 for record in applied if record.get("branch") == "current_only")
clean_cache_update_applied = sum(1 for record in applied if record.get("branch") == "clean_cache_update")
fallback_count = sum(1 for record in body if record.get("fallback"))
warning_count = summary[-1].get("warning_count") if summary else None

formula_errors = []
for idx, record in enumerate(body, 1):
    original_expected = (
        record["protected_sink_tokens"] +
        record["original_history_tokens"] +
        record["protected_current_tokens"]
    )
    compressed_expected = (
        record["protected_sink_tokens"] +
        record["compressed_history_tokens"] +
        record["protected_current_tokens"]
    )
    if record["original_visible_tokens"] != original_expected:
        formula_errors.append((idx, "original_visible"))
    if record["compressed_visible_tokens"] != compressed_expected:
        formula_errors.append((idx, "compressed_visible"))
    if record["compressed_history_tokens"] > record["original_history_tokens"]:
        formula_errors.append((idx, "history_order"))
    if record["compressed_visible_tokens"] > record["original_visible_tokens"]:
        formula_errors.append((idx, "visible_order"))
    if record["saved_visible_tokens"] < 0:
        formula_errors.append((idx, "negative_saved"))

print("event_counts:", dict(events))
print("cache_body_events:", len(body))
print("applied_by_branch:", dict(applied_by_branch))
print("current_only_applied:", current_only_applied)
print("clean_cache_update_applied:", clean_cache_update_applied)
print("fallback_count:", fallback_count)
print("warning_count:", warning_count)
print("formula_errors:", formula_errors[:20], "total=", len(formula_errors))
print("tensor_like_lines:", tensor_like_lines[:20], "total=", len(tensor_like_lines))

assert body, "no cache_body_compression events"
assert summary, "no cache_body_summary event"
assert current_only_applied == 0
assert clean_cache_update_applied == 0
assert not formula_errors
assert not tensor_like_lines
PY
```

## 21-Frame Success Criteria

1. Disabled smoke generates a video.
2. Cache-body smoke generates a video.
3. Disabled log has no `[FlowCache]`.
4. Cache-body log has `[FlowCache][cache_body_compression]`.
5. Cache-body log has `[FlowCache][cache_body_summary]`.
6. No `Traceback`, `RuntimeError`, or OOM.
7. `warning_count = 0`, or warnings are rare and explained.
8. `fallback_count = 0`, or fallbacks are rare and explained.
9. `current_only applied = 0`.
10. `clean_cache_update applied = 0`.
11. Protected sink token count is unchanged.
12. Protected current token count is unchanged.
13. `compressed_history_tokens <= original_history_tokens`.
14. `compressed_visible_tokens <= original_visible_tokens`.
15. `saved_visible_tokens >= 0`.
16. JSONL is non-empty.
17. JSONL contains no tensor contents.
18. Output videos exist and are non-empty.
19. No importance/redundancy scoring was implemented.
20. No output reuse was implemented.
21. No L1rel was implemented.
22. Training logic was not changed.

## Next Round

If Round 5A is stable and visible KV length drops as expected, the next step is to decide whether to pursue true in-place cache compaction or a dynamic/smaller cache buffer. That round should focus on memory allocation and eviction semantics, not scoring.
