# Round 5B.2: prefix-compatible persistent sidecar reuse

## 本轮做什么

Round 5B.2 修复 Round 5B.1 中 `sidecar_keep_valid_count > 0` 但 `sidecar_reuse_count = 0` 的问题。

5B.1 已经证明普通不重叠 cache write 不会破坏 sidecar，但下一次 denoise attention 的 history range 通常会从旧 range 变成更长 range，例如：

```text
old sidecar source: [4680, 14040)
new history source: [4680, 18720)
```

旧 sidecar 是新 history 的前缀。5B.1 因为要求 `source_end` 完全匹配，把它判成 `source_end_mismatch` 并重新 create。

5B.2 改为：

- 如果旧 sidecar source 是当前 history source 的前缀；
- 且 ratio / strategy / layer / dtype / device / tensor shape 都匹配；
- 且旧 source 没被 cache write 或 eviction 破坏；
- 则复用旧 compressed prefix；
- 只对新增 delta range 做 uniform token selection；
- 拼成覆盖当前完整 history range 的新 sidecar。

这样会记录：

```text
[FlowCache][persistent_cache_sidecar] action=reuse reason=prefix_source_reused
[FlowCache][persistent_cache_sidecar] action=create reason=extended_from_prefix_reuse
```

## 本轮不做什么

本轮仍不做：

- importance / redundancy scoring；
- output reuse；
- L1rel threshold；
- current_only 压缩；
- 默认 clean_cache_update 压缩；
- sink 压缩；
- current denoising KV 压缩；
- in-place 修改 `kv_cache["k"]` / `kv_cache["v"]`；
- 修改 cache eviction 主逻辑；
- 修改训练逻辑；
- 修改 checkpoint。

## 为什么这个 reuse 是保守的

5B.2 不把旧 sidecar 直接当成新完整 history 使用。它只复用旧 sidecar 覆盖的 prefix，并从新增 delta range 中继续按 uniform selection 取 token。

因此当前 compressed history 的组成是：

```text
compressed_current_history =
  compressed_old_prefix + uniformly_selected_new_delta
```

这仍然是 uniform / segment-uniform selection，没有引入 scoring，也没有复用 attention output。

## 为什么不要求 peak memory 下降

当前 `kv_cache["k"]` / `kv_cache["v"]` 仍是固定预分配 tensor。5B.2 只证明 persistent sidecar 可以被复用，不改变主 cache buffer 大小。

因此 peak allocated / reserved 不一定下降。真正显存收益仍留到 dynamic compacted KV cache buffer。

## 服务器 21-frame 命令

先进入服务器仓库：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

创建 config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round5b_2.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round5b_2.yaml <<'YAML'

flowcache:
  enabled: true
  debug: false

  metadata_enabled: true
  metadata_output_path: logs/round5b_2_persistent_smoke.jsonl
  log_kv_ranges: false
  log_summary: true

  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false

  kv_compress_enabled: false
  kv_compress_dry_run: false
  kv_compress_real_enabled: false

  cache_body_compress_enabled: false
  cache_body_compress_real_enabled: false

  persistent_cache_compress_enabled: true
  persistent_cache_compress_real_enabled: true
  persistent_cache_compress_target_ratio: 0.5
  persistent_cache_compress_strategy: uniform
  persistent_cache_compress_min_candidate_tokens: 4680
  persistent_cache_compress_apply_to_denoise: true
  persistent_cache_compress_apply_to_clean_cache_update: false
  persistent_cache_compress_apply_to_current_only: false
  persistent_cache_compress_protect_sink: true
  persistent_cache_compress_protect_current: true
  persistent_cache_compress_sidecar_enabled: true
  persistent_cache_compress_debug_verify: true
  persistent_cache_compress_max_windows: null
  persistent_cache_compress_max_layers: null

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0

  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
```

disabled smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_2_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_2_disabled_smoke.log
```

persistent smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5b_2.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_2_persistent_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_2_persistent_smoke.log
```

检查：

```bash
grep -n "\[FlowCache\]" logs/round5b_2_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[persistent_cache_sidecar\]" logs/round5b_2_persistent_smoke.log | head -n 180
grep -n "\[FlowCache\]\[persistent_cache_summary\]" logs/round5b_2_persistent_smoke.log
grep -n "\[FlowCache\]\[warning\]" logs/round5b_2_persistent_smoke.log | head -n 80
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round5b_2_*.log
wc -l logs/round5b_2_persistent_smoke.jsonl
ls -lh videos/round5b_2_disabled_smoke
ls -lh videos/round5b_2_persistent_smoke
```

JSONL 快速检查：

```bash
python - <<'PY'
import json
from collections import Counter

path = "logs/round5b_2_persistent_smoke.jsonl"
events = Counter()
sidecar = Counter()
applied = Counter()
fallback_count = 0
warning_count = 0
current_only_applied = 0
clean_cache_update_applied = 0
formula_errors = 0
tensor_like = 0
parse_errors = 0
summary = None

for line_no, line in enumerate(open(path, encoding="utf-8"), 1):
    try:
        item = json.loads(line)
    except Exception:
        parse_errors += 1
        continue
    if "tensor(" in str(item).lower():
        tensor_like += 1
    event = item.get("event")
    events[event] += 1
    if event == "persistent_cache_sidecar":
        sidecar[item.get("action", "unknown")] += 1
    if event == "persistent_cache_compression":
        branch = item.get("branch")
        is_applied = bool(item.get("applied"))
        if is_applied:
            applied[branch] += 1
        fallback_count += int(bool(item.get("fallback")))
        current_only_applied += int(branch == "current_only" and is_applied)
        clean_cache_update_applied += int(branch == "clean_cache_update" and is_applied)
        expected_original = item.get("protected_sink_tokens", 0) + item.get("original_history_tokens", 0) + item.get("protected_current_tokens", 0)
        expected_compressed = item.get("protected_sink_tokens", 0) + item.get("compressed_history_tokens", 0) + item.get("protected_current_tokens", 0)
        if item.get("original_visible_tokens") != expected_original:
            formula_errors += 1
        if item.get("compressed_visible_tokens") != expected_compressed:
            formula_errors += 1
    if event == "persistent_cache_summary":
        summary = item
        warning_count = item.get("warning_count", warning_count)

print("events", dict(events))
print("sidecar", dict(sidecar))
print("applied", dict(applied))
print("fallback_count", fallback_count)
print("warning_count", warning_count)
print("current_only_applied", current_only_applied)
print("clean_cache_update_applied", clean_cache_update_applied)
print("formula_errors", formula_errors)
print("tensor_like", tensor_like)
print("parse_errors", parse_errors)
if summary:
    keys = [
        "sidecar_create_count",
        "sidecar_reuse_count",
        "sidecar_keep_valid_count",
        "sidecar_invalidate_count",
        "sidecar_fallback_count",
        "fallback_count",
        "warning_count",
        "applied_weighted_visible_saving_ratio",
        "overall_weighted_visible_saving_ratio",
    ]
    print("summary_core", {key: summary.get(key) for key in keys})
PY
```

## 21-frame 成功标准

必须满足：

- disabled smoke 生成视频，且日志无 `[FlowCache]`；
- persistent smoke 生成视频；
- 有 `[FlowCache][persistent_cache_sidecar]`；
- 有 `[FlowCache][persistent_cache_summary]`；
- `sidecar_create_count > 0`；
- `sidecar_reuse_count > 0`；
- `sidecar_fallback_count = 0` 或很少且可解释；
- `fallback_count = 0`；
- `warning_count = 0`；
- `current_only_applied = 0`；
- `clean_cache_update_applied = 0`；
- JSONL 非空，不包含 tensor 内容；
- 无 Traceback / RuntimeError / OOM。

如果 `sidecar_reuse_count > 0` 且日志干净，再考虑 81-frame；否则不要跑 81-frame。

## 需要回传

先只回传 21-frame：

- `logs/round5b_2_disabled_smoke.log`
- `logs/round5b_2_persistent_smoke.log`
- `logs/round5b_2_persistent_smoke.jsonl`
- `configs/rolling_forcing_dmd_flowcache_round5b_2.yaml`
- summary grep 输出
- warning/error grep 输出
- JSONL 快速检查输出
- 两个输出视频目录的 `ls -lh`
