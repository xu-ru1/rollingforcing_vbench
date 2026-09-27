# Round 5B.1: persistent sidecar reuse fix

## 1. Round 5B.1 做什么

Round 5B.1 只修复 persistent history KV sidecar 的复用问题。

本轮保留 Round 5B 的 sidecar 方案，不引入新算法。核心变化是收窄 sidecar invalidation 条件：普通 current KV 写入如果不覆盖 sidecar 的 `source_start/source_end`，不再立即 invalidate sidecar。

目标是在 21-frame smoke 中看到：

- `sidecar_create_count > 0`
- `sidecar_reuse_count > 0`
- `sidecar_keep_valid_count > 0` 代表普通写入没有破坏 sidecar source range
- `fallback_count = 0` 或很少且可解释
- `warning_count = 0`
- `current_only_applied = 0`
- `clean_cache_update_applied = 0`

## 2. Round 5B.1 不做什么

本轮不做：

- importance / redundancy scoring；
- output reuse；
- L1rel threshold；
- current_only 压缩；
- 默认 clean_cache_update 压缩；
- sink 压缩；
- current denoising KV 压缩；
- in-place 修改 `kv_cache["k"]` / `kv_cache["v"]` 主 tensor；
- 修改 cache eviction 主逻辑；
- 修改训练逻辑；
- 修改 checkpoint。

## 3. Round 5B 为什么 reuse=0

Round 5B 的 21-frame smoke 中：

```text
sidecar_create_count=120
sidecar_reuse_count=0
sidecar_invalidate_count=120
```

原因有两个：

1. 每次主 cache 写入后都会无条件 invalidate 旧 sidecar。
2. sidecar validity 要求 `global_end_index/local_end_index` 完全等于创建时的值，普通 append 也会让 sidecar 被判为失效。

这很安全，但过于保守，导致 sidecar 虽然创建了，却无法在后续 attention 中复用。

## 4. 本轮如何收窄 invalidation

本轮新增基于 range 的 invalidation 判断：

- sidecar 记录 `source_start/source_end`；
- cache 写入记录 `write_start/write_end`；
- 如果写入 range 与 sidecar source range 不重叠，则保留 sidecar valid；
- 如果写入 range 与 sidecar source range 重叠，则 invalidate；
- 如果发生 eviction，则 invalidate；
- sidecar validity 不再把 `global_end_index/local_end_index` append 视为自动失效条件。

保留 sidecar 时会记录：

```text
[FlowCache][persistent_cache_sidecar] action=keep_valid
```

## 5. 什么情况下允许 reuse

允许 reuse 的条件：

- sidecar valid flag 为 true；
- layer_idx 匹配；
- 当前 history range 的 `source_start/source_end` 与 sidecar metadata 匹配；
- target_ratio 匹配；
- strategy 匹配；
- compressed token 数与计划一致；
- sidecar key/value tensor 存在；
- key/value shape 合法；
- key/value dtype/device 与当前 KV 匹配；
- debug verify 开启时没有 NaN/Inf；
- 自 sidecar 创建后，没有写入或 eviction 破坏该 source range。

## 6. 什么情况下必须 invalidate

必须 invalidate 的情况：

- eviction 发生；
- cache write range 与 sidecar source range overlap；
- source range metadata 缺失或非法；
- sidecar metadata 与当前读取条件不匹配；
- ratio/strategy 变化；
- dtype/device 不匹配；
- sidecar tensor shape 不合法；
- debug verify 发现 NaN/Inf；
- 显式 fallback 发生。

## 7. 为什么不要求 peak memory 立刻下降

当前 RollingForcing 的 `kv_cache["k"]` / `kv_cache["v"]` 是固定预分配 tensor。Round 5B.1 仍然不缩小主 cache buffer，也不做 dynamic cache allocation。

因此即使 sidecar 能复用：

- peak allocated / reserved 也不一定下降；
- sidecar 自身可能带来少量额外 tensor；
- 本轮重点是证明 persistent compressed-history representation 可以被后续 attention 复用。

真正的显存收益要留到后续 dynamic compacted KV cache buffer。

## 8. 服务器实验命令

进入服务器仓库：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

### 8.1 py_compile

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

### 8.2 创建 21-frame Round 5B.1 config

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round5b_1.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round5b_1.yaml <<'YAML'

flowcache:
  enabled: true
  debug: false

  metadata_enabled: true
  metadata_output_path: logs/round5b_1_persistent_smoke.jsonl
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

### 8.3 21-frame disabled smoke

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_1_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_1_disabled_smoke.log
```

### 8.4 21-frame persistent smoke

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5b_1.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_1_persistent_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_1_persistent_smoke.log
```

### 8.5 21-frame 检查

```bash
grep -n "\[FlowCache\]" logs/round5b_1_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[persistent_cache_compression\]" logs/round5b_1_persistent_smoke.log | head -n 120
grep -n "\[FlowCache\]\[persistent_cache_sidecar\]" logs/round5b_1_persistent_smoke.log | head -n 160
grep -n "\[FlowCache\]\[persistent_cache_summary\]" logs/round5b_1_persistent_smoke.log
grep -n "\[FlowCache\]\[warning\]" logs/round5b_1_persistent_smoke.log | head -n 80
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round5b_1_*.log
wc -l logs/round5b_1_persistent_smoke.jsonl
ls -lh videos/round5b_1_disabled_smoke
ls -lh videos/round5b_1_persistent_smoke
```

JSONL 检查：

```bash
python - <<'PY'
import json
from collections import Counter

path = "logs/round5b_1_persistent_smoke.jsonl"
event_counts = Counter()
applied_by_branch = Counter()
sidecar_counts = Counter()
fallback_count = 0
warning_count = 0
current_only_applied = 0
clean_cache_update_applied = 0
formula_errors = []
tensor_like_count = 0
parse_errors = 0

def looks_tensor_like(value):
    if isinstance(value, str):
        lowered = value.lower()
        return "tensor(" in lowered or "device=" in lowered or "dtype=" in lowered
    if isinstance(value, dict):
        return any(looks_tensor_like(v) for v in value.values())
    if isinstance(value, list):
        return any(looks_tensor_like(v) for v in value)
    return False

with open(path, "r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        try:
            item = json.loads(line)
        except Exception:
            parse_errors += 1
            continue

        if looks_tensor_like(item):
            tensor_like_count += 1

        event = item.get("event")
        event_counts[event] += 1

        if event == "persistent_cache_sidecar":
            sidecar_counts[item.get("action", "unknown")] += 1

        if event == "persistent_cache_compression":
            branch = item.get("branch")
            applied = bool(item.get("applied"))
            if applied:
                applied_by_branch[branch] += 1
            fallback_count += int(bool(item.get("fallback")))
            current_only_applied += int(branch == "current_only" and applied)
            clean_cache_update_applied += int(branch == "clean_cache_update" and applied)

            expected_original = (
                item.get("protected_sink_tokens", 0) +
                item.get("original_history_tokens", 0) +
                item.get("protected_current_tokens", 0)
            )
            expected_compressed = (
                item.get("protected_sink_tokens", 0) +
                item.get("compressed_history_tokens", 0) +
                item.get("protected_current_tokens", 0)
            )
            if item.get("original_visible_tokens") != expected_original:
                formula_errors.append((line_no, "original_visible"))
            if item.get("compressed_visible_tokens") != expected_compressed:
                formula_errors.append((line_no, "compressed_visible"))
            if item.get("compressed_history_tokens", 0) > item.get("original_history_tokens", 0):
                formula_errors.append((line_no, "history_order"))
            if item.get("compressed_visible_tokens", 0) > item.get("original_visible_tokens", 0):
                formula_errors.append((line_no, "visible_order"))
            if item.get("saved_visible_tokens", 0) < 0:
                formula_errors.append((line_no, "negative_saved"))

        if event == "persistent_cache_summary":
            warning_count = item.get("warning_count", warning_count)

print("event_counts", dict(event_counts))
print("applied_by_branch", dict(applied_by_branch))
print("current_only_applied", current_only_applied)
print("clean_cache_update_applied", clean_cache_update_applied)
print("fallback_count", fallback_count)
print("warning_count", warning_count)
print("sidecar_create_count", sidecar_counts.get("create", 0))
print("sidecar_reuse_count", sidecar_counts.get("reuse", 0))
print("sidecar_keep_valid_count", sidecar_counts.get("keep_valid", 0))
print("sidecar_invalidate_count", sidecar_counts.get("invalidate", 0))
print("sidecar_fallback_count", sidecar_counts.get("fallback", 0))
print("formula_errors_count", len(formula_errors))
print("formula_errors_head", formula_errors[:10])
print("tensor_like_count", tensor_like_count)
print("parse_errors", parse_errors)
PY
```

## 9. 哪些实验必须先跑

第一阶段必须按顺序跑：

1. py_compile；
2. 21-frame disabled smoke；
3. 21-frame persistent sidecar smoke；
4. 21-frame 日志和 JSONL 检查。

如果 21-frame 仍然 `sidecar_reuse_count=0`，不要跑 81-frame，先回传日志分析。

## 10. 哪些实验可以并行跑

只有当 21-frame persistent smoke 满足：

- `sidecar_create_count > 0`
- `sidecar_reuse_count > 0`
- `fallback_count = 0` 或很少且可解释
- `warning_count = 0`
- `current_only_applied = 0`
- `clean_cache_update_applied = 0`

才进入第二阶段。

第二阶段可以把 81-frame 三组放不同 GPU 并行跑：

- baseline；
- persistent ratio050；
- persistent ratio075。

## 11. 81-frame 命令

创建 ratio050 config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio050.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio050.yaml <<'YAML'

flowcache:
  enabled: true
  debug: false
  metadata_enabled: true
  metadata_output_path: logs/round5b_1_81_ratio050.jsonl
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
  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0
  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
```

创建 ratio075 config：

```bash
cp configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio050.yaml configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio075.yaml
perl -0pi -e 's#metadata_output_path: logs/round5b_1_81_ratio050.jsonl#metadata_output_path: logs/round5b_1_81_ratio075.jsonl#g' configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio075.yaml
perl -0pi -e 's#persistent_cache_compress_target_ratio: 0.5#persistent_cache_compress_target_ratio: 0.75#g' configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio075.yaml
```

81-frame baseline：

```bash
CUDA_VISIBLE_DEVICES=5 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_1_81_baseline \
  --num_output_frames 81 \
  --num_samples 3 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round5b_1_81_baseline.log
```

81-frame ratio050：

```bash
CUDA_VISIBLE_DEVICES=6 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_1_81_ratio050 \
  --num_output_frames 81 \
  --num_samples 3 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round5b_1_81_ratio050.log
```

81-frame ratio075：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5b_1_81_ratio075.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_1_81_ratio075 \
  --num_output_frames 81 \
  --num_samples 3 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round5b_1_81_ratio075.log
```

三条 81-frame 命令可以在不同终端、不同 GPU 上并行跑。

## 12. 成功标准

21-frame 成功标准：

1. disabled smoke 成功生成视频；
2. persistent smoke 成功生成视频；
3. disabled 日志没有 `[FlowCache]`；
4. persistent 日志有 `[FlowCache][persistent_cache_compression]`；
5. persistent 日志有 `[FlowCache][persistent_cache_sidecar]`；
6. persistent 日志有 `[FlowCache][persistent_cache_summary]`；
7. `sidecar_create_count > 0`；
8. `sidecar_reuse_count > 0`；
9. `sidecar_fallback_count = 0` 或很少且可解释；
10. `warning_count = 0`；
11. `fallback_count = 0`；
12. `current_only_applied = 0`；
13. `clean_cache_update_applied = 0`；
14. `compressed_history_tokens <= original_history_tokens`；
15. `compressed_visible_tokens <= original_visible_tokens`；
16. `saved_visible_tokens >= 0`；
17. JSONL 非空；
18. JSONL 不包含 tensor 内容；
19. 无 Traceback / RuntimeError / OOM；
20. 没有实现 scoring / output reuse / L1rel；
21. 没有改训练逻辑。

81-frame 成功标准：

1. baseline / ratio050 / ratio075 都成功生成 3 个视频；
2. baseline 日志没有 `[FlowCache]`；
3. ratio050 / ratio075 有 `persistent_cache_summary`；
4. `sidecar_reuse_count > 0`；
5. `warning_count = 0` 或很少且可解释；
6. `fallback_count = 0` 或很少且可解释；
7. `current_only_applied = 0`；
8. `clean_cache_update_applied = 0`；
9. 输出视频存在且非空；
10. 记录 runtime / peak allocated / peak reserved；
11. 记录 create/reuse/invalidate/fallback 分布；
12. 判断是否比 Round 5B-0 更接近 persistent reuse 目标。

## 13. 需要回传哪些文件/日志

第一阶段 21-frame 回传：

1. `logs/round5b_1_disabled_smoke.log`
2. `logs/round5b_1_persistent_smoke.log`
3. `logs/round5b_1_persistent_smoke.jsonl`
4. `configs/rolling_forcing_dmd_flowcache_round5b_1.yaml`
5. `docs/round5b_1_sidecar_reuse_fix.md`
6. `videos/round5b_1_disabled_smoke` 的 `ls -lh` 输出
7. `videos/round5b_1_persistent_smoke` 的 `ls -lh` 输出
8. summary grep 输出
9. warning/error grep 输出
10. JSONL 检查脚本输出

第二阶段 81-frame 如果跑了，回传：

1. `logs/round5b_1_81_baseline.log`
2. `logs/round5b_1_81_ratio050.log`
3. `logs/round5b_1_81_ratio075.log`
4. `logs/round5b_1_81_ratio050.jsonl`
5. `logs/round5b_1_81_ratio075.jsonl`
6. `videos/round5b_1_81_baseline` 的 `ls -lh` 输出
7. `videos/round5b_1_81_ratio050` 的 `ls -lh` 输出
8. `videos/round5b_1_81_ratio075` 的 `ls -lh` 输出
9. runtime / peak memory 表格
10. create/reuse/invalidate/fallback 表格
