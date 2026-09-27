# Round 4: 真实 KV Compression Prototype

## 本轮做了什么

Round 4 在默认关闭的前提下，加入了一个非常保守的真实 KV compression prototype：

- 新增 `kv_compress_real_enabled` 开关，默认 `false`。
- 只在 `flowcache.enabled=true`、`kv_compress_enabled=true`、`kv_compress_real_enabled=true` 且 `kv_compress_dry_run=false` 时生效。
- 只尝试作用于 `denoise_attention` / `anchor_working_current` 分支。
- 只对 `working_cache_key` / `working_cache_value` 的 history token 做 uniform token selection。
- 强制保护 attention sink，也就是 `anchor_cache_key/value`。
- 强制保护 current denoising KV，也就是 `roped_key/value`。
- 不修改 `kv_cache` 本体，不修改 `global_end_index` / `local_end_index`，不修改 eviction。
- 保留 Round 3A/3B 的 metadata、candidate、dry-run 能力，用于对照。
- 可选 `kv_compress_shadow_compare=true` 时，对少量事件同时计算原始 attention 和压缩 attention，并记录输出差异。

## 本轮没做什么

- 没有实现最终 FlowCache compression 算法。
- 没有做 importance / redundancy scoring。
- 没有压缩 sink。
- 没有压缩 current。
- 没有压缩 `current_only`。
- 默认没有压缩 `clean_cache_update`，即使显式打开也会在 Round 4 标记为暂不支持。
- 没有修改 cache 写入逻辑。
- 没有修改 cache eviction。
- 没有实现 output reuse。
- 没有实现 L1rel threshold。
- 没有修改训练逻辑或 checkpoint 格式。

## 配置项

新增或补充的安全默认配置位于 `configs/default_config.yaml`：

```yaml
flowcache:
  enabled: false
  debug: false

  metadata_enabled: false
  metadata_output_path: null
  log_kv_ranges: false
  log_summary: true

  compression_candidate_enabled: false
  log_attention_parts: false
  log_compression_candidates: false

  kv_compress_enabled: false
  kv_compress_dry_run: false
  kv_compress_real_enabled: false
  kv_compress_target_ratio: 0.5
  kv_compress_min_candidate_tokens: 4680
  kv_compress_strategy: uniform
  kv_compress_protect_sink: true
  kv_compress_protect_current: true
  kv_compress_apply_to_denoise: true
  kv_compress_apply_to_clean_cache_update: false
  kv_compress_apply_to_current_only: false

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 5
  kv_compress_max_windows: null
  kv_compress_max_layers: null

  output_reuse_enabled: false
  l1rel_threshold: 0.0
```

## 为什么第一版只做 uniform token selection

Round 4 的目标是验证真实缩短 attention KV 输入是否能安全跑通，而不是追求最优画质或速度。因此第一版只采用 uniform / linspace keep strategy：

```text
keep_tokens = ceil(original_history_tokens * kv_compress_target_ratio)
```

然后在 history token 维度上等间隔保留 token，保持升序和时间顺序。这个策略简单、可解释、容易回滚，也方便和 Round 3B dry-run 的 projected token 统计对齐。

## 为什么只压 anchor_working_current / denoise_attention

RollingForcing 的 denoise 分支里 attention 输入语义最清晰：

```text
input_key   = concat(anchor_cache_key, working_cache_key, roped_key)
input_value = concat(anchor_cache_value, working_cache_value, value)
```

Round 4 只替换中间的 `working_cache_key/value`：

```text
compressed_input_key   = concat(anchor_cache_key, compressed_working_cache_key, roped_key)
compressed_input_value = concat(anchor_cache_value, compressed_working_cache_value, value)
```

这样可以明确保护 sink/current，并且只影响本次 self-attention 的可见 history KV。

## 为什么保护 sink/current

- sink / anchor KV 对长程稳定性和全局上下文很关键，Round 4 不压缩。
- current denoising KV 是当前窗口正在生成或去噪的 token，不能压缩。
- FlowCache 后续真正的 history compression 应该只作用于已经离开 current window 的 clean/history 区域。

## 为什么默认不压 clean_cache_update

Round 3A 已确认 `clean_cache_update` 分支里实际 attention 输入是 `working_cache_key` 整段，而不是显式 `anchor + working + current` 三段拼接。Round 4 为避免误伤 clean cache 更新，仍只观察该分支，并在 real compression 统计里记录 skipped。

## 为什么不修改 kv_cache 本体

Round 4 只压缩本次 attention 的 `input_key/input_value`，不把压缩后的 KV 写回 `kv_cache`。这样可以保证：

- cache 写入逻辑不变；
- cache eviction 不变；
- `global_end_index` / `local_end_index` 不变；
- 后续 window 仍从原始 KV cache 读取；
- prototype 可以快速开关和回滚。

## fallback 策略

真实压缩前会检查：

- history key/value token 数是否一致；
- `keep_tokens` 是否在合法范围内；
- 压缩后拼接长度是否等于计划长度；
- compressed attention 是否能正常执行。

如果任一步失败，会回退到原始 `input_key/input_value`，记录 `[FlowCache][warning]` 和 `real_compression` 的 `fallback=true`，不会中断 inference。

## shadow compare 策略

当 `kv_compress_shadow_compare=true` 时，前 `kv_compress_shadow_max_events` 个真实压缩事件会同时计算：

- 原始 attention 输出；
- 压缩 attention 输出。

实际输出仍使用压缩 attention。日志和 JSONL 记录：

- `mean_abs_diff`
- `max_abs_diff`
- `relative_l1`
- `cosine_similarity`

shadow compare 只用于短 smoke，会增加耗时。

## 服务器验证命令

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

新建 Round 4 config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round4.yaml
cat <<'YAML' >> configs/rolling_forcing_dmd_flowcache_round4.yaml

flowcache:
  enabled: true
  debug: true

  metadata_enabled: true
  metadata_output_path: logs/round4_real_compression.jsonl
  log_kv_ranges: true
  log_summary: true

  compression_candidate_enabled: true
  log_attention_parts: true
  log_compression_candidates: true

  kv_compress_enabled: true
  kv_compress_dry_run: false
  kv_compress_real_enabled: true
  kv_compress_target_ratio: 0.5
  kv_compress_min_candidate_tokens: 4680
  kv_compress_strategy: uniform
  kv_compress_protect_sink: true
  kv_compress_protect_current: true
  kv_compress_apply_to_denoise: true
  kv_compress_apply_to_clean_cache_update: false
  kv_compress_apply_to_current_only: false

  kv_compress_shadow_compare: true
  kv_compress_shadow_max_events: 5
  kv_compress_max_windows: null
  kv_compress_max_layers: null

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
  --output_folder videos/round4_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_disabled_smoke.log
```

real compression smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round4_real_compression_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_real_compression_smoke.log
```

结果检查：

```bash
grep -n "\[FlowCache\]" logs/round4_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[real_compression\]" logs/round4_real_compression_smoke.log | head -n 120
grep -n "\[FlowCache\]\[attention_output_diff\]" logs/round4_real_compression_smoke.log | head -n 40
grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_real_compression_smoke.log
grep -n "\[FlowCache\]\[warning\]" logs/round4_real_compression_smoke.log | head -n 80

head -n 60 logs/round4_real_compression.jsonl
wc -l logs/round4_real_compression.jsonl

python - <<'PY'
import json
from collections import Counter, defaultdict

p = "logs/round4_real_compression.jsonl"
cnt = Counter()
branch = Counter()
applied = Counter()
saved_by_branch = defaultdict(int)
orig_by_branch = defaultdict(int)
fallbacks = 0

with open(p, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        o = json.loads(line)
        e = o.get("event")
        cnt[e] += 1
        if e == "real_compression":
            b = o.get("branch", "unknown")
            branch[b] += 1
            if o.get("applied"):
                applied[b] += 1
                saved_by_branch[b] += int(o.get("saved_tokens", 0) or 0)
                orig_by_branch[b] += int(o.get("original_total_kv_tokens", 0) or 0)
            if o.get("fallback"):
                fallbacks += 1

print("event_counts:", cnt)
print("real_compression_branch_counts:", branch)
print("applied_by_branch:", applied)
print("fallbacks:", fallbacks)
for b in sorted(branch):
    orig = orig_by_branch[b]
    saved = saved_by_branch[b]
    print(b, "orig=", orig, "saved=", saved, "saving_ratio=", saved / orig if orig else None)
PY

ls -lh videos/round4_disabled_smoke
ls -lh videos/round4_real_compression_smoke
```

## Round 4 成功标准

1. disabled smoke 成功生成视频。
2. real compression smoke 成功生成视频。
3. disabled 日志没有 `[FlowCache]`。
4. real compression 日志有 `[FlowCache][real_compression]`。
5. real compression 日志有 `[FlowCache][real_compression_summary]`。
6. JSONL 文件存在且非空。
7. JSONL 包含 `real_compression` 事件。
8. `applied=true` 只发生在 `anchor_working_current` / `denoise_attention`。
9. `current_only` 的 applied 必须为 0。
10. `clean_cache_update` 默认 applied 必须为 0。
11. `protected_sink_tokens` 不变。
12. `protected_current_tokens` 不变。
13. `compressed_history_tokens <= original_history_tokens`。
14. `compressed_total_kv_tokens <= original_total_kv_tokens`。
15. `saved_tokens >= 0`。
16. `fallback_count` 为 0 或非常少，且有 warning 说明。
17. 没有修改 `kv_cache` 本体。
18. 没有改变 cache eviction。
19. 没有实现 output reuse。
20. 没有实现 L1rel threshold。
21. 没有改变训练逻辑。
22. 输出视频文件存在且非空。
23. 如果 shadow compare 开启，日志中有 `attention_output_diff`，且没有 NaN/Inf。

## 下一轮建议

- 多 prompt smoke，观察画质退化是否明显。
- 做 `kv_compress_target_ratio` ablation，例如 0.75 / 0.5 / 0.25。
- 先限制 `kv_compress_max_windows` / `kv_compress_max_layers` 做局部实验。
- 如果 uniform 可跑通，再考虑 importance / redundancy scoring。
- 最后再接 output reuse / L1rel threshold，不要和真实 KV compression 同轮混改。
