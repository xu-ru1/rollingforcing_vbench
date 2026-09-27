# Round 3B KV Compression Dry-Run

Round 3B 在 Round 3A 的 `compression_candidate` metadata 基础上增加 KV compression dry-run。它只计算“如果压缩 history candidate，会节省多少 token”，不修改 K/V tensor，不改变 attention 输入，不改变 cache，也不改变输出视频。

## 1. Round 3B 做了什么

| 文件 | 修改内容 | 目的 |
| --- | --- | --- |
| `configs/default_config.yaml` | 新增 `kv_compress_dry_run`、`kv_compress_target_ratio`、`kv_compress_min_candidate_tokens`、branch apply 开关等字段，默认关闭 | 给 dry-run 提供安全配置入口 |
| `utils/flowcache.py` | 新增 `FlowCacheCompressionDryRun`、`record_compression_dry_run`、`build_compression_dry_run_plan`、`summarize_compression_dry_run` | 基于 candidate metadata 计算 projected token saving |
| `pipeline/rolling_forcing_inference.py` | inference 结束时调用 `summarize_compression_dry_run()` | 输出统一 `[FlowCache][compression_dry_run_summary]` |
| `docs/round3b_kv_compression_dryrun.md` | 本文档 | 记录 dry-run 公式、branch 策略和验证方式 |

Dry-run 事件会写入 `metadata_output_path`，例如 `logs/round3b_compression_dryrun.jsonl`。

## 2. Round 3B 没做什么

1. 没有实现真实 KV compression。
2. 没有修改 K/V tensor。
3. 没有改变 `input_key` / `input_value`。
4. 没有改变 attention 结果。
5. 没有改变 cache 写入或 eviction。
6. 没有实现 output reuse。
7. 没有实现 L1rel threshold。
8. 没有改变训练逻辑。
9. 没有删除 Self Forcing、DMD、Wan 相关代码。

## 3. 新增配置

默认配置仍然安全关闭：

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
  kv_compress_target_ratio: 0.5
  kv_compress_min_candidate_tokens: 0
  kv_compress_protect_sink: true
  kv_compress_protect_current: true
  kv_compress_apply_to_denoise: true
  kv_compress_apply_to_clean_cache_update: false
  kv_compress_apply_to_current_only: false

  output_reuse_enabled: false
  l1rel_threshold: 0.0
```

关键开关：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `kv_compress_enabled` | `false` | compression 总开关；默认关闭时 dry-run 也 no-op |
| `kv_compress_dry_run` | `false` | 只计算 projected plan，不改 tensor |
| `kv_compress_target_ratio` | `0.5` | 假设把 candidate history token 保留到多少比例 |
| `kv_compress_min_candidate_tokens` | `0` | candidate token 数小于等于该值时不应用 |
| `kv_compress_apply_to_denoise` | `true` | 对 `denoise_attention` 的 `anchor_working_current` branch 做 dry-run saving |
| `kv_compress_apply_to_clean_cache_update` | `false` | clean cache update 默认只观察，不计入 saving |
| `kv_compress_apply_to_current_only` | `false` | current-only 不压缩；代码中也强制 skip |

## 4. Dry-Run 公式

输入来自 Round 3A 的 `compression_candidate`：

```text
protected_sink_tokens
protected_current_tokens
compressible_history_tokens
total_kv_tokens
candidate_region_start
candidate_region_end
```

计算方式：

```text
if kv_compress_enabled == false:
    no-op
elif kv_compress_dry_run == false:
    no-op
elif branch 不允许应用:
    projected_history_tokens = compressible_history_tokens
elif compressible_history_tokens <= kv_compress_min_candidate_tokens:
    projected_history_tokens = compressible_history_tokens
else:
    projected_history_tokens = ceil(
        compressible_history_tokens * kv_compress_target_ratio
    )

projected_total_kv_tokens =
    protected_sink_tokens + projected_history_tokens + protected_current_tokens

saved_tokens = total_kv_tokens - projected_total_kv_tokens
saving_ratio = saved_tokens / total_kv_tokens
candidate_keep_ratio = projected_history_tokens / compressible_history_tokens
```

本轮只保存上述数字，不返回 tensor，不创建 GPU tensor。

## 5. Branch 策略

| source event / branch | 默认策略 | 原因 |
| --- | --- | --- |
| `denoise_attention` + `anchor_working_current` | apply | 这是未来 clean/history KV compression 的主要候选区 |
| `clean_cache_update_attention` / `clean_cache_update` | skip | Round 3A 发现该 branch 的 attention 输入是一整段 `working_cache_key`，不是显式三段拼接，先保守观察 |
| `current_only` | 强制 skip | 没有 history candidate，且 current KV 不应压缩 |
| 未知 branch | skip + warning | 避免误标记影响后续判断 |

## 6. 为什么 Clean Cache Update 默认只观察

`clean_cache_update` 路径里 attention 输入代码本身是 `working_cache_key` 整段，而不是显式 `anchor_cache_key + working_cache_key + roped_key`。Round 3A 的 hook 已按语义拆出 sink/current/history candidate，但真实压缩前还需要确认这条路径上的 cache 更新语义。Round 3B 因此默认 `kv_compress_apply_to_clean_cache_update=false`，只写 dry-run 记录，不计入 projected saving。

## 7. 为什么 Current-Only 不压缩

`current_only` branch 没有可压缩 history 区间，attention 只看当前计算出的 KV。压缩 current 会直接改变当前 denoising 的 attention 结果，所以代码中强制 `current_only` 为 skip。

## 8. 服务器验证命令

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
mkdir -p logs videos/round3b_disabled_smoke videos/round3b_dryrun_smoke
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

新建 Round 3B config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round3b.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round3b.yaml <<'YAML'

flowcache:
  enabled: true
  debug: true

  metadata_enabled: true
  metadata_output_path: logs/round3b_compression_dryrun.jsonl
  log_kv_ranges: true
  log_summary: true

  compression_candidate_enabled: true
  log_attention_parts: true
  log_compression_candidates: true

  kv_compress_enabled: true
  kv_compress_dry_run: true
  kv_compress_target_ratio: 0.5
  kv_compress_min_candidate_tokens: 0
  kv_compress_protect_sink: true
  kv_compress_protect_current: true
  kv_compress_apply_to_denoise: true
  kv_compress_apply_to_clean_cache_update: false
  kv_compress_apply_to_current_only: false

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
  --output_folder videos/round3b_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round3b_disabled_smoke.log
```

dry-run smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round3b.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round3b_dryrun_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round3b_dryrun_smoke.log
```

结果检查：

```bash
grep -n "\[FlowCache\]" logs/round3b_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[compression_dry_run\]" logs/round3b_dryrun_smoke.log | head -n 120
grep -n "\[FlowCache\]\[compression_dry_run_summary\]" logs/round3b_dryrun_smoke.log
grep -n "\[FlowCache\]\[warning\]" logs/round3b_dryrun_smoke.log | head -n 40

head -n 40 logs/round3b_compression_dryrun.jsonl
wc -l logs/round3b_compression_dryrun.jsonl

python - <<'PY'
import json
from collections import Counter
p = "logs/round3b_compression_dryrun.jsonl"
cnt = Counter()
saved = 0
orig = 0
proj = 0
dry = 0
with open(p, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        o = json.loads(line)
        cnt[o.get("event")] += 1
        if o.get("event") == "compression_dry_run":
            dry += 1
            saved += int(o.get("saved_tokens", 0) or 0)
            orig += int(o.get("original_total_kv_tokens", 0) or 0)
            proj += int(o.get("projected_total_kv_tokens", 0) or 0)
print("event_counts:", cnt)
print("dryrun_events:", dry)
print("orig_tokens:", orig)
print("projected_tokens:", proj)
print("saved_tokens:", saved)
print("overall_saving_ratio:", saved / orig if orig else None)
PY

ls -lh videos/round3b_disabled_smoke
ls -lh videos/round3b_dryrun_smoke
```

## 9. Round 3B 成功标准

1. disabled smoke 成功生成视频。
2. dry-run smoke 成功生成视频。
3. disabled 日志没有 `[FlowCache]`。
4. dry-run 日志有 `[FlowCache][compression_dry_run]`。
5. dry-run 日志有 `[FlowCache][compression_dry_run_summary]`。
6. JSONL 文件存在且非空。
7. JSONL 包含 `compression_dry_run` 事件。
8. `compression_dry_run` 事件包含 `projected_total_kv_tokens`、`saved_tokens`、`saving_ratio`。
9. `current_only` branch 没有被 applied。
10. `clean_cache_update` 默认没有被 applied。
11. `anchor_working_current` / `denoise_attention` branch 有 dry-run saving。
12. `saved_tokens >= 0`。
13. `projected_total_kv_tokens <= original_total_kv_tokens`。
14. sink/current protected token 不变。
15. 没有实现真实 compression。
16. 没有实现 reuse。
17. 没有改变 attention 结果。
18. 没有改变训练逻辑。

## 10. 下一轮 Round 3C 建议

Round 3C 可以做真实 clean/history KV compression prototype，但仍应默认关闭：

1. 只作用于 `denoise_attention + anchor_working_current` 的 history candidate。
2. 强制保护 sink/current。
3. 第一版只实现单层或单 window 可控实验。
4. 输出压缩前后 token 数和 attention 输出差异统计。
5. 继续保留 dry-run 模式作为对照。
