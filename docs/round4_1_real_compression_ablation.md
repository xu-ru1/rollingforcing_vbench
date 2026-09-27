# Round 4.1: Real KV Compression Ablation & Evaluation

## 本轮做什么

Round 4.1 只做评估与统计增强，不新增新的压缩算法：

- 修正 `[FlowCache][real_compression_summary]` 的 saving ratio 口径。
- 同时输出 applied-only 与 overall 两套 token saving 统计。
- 增加统一评估字段，便于比较不同 `kv_compress_target_ratio`。
- 给出 `target_ratio=0.75 / 0.5 / 0.25` 的服务器 ablation 命令。
- 给出局部 window/layer 压缩实验命令。
- 给出多 prompt 小规模评估方案。

## 本轮不做什么

- 不实现 importance / redundancy scoring。
- 不实现 output reuse。
- 不实现 L1rel threshold。
- 不压缩 `current_only`。
- 默认不压缩 `clean_cache_update`。
- 不修改 `kv_cache` 本体。
- 不修改 cache eviction。
- 不修改训练逻辑。
- 不改变 Round 4 已实现的 uniform real compression 逻辑。

## Summary 口径

Round 4 里的 `weighted_saving_ratio` 实际是 applied-only 口径。Round 4.1 拆成两个字段：

- `applied_weighted_saving_ratio`：只统计 `applied=true` 的 `real_compression` 事件。
- `overall_weighted_saving_ratio`：统计所有 `real_compression` 事件，包括 skipped、`current_only`、`clean_cache_update`。

同时输出：

- `applied_original_tokens`
- `applied_compressed_tokens`
- `applied_saved_tokens`
- `overall_original_tokens`
- `overall_compressed_tokens`
- `overall_saved_tokens`
- `real_compression_applied_by_branch`
- `real_compression_skipped_by_reason`
- `attention_output_diff_avg_relative_l1`
- `attention_output_diff_max_relative_l1`
- `attention_output_diff_avg_cosine`
- `attention_output_diff_min_cosine`

如果开启 eval 相关配置，还会输出：

- `total_runtime_sec`
- `peak_cuda_allocated_gb`
- `peak_cuda_reserved_gb`

## 配置项

新增配置项默认关闭：

```yaml
flowcache:
  eval_metrics_enabled: false
  eval_runtime_enabled: false
  eval_memory_enabled: false
```

ablation 配置里建议打开：

```yaml
  eval_metrics_enabled: true
  eval_runtime_enabled: true
  eval_memory_enabled: true
```

## Target Ratio Ablation 设计

建议测试三组：

- `kv_compress_target_ratio: 0.75`：轻压缩，预期画质风险最低，saving 较小。
- `kv_compress_target_ratio: 0.5`：Round 4 已验证的中等压缩。
- `kv_compress_target_ratio: 0.25`：强压缩，预期 saving 更高，但 attention output diff 和画质风险也更高。

判断趋势时重点看：

- `applied_weighted_saving_ratio`
- `overall_weighted_saving_ratio`
- `attention_output_diff_avg_relative_l1`
- `attention_output_diff_max_relative_l1`
- `attention_output_diff_avg_cosine`
- 输出视频是否有明显闪烁、结构漂移、细节退化。

## 局部 Window/Layer 压缩实验

已有两个限制字段：

```yaml
kv_compress_max_windows: null
kv_compress_max_layers: null
```

可以改成只压最后窗口：

```yaml
kv_compress_max_windows: 10
kv_compress_max_layers: null
```

或只压单层：

```yaml
kv_compress_max_windows: null
kv_compress_max_layers: 0
```

或只压最后窗口的前几层：

```yaml
kv_compress_max_windows: 10
kv_compress_max_layers: 0,1,2,3
```

这些局部实验可以帮助判断质量退化主要来自哪些 window/layer。

## 多 Prompt 小规模实验设计

建议先准备一个小 prompt 文件，例如 `logs/round4_1_prompts.txt`，包含 3 到 5 条短 prompt。保持：

- `--num_output_frames 21`
- `--num_samples 1`
- 同一个 checkpoint
- 同一个随机种子策略，如果仓库入口已有 seed 参数则固定 seed
- 三组 target ratio 使用完全相同 prompt 文件

建议先跑：

- baseline disabled
- ratio 0.75
- ratio 0.5
- ratio 0.25

每组比较：

- 视频是否生成；
- 文件大小；
- summary 里的 saving ratio；
- attention output diff；
- 是否 warning/fallback；
- 主观画质。

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

baseline disabled：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round4_1_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_1_disabled_smoke.log
```

创建 ratio 0.75 配置：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round4_1_ratio075.yaml
cat <<'YAML' >> configs/rolling_forcing_dmd_flowcache_round4_1_ratio075.yaml

flowcache:
  enabled: true
  debug: false
  metadata_enabled: true
  metadata_output_path: logs/round4_1_ratio075.jsonl
  log_kv_ranges: false
  log_summary: true
  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false
  kv_compress_enabled: true
  kv_compress_dry_run: false
  kv_compress_real_enabled: true
  kv_compress_target_ratio: 0.75
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
  eval_metrics_enabled: true
  eval_runtime_enabled: true
  eval_memory_enabled: true
  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
```

运行 ratio 0.75：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_1_ratio075.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round4_1_ratio075 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_1_ratio075.log
```

创建 ratio 0.5 配置：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round4_1_ratio050.yaml
cat <<'YAML' >> configs/rolling_forcing_dmd_flowcache_round4_1_ratio050.yaml

flowcache:
  enabled: true
  debug: false
  metadata_enabled: true
  metadata_output_path: logs/round4_1_ratio050.jsonl
  log_kv_ranges: false
  log_summary: true
  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false
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
  eval_metrics_enabled: true
  eval_runtime_enabled: true
  eval_memory_enabled: true
  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
```

运行 ratio 0.5：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_1_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round4_1_ratio050 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_1_ratio050.log
```

创建 ratio 0.25 配置：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round4_1_ratio025.yaml
cat <<'YAML' >> configs/rolling_forcing_dmd_flowcache_round4_1_ratio025.yaml

flowcache:
  enabled: true
  debug: false
  metadata_enabled: true
  metadata_output_path: logs/round4_1_ratio025.jsonl
  log_kv_ranges: false
  log_summary: true
  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false
  kv_compress_enabled: true
  kv_compress_dry_run: false
  kv_compress_real_enabled: true
  kv_compress_target_ratio: 0.25
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
  eval_metrics_enabled: true
  eval_runtime_enabled: true
  eval_memory_enabled: true
  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
```

运行 ratio 0.25：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_1_ratio025.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round4_1_ratio025 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_1_ratio025.log
```

## 局部压缩命令示例

只压最后 window：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round4_1_ratio050_win10.yaml
cat <<'YAML' >> configs/rolling_forcing_dmd_flowcache_round4_1_ratio050_win10.yaml

flowcache:
  enabled: true
  debug: false
  metadata_enabled: true
  metadata_output_path: logs/round4_1_ratio050_win10.jsonl
  log_kv_ranges: false
  log_summary: true
  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false
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
  kv_compress_max_windows: 10
  kv_compress_max_layers: null
  eval_metrics_enabled: true
  eval_runtime_enabled: true
  eval_memory_enabled: true
  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML

CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_1_ratio050_win10.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round4_1_ratio050_win10 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_1_ratio050_win10.log
```

只压 layer 0：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round4_1_ratio050_layer0.yaml
cat <<'YAML' >> configs/rolling_forcing_dmd_flowcache_round4_1_ratio050_layer0.yaml

flowcache:
  enabled: true
  debug: false
  metadata_enabled: true
  metadata_output_path: logs/round4_1_ratio050_layer0.jsonl
  log_kv_ranges: false
  log_summary: true
  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false
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
  kv_compress_max_layers: 0
  eval_metrics_enabled: true
  eval_runtime_enabled: true
  eval_memory_enabled: true
  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML

CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_1_ratio050_layer0.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round4_1_ratio050_layer0 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_1_ratio050_layer0.log
```

## 统一检查命令

```bash
grep -n "\[FlowCache\]" logs/round4_1_disabled_smoke.log | head -n 40

for tag in ratio075 ratio050 ratio025; do
  echo "===== ${tag} summary ====="
  grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_1_${tag}.log
  echo "===== ${tag} diff ====="
  grep -n "\[FlowCache\]\[attention_output_diff\]" logs/round4_1_${tag}.log | head -n 20
  echo "===== ${tag} warning ====="
  grep -n "\[FlowCache\]\[warning\]" logs/round4_1_${tag}.log | head -n 40
  echo "===== ${tag} video ====="
  ls -lh videos/round4_1_${tag}
done
```

JSONL 统计：

```bash
python - <<'PY'
import json
from collections import Counter, defaultdict

for tag in ["ratio075", "ratio050", "ratio025"]:
    p = f"logs/round4_1_{tag}.jsonl"
    cnt = Counter()
    branch = Counter()
    applied = Counter()
    skipped_reason = Counter()
    fallback = 0
    summary = None
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
                else:
                    skipped_reason[o.get("skipped_reason", "unknown")] += 1
                fallback += int(bool(o.get("fallback")))
            elif e == "real_compression_summary":
                summary = o

    print("====", tag, "====")
    print("event_counts:", cnt)
    print("branch_counts:", branch)
    print("applied_by_branch:", applied)
    print("skipped_by_reason:", skipped_reason)
    print("fallback_count:", fallback)
    if summary:
        print("applied_weighted_saving_ratio:", summary.get("applied_weighted_saving_ratio"))
        print("overall_weighted_saving_ratio:", summary.get("overall_weighted_saving_ratio"))
        print("avg_relative_l1:", summary.get("attention_output_diff_avg_relative_l1"))
        print("max_relative_l1:", summary.get("attention_output_diff_max_relative_l1"))
        print("avg_cosine:", summary.get("attention_output_diff_avg_cosine"))
        print("min_cosine:", summary.get("attention_output_diff_min_cosine"))
        print("total_runtime_sec:", summary.get("total_runtime_sec"))
        print("peak_cuda_allocated_gb:", summary.get("peak_cuda_allocated_gb"))
        print("peak_cuda_reserved_gb:", summary.get("peak_cuda_reserved_gb"))
PY
```

## 如何判断压缩是否值得继续

继续推进的信号：

- 三组 ratio 都能稳定生成视频；
- warning/fallback 接近 0；
- `target_ratio` 越小，saving ratio 越大；
- `target_ratio=0.75` 和 `0.5` 的画质主观变化较小；
- shadow compare 的 cosine 保持较高，relative L1 没有异常跳变。

需要收敛策略的信号：

- `target_ratio=0.25` 明显退化；
- 某些 prompt 出现严重闪烁或结构漂移；
- 某些 window/layer 的局部压缩引起明显不稳定；
- attention diff 在低 ratio 下大幅上升。

## Round 4.1 成功标准

1. disabled smoke 成功。
2. ratio 0.75 / 0.5 / 0.25 三组都能生成视频。
3. disabled 日志没有 `[FlowCache]`。
4. 三组 real compression 日志都有 `[FlowCache][real_compression_summary]`。
5. 三组 JSONL 都非空。
6. `applied_weighted_saving_ratio` 和 `overall_weighted_saving_ratio` 都存在。
7. `current_only applied = 0`。
8. `clean_cache_update applied = 0`。
9. `fallback_count = 0` 或非常少且有 warning。
10. 没有 Traceback / RuntimeError / OOM。
11. 视频文件存在且非空。
12. shadow compare 没有 NaN/Inf。
13. `target_ratio` 越小，saving ratio 应总体越大。
14. `target_ratio` 越小，attention output diff 通常应不小于更高 ratio。
15. 没有实现 output reuse。
16. 没有实现 L1rel threshold。
17. 没有改变训练逻辑。

## 下一轮建议

- 如果 uniform compression 质量稳定，再做 importance / redundancy scoring。
- 如果质量明显下降，先降低压缩强度或限制压缩范围。
- 暂时不要做 output reuse，避免把质量退化来源混在一起。
