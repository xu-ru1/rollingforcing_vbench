# Round 4.3: Longer-Video No-Shadow Benchmark

## 本轮做什么

Round 4.3 只做评估补全，不新增压缩算法：

- 补齐 baseline runtime / peak CUDA memory。
- 固定 `ratio=0.5` 作为主候选。
- 保留 `ratio=0.75` 作为保守候选。
- 跑 45 frames 和 81 frames。
- 关闭 `kv_compress_shadow_compare`。
- 关闭 FlowCache debug 细粒度日志。
- 对比 baseline / ratio075 / ratio050 的 runtime、peak memory、视频大小和主观质量。

## 本轮不做什么

- 不实现 importance / redundancy scoring。
- 不实现 output reuse。
- 不实现 L1rel threshold。
- 不压缩 `current_only`。
- 默认不压缩 `clean_cache_update`。
- 不修改 `kv_cache` 本体。
- 不修改 cache eviction。
- 不修改训练逻辑。

## 说明：Baseline Metrics

baseline 不启用 FlowCache，因此不会有 `[FlowCache][real_compression_summary]`。本轮使用 `inference.py --eval_metrics` 在脚本结束时打印一行：

```text
[EvalMetrics] runtime_sec=... peak_cuda_allocated_gb=... peak_cuda_reserved_gb=... saved_videos=...
```

这不会打印 `[FlowCache]`，所以 baseline 默认关闭路径仍然可以保持干净。

## 服务器准备

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing

python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py scripts/summarize_flowcache_eval.py
```

创建 prompt 文件，避免覆盖已有文件：

```bash
mkdir -p logs videos
if [ -f logs/round4_2_prompts.txt ]; then
  echo "logs/round4_2_prompts.txt already exists; keep existing file."
else
  cat > logs/round4_2_prompts.txt <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
A futuristic city street at night, neon lights, slow camera pan.
A small dog running through a flower field, bright daylight, smooth motion.
EOF
fi
```

## 创建 Configs

```bash
make_round4_3_config() {
  frames="$1"
  tag="$2"
  ratio="$3"
  cfg="configs/rolling_forcing_dmd_flowcache_round4_3_${frames}_${tag}.yaml"
  if [ -f "$cfg" ]; then
    echo "$cfg already exists; keep existing file."
    return
  fi

  cp configs/rolling_forcing_dmd.yaml "$cfg"
  cat <<YAML >> "$cfg"

flowcache:
  enabled: true
  debug: false

  metadata_enabled: true
  metadata_output_path: logs/round4_3_${frames}_${tag}.jsonl
  log_kv_ranges: false
  log_summary: true

  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false

  kv_compress_enabled: true
  kv_compress_dry_run: false
  kv_compress_real_enabled: true
  kv_compress_strategy: uniform
  kv_compress_target_ratio: ${ratio}
  kv_compress_min_candidate_tokens: 4680
  kv_compress_protect_sink: true
  kv_compress_protect_current: true
  kv_compress_apply_to_denoise: true
  kv_compress_apply_to_clean_cache_update: false
  kv_compress_apply_to_current_only: false

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0

  eval_metrics_enabled: true
  eval_runtime_enabled: true
  eval_memory_enabled: true

  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
}

make_round4_3_config 45 ratio075 0.75
make_round4_3_config 45 ratio050 0.5
make_round4_3_config 81 ratio075 0.75
make_round4_3_config 81 ratio050 0.5
```

## 45-Frame Runs

baseline:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_3_45_baseline \
  --num_output_frames 45 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_3_45_baseline.log
```

ratio075:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_3_45_ratio075.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_3_45_ratio075 \
  --num_output_frames 45 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_3_45_ratio075.log
```

ratio050:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_3_45_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_3_45_ratio050 \
  --num_output_frames 45 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_3_45_ratio050.log
```

## 81-Frame Runs

baseline:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_3_81_baseline \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_3_81_baseline.log
```

ratio075:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_3_81_ratio075.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_3_81_ratio075 \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_3_81_ratio075.log
```

ratio050:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_3_81_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_3_81_ratio050 \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_3_81_ratio050.log
```

## 统一检查命令

baseline 不应出现 `[FlowCache]`：

```bash
grep -n "\[FlowCache\]" logs/round4_3_45_baseline.log | head -n 40
grep -n "\[FlowCache\]" logs/round4_3_81_baseline.log | head -n 40
```

ratio summary：

```bash
grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_3_45_ratio075.log
grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_3_45_ratio050.log
grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_3_81_ratio075.log
grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_3_81_ratio050.log
```

warning/error：

```bash
grep -n "\[FlowCache\]\[warning\]" logs/round4_3_45_ratio075.log | head -n 40
grep -n "\[FlowCache\]\[warning\]" logs/round4_3_45_ratio050.log | head -n 40
grep -n "\[FlowCache\]\[warning\]" logs/round4_3_81_ratio075.log | head -n 40
grep -n "\[FlowCache\]\[warning\]" logs/round4_3_81_ratio050.log | head -n 40
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round4_3_*.log
```

JSONL line counts：

```bash
wc -l logs/round4_3_45_ratio075.jsonl
wc -l logs/round4_3_45_ratio050.jsonl
wc -l logs/round4_3_81_ratio075.jsonl
wc -l logs/round4_3_81_ratio050.jsonl
```

video outputs：

```bash
ls -lh videos/round4_3_45_baseline
ls -lh videos/round4_3_45_ratio075
ls -lh videos/round4_3_45_ratio050
ls -lh videos/round4_3_81_baseline
ls -lh videos/round4_3_81_ratio075
ls -lh videos/round4_3_81_ratio050
```

metrics extraction：

```bash
grep -n "\[EvalMetrics\]" logs/round4_3_45_baseline.log
grep -n "\[EvalMetrics\]" logs/round4_3_45_ratio075.log
grep -n "\[EvalMetrics\]" logs/round4_3_45_ratio050.log
grep -n "\[EvalMetrics\]" logs/round4_3_81_baseline.log
grep -n "\[EvalMetrics\]" logs/round4_3_81_ratio075.log
grep -n "\[EvalMetrics\]" logs/round4_3_81_ratio050.log
```

summary script：

```bash
python scripts/summarize_flowcache_eval.py \
  --log-dir logs \
  --video-root videos \
  --experiments \
    round4_3_45_baseline round4_3_45_ratio075 round4_3_45_ratio050 \
    round4_3_81_baseline round4_3_81_ratio075 round4_3_81_ratio050 \
  --output-md logs/round4_3_summary.md \
  --output-csv logs/round4_3_summary.csv
```

## 汇总表格模板

| frames | variant | runtime_sec | peak_cuda_allocated_gb | peak_cuda_reserved_gb | applied_saving | overall_saving | video_count | video_total_mb | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 45 | baseline |  |  |  |  |  |  |  |  |
| 45 | ratio075 |  |  |  |  |  |  |  |  |
| 45 | ratio050 |  |  |  |  |  |  |  |  |
| 81 | baseline |  |  |  |  |  |  |  |  |
| 81 | ratio075 |  |  |  |  |  |  |  |  |
| 81 | ratio050 |  |  |  |  |  |  |  |  |

## 人工视觉检查

请记录：

| frames | prompt_id | baseline_ok | ratio075_ok | ratio050_ok | artifact | note |
| --- | --- | --- | --- | --- | --- | --- |
| 45 | 0 |  |  |  |  |  |
| 45 | 1 |  |  |  |  |  |
| 45 | 2 |  |  |  |  |  |
| 81 | 0 |  |  |  |  |  |
| 81 | 1 |  |  |  |  |  |
| 81 | 2 |  |  |  |  |  |

重点检查：

- 闪烁；
- 主体漂移；
- 背景漂移；
- 运动断裂；
- prompt 语义丢失；
- 长视频后半段退化。

## Round 4.3 成功标准

1. 45-frame baseline / ratio075 / ratio050 全部成功。
2. 81-frame baseline / ratio075 / ratio050 全部成功。
3. baseline 日志没有 `[FlowCache]`。
4. ratio 日志有 `[FlowCache][real_compression_summary]`。
5. 没有 Traceback / RuntimeError / OOM。
6. warning/fallback 为 0 或很少且可解释。
7. `current_only applied = 0`。
8. `clean_cache_update applied = 0`。
9. 输出视频存在且非空。
10. baseline runtime / peak memory 已补齐。
11. 能计算 ratio075 / ratio050 相对 baseline 的 runtime 变化。
12. 人工视觉检查完成。
13. 没有实现 importance / redundancy scoring。
14. 没有实现 output reuse。
15. 没有实现 L1rel。
16. 没有改变训练逻辑。

## 回传文件

请回传：

- `logs/round4_3_45_baseline.log`
- `logs/round4_3_45_ratio075.log`
- `logs/round4_3_45_ratio050.log`
- `logs/round4_3_81_baseline.log`
- `logs/round4_3_81_ratio075.log`
- `logs/round4_3_81_ratio050.log`
- `logs/round4_3_45_ratio075.jsonl`
- `logs/round4_3_45_ratio050.jsonl`
- `logs/round4_3_81_ratio075.jsonl`
- `logs/round4_3_81_ratio050.jsonl`
- `logs/round4_3_summary.md`
- `logs/round4_3_summary.csv`
- 四个 Round 4.3 ratio config yaml。
- `ls -lh` 视频目录输出。
- 人工视觉检查表。
