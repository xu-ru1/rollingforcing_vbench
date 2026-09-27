# Round 4.2: Multi Prompt Real KV Compression Evaluation

## 本轮做什么

Round 4.2 用更接近真实使用的设置评估当前 uniform real KV compression：

- 使用 3 条短 prompt。
- 关闭 `kv_compress_shadow_compare`。
- 对比 baseline / ratio075 / ratio050 / ratio025。
- 记录 runtime、peak CUDA memory、JSONL summary、视频文件大小。
- 使用统一脚本汇总日志。
- 给出人工视觉检查清单。
- 判断 `ratio=0.75` 和 `ratio=0.5` 哪个更适合作为后续默认设置。

## 本轮不做什么

- 不新增 importance / redundancy scoring。
- 不实现 output reuse。
- 不实现 L1rel threshold。
- 不压缩 `current_only`。
- 默认不压缩 `clean_cache_update`。
- 不修改 `kv_cache` 本体。
- 不修改 cache eviction。
- 不修改训练逻辑。
- 不改变 Round 4 已实现的 uniform token selection 逻辑。

## 为什么关闭 Shadow Compare

Round 4 和 Round 4.1 的 shadow compare 会在少量 event 上同时计算原始 attention 和压缩 attention，用于估计输出差异。但它会带来额外 attention 计算，runtime 和 peak memory 都不是纯压缩路径的真实参考。

Round 4.2 关闭 shadow compare，目的是得到更接近真实 inference 的：

- runtime；
- peak CUDA memory；
- 输出视频；
- 多 prompt 稳定性。

注意：由于 Round 4 prototype 仍不压缩 `kv_cache` 本体，只压缩本次 attention input，所以显存收益可能有限；runtime 也可能受 Python/IO/attention kernel 影响。Round 4.2 的重点是稳定性和质量边界。

## 为什么测 0.75 / 0.5 / 0.25

- `0.75`：轻压缩，质量风险最低，适合作为保守默认候选。
- `0.5`：Round 4 已验证中等压缩，saving 更明显，是主要候选。
- `0.25`：强压缩压力测试，用于观察质量崩坏边界。

如果 `0.5` 多 prompt 主观质量稳定，下一轮可以把它作为默认候选继续做 importance/redundancy scoring；如果 `0.5` 明显退化但 `0.75` 稳定，则优先采用 `0.75`。

## 服务器准备

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py scripts/summarize_flowcache_eval.py
```

创建 prompt 文件。为避免覆盖已有文件，使用 guarded 写法：

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

## 生成配置

以下命令基于 `configs/rolling_forcing_dmd.yaml` 生成三组 no-shadow config：

```bash
make_round4_2_config() {
  tag="$1"
  ratio="$2"
  cfg="configs/rolling_forcing_dmd_flowcache_round4_2_${tag}.yaml"
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
  metadata_output_path: logs/round4_2_${tag}.jsonl
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

make_round4_2_config ratio075 0.75
make_round4_2_config ratio050 0.5
make_round4_2_config ratio025 0.25
```

## 运行 Baseline

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_2_baseline \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_2_baseline.log
```

## 运行 Ratio 0.75

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_2_ratio075.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_2_ratio075 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_2_ratio075.log
```

## 运行 Ratio 0.5

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_2_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_2_ratio050 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_2_ratio050.log
```

## 运行 Ratio 0.25

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_2_ratio025.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_2_ratio025 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round4_2_ratio025.log
```

## 统一检查命令

```bash
grep -n "\[FlowCache\]" logs/round4_2_baseline.log | head -n 40

grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_2_ratio075.log
grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_2_ratio050.log
grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_2_ratio025.log

grep -n "\[FlowCache\]\[warning\]" logs/round4_2_ratio075.log | head -n 40
grep -n "\[FlowCache\]\[warning\]" logs/round4_2_ratio050.log | head -n 40
grep -n "\[FlowCache\]\[warning\]" logs/round4_2_ratio025.log | head -n 40

grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round4_2_*.log

wc -l logs/round4_2_ratio075.jsonl
wc -l logs/round4_2_ratio050.jsonl
wc -l logs/round4_2_ratio025.jsonl

ls -lh videos/round4_2_baseline
ls -lh videos/round4_2_ratio075
ls -lh videos/round4_2_ratio050
ls -lh videos/round4_2_ratio025
```

运行统一汇总脚本：

```bash
python scripts/summarize_flowcache_eval.py \
  --log-dir logs \
  --video-root videos \
  --experiments round4_2_baseline round4_2_ratio075 round4_2_ratio050 round4_2_ratio025 \
  --output-md logs/round4_2_summary.md \
  --output-csv logs/round4_2_summary.csv
```

## 人工视觉检查清单

请逐个 prompt 对 baseline / ratio075 / ratio050 / ratio025 做主观检查：

- 是否黑屏/花屏；
- 主体是否稳定；
- 运动是否连续；
- 是否闪烁；
- 背景是否漂移；
- prompt 语义是否保持；
- ratio075 / ratio050 / ratio025 与 baseline 的差异。

记录表格模板：

| prompt_id | baseline_ok | r075_ok | r050_ok | r025_ok | visible_artifact | note |
| --- | --- | --- | --- | --- | --- | --- |
| 0 |  |  |  |  |  |  |
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |

建议 artifact 关键词：

- `none`
- `minor_flicker`
- `major_flicker`
- `subject_drift`
- `background_drift`
- `semantic_loss`
- `motion_jitter`
- `black_or_corrupt`

## 如何判断下一轮走向

优先选择 `ratio=0.75` 的情况：

- `ratio=0.5` 在 3 条 prompt 中出现明显闪烁、主体漂移或语义偏移；
- `ratio=0.75` 与 baseline 差异很小；
- `ratio=0.5` 的 runtime 或质量收益不稳定。

可以选择 `ratio=0.5` 的情况：

- 3 条 prompt 人工检查都基本可接受；
- warning/fallback 为 0；
- `current_only` 和 `clean_cache_update` applied 仍为 0；
- 相比 `0.75` 有明显 token saving；
- 视频无明显稳定性退化。

`ratio=0.25` 暂时建议只作为压力测试。如果 0.25 也稳定，再考虑后续重要性评分；如果明显退化，说明 uniform compression 已经到达质量边界。

## 需要回传的文件

请回传：

- `logs/round4_2_baseline.log`
- `logs/round4_2_ratio075.log`
- `logs/round4_2_ratio050.log`
- `logs/round4_2_ratio025.log`
- `logs/round4_2_ratio075.jsonl`
- `logs/round4_2_ratio050.jsonl`
- `logs/round4_2_ratio025.jsonl`
- `logs/round4_2_summary.md`
- `logs/round4_2_summary.csv`
- `configs/rolling_forcing_dmd_flowcache_round4_2_ratio075.yaml`
- `configs/rolling_forcing_dmd_flowcache_round4_2_ratio050.yaml`
- `configs/rolling_forcing_dmd_flowcache_round4_2_ratio025.yaml`
- `ls -lh videos/round4_2_baseline videos/round4_2_ratio075 videos/round4_2_ratio050 videos/round4_2_ratio025` 的输出
- 人工视觉检查表。

## Round 4.2 成功标准

1. baseline 多 prompt 成功。
2. ratio075 / ratio050 / ratio025 三组多 prompt 成功。
3. baseline 日志没有 `[FlowCache]`。
4. 三组 compression 日志有 `[FlowCache][real_compression_summary]`。
5. 没有 `[FlowCache][warning]` 或 warning 很少且可解释。
6. 没有 Traceback / RuntimeError / OOM。
7. 三组 JSONL 非空。
8. 三组视频文件存在且非空。
9. `current_only applied = 0`。
10. `clean_cache_update applied = 0`。
11. `target_ratio` 越小，saving 越大。
12. no-shadow runtime 可以作为比 Round 4.1 更真实的测速参考。
13. 人工视觉检查至少完成 3 prompts。
14. 没有实现 output reuse。
15. 没有实现 L1rel threshold。
16. 没有改变训练逻辑。
