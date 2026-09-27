# Round 4.4-lite: Minimal Overhead Check

## 目标

Round 4.4-lite 只做最小开销确认，不新增压缩算法，不实现 output reuse，不实现 L1rel。

本轮要验证：在真实 KV compression 已开启的情况下，关闭 JSONL、关闭 per-event metadata、关闭 debug 后，ratio050 是否仍然明显慢于 baseline。

如果 no-jsonl 模式仍明显慢于 baseline，且 peak CUDA memory 不下降，说明当前“只压缩本次 attention input”的 prototype 不是性能收益版本，下一轮应进入 history KV cache 本体压缩 prototype。

## 本轮不做什么

- 不实现 importance / redundancy scoring。
- 不实现 output reuse。
- 不实现 L1rel threshold。
- 不压缩 current_only。
- 不压缩 clean_cache_update。
- 不修改 kv_cache 本体。
- 不修改 cache eviction。
- 不修改训练逻辑。

## 配置模式

关键是关闭 JSONL 和细粒度日志，只保留最终 summary：

```yaml
flowcache:
  enabled: true
  debug: false

  metadata_enabled: false
  metadata_output_path: null
  log_kv_ranges: false
  log_summary: true

  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false

  kv_compress_enabled: true
  kv_compress_dry_run: false
  kv_compress_real_enabled: true
  kv_compress_strategy: uniform
  kv_compress_target_ratio: 0.5
  kv_compress_min_candidate_tokens: 4680
  kv_compress_protect_sink: true
  kv_compress_protect_current: true
  kv_compress_apply_to_denoise: true
  kv_compress_apply_to_clean_cache_update: false
  kv_compress_apply_to_current_only: false

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0

  output_reuse_enabled: false
  l1rel_threshold: 0.0
```

预期日志：

- baseline 日志没有 `[FlowCache]`。
- ratio050_nojsonl 日志只需要有最终 `[FlowCache][real_compression_summary]`。
- 不应出现 `[FlowCache][candidate_summary]`、`[FlowCache][attention_parts]`、`[FlowCache][compression_candidate]`。
- 不生成 JSONL 文件。

## 服务器命令

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

准备 prompt 文件：

```bash
mkdir -p logs videos

if [ ! -f logs/round4_2_prompts.txt ]; then
cat > logs/round4_2_prompts.txt <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
A futuristic city street at night, neon lights, slow camera pan.
A small dog running through a flower field, bright daylight, smooth motion.
EOF
fi
```

创建 no-jsonl 配置：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round4_4_81_ratio050_nojsonl.yaml

cat >> configs/rolling_forcing_dmd_flowcache_round4_4_81_ratio050_nojsonl.yaml <<'EOF'

flowcache:
  enabled: true
  debug: false

  metadata_enabled: false
  metadata_output_path: null
  log_kv_ranges: false
  log_summary: true

  compression_candidate_enabled: true
  log_attention_parts: false
  log_compression_candidates: false

  kv_compress_enabled: true
  kv_compress_dry_run: false
  kv_compress_real_enabled: true
  kv_compress_strategy: uniform
  kv_compress_target_ratio: 0.5
  kv_compress_min_candidate_tokens: 4680
  kv_compress_protect_sink: true
  kv_compress_protect_current: true
  kv_compress_apply_to_denoise: true
  kv_compress_apply_to_clean_cache_update: false
  kv_compress_apply_to_current_only: false

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0

  output_reuse_enabled: false
  l1rel_threshold: 0.0
EOF
```

81-frame baseline 可以复用 `logs/round4_3_81_baseline.log`。如果需要补跑：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_4_81_baseline \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_4_81_baseline.log
```

运行 ratio050_nojsonl：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round4_4_81_ratio050_nojsonl.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round4_4_81_ratio050_nojsonl \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round4_4_81_ratio050_nojsonl.log
```

## 检查命令

```bash
grep -n "\[FlowCache\]" logs/round4_4_81_baseline.log | head -n 40
grep -n "\[FlowCache\]" logs/round4_4_81_ratio050_nojsonl.log

grep -n "\[FlowCache\]\[real_compression_summary\]" logs/round4_4_81_ratio050_nojsonl.log
grep -n "\[FlowCache\]\[warning\]\|Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round4_4_81_ratio050_nojsonl.log | head -n 80
grep -n "fallback_count=[1-9]" logs/round4_4_81_ratio050_nojsonl.log

grep -n "\[EvalMetrics\]" logs/round4_4_81_baseline.log
grep -n "\[EvalMetrics\]" logs/round4_4_81_ratio050_nojsonl.log

test ! -f logs/round4_4_81_ratio050_nojsonl.jsonl && echo "jsonl disabled: ok"

ls -lh videos/round4_4_81_baseline
ls -lh videos/round4_4_81_ratio050_nojsonl
```

对比 runtime 和 peak memory：

```bash
python - <<'PY'
import os, re

baseline = "logs/round4_4_81_baseline.log"
if not os.path.exists(baseline):
    baseline = "logs/round4_3_81_baseline.log"
ratio = "logs/round4_4_81_ratio050_nojsonl.log"

pat = re.compile(
    r"\[EvalMetrics\].*runtime_sec=([0-9.]+).*"
    r"peak_cuda_allocated_gb=([0-9.]+).*"
    r"peak_cuda_reserved_gb=([0-9.]+)"
)

def read(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    m = pat.search(text)
    if not m:
        return None
    return tuple(float(x) for x in m.groups())

b = read(baseline)
r = read(ratio)
print("baseline_log:", baseline, b)
print("ratio_log:", ratio, r)
if b and r:
    print("runtime_delta_sec:", r[0] - b[0])
    print("runtime_delta_pct:", (r[0] / b[0] - 1.0) * 100)
    print("allocated_delta_gb:", r[1] - b[1])
    print("reserved_delta_gb:", r[2] - b[2])
PY
```

## 判断标准

如果 ratio050_nojsonl 仍明显慢于 baseline，并且 peak allocated / reserved 不下降：

- 当前临时 input compression 不是性能收益版本。
- 下一轮应转向 history KV cache 本体压缩 prototype。

如果 ratio050_nojsonl 接近或快于 baseline：

- Round 4.3 的慢主要来自 JSONL / metadata / summary 开销。
- 下一轮优先优化 logging 和 summary，再考虑更复杂压缩策略。

## 需要回传

- `logs/round4_4_81_ratio050_nojsonl.log`
- `logs/round4_4_81_baseline.log`，如果复用 baseline，则回传 `logs/round4_3_81_baseline.log`
- runtime 对比 Python 输出
- `ls -lh videos/round4_4_81_ratio050_nojsonl` 输出
- 如补跑 baseline，也回传 `ls -lh videos/round4_4_81_baseline` 输出

