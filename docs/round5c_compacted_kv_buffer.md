# Round 5C：compacted KV buffer prototype

## Round 5C 做什么

Round 5C 将目标从 Round 5B 的 persistent sidecar reuse，推进到更接近收益路径的 compacted KV read buffer。它在 self-attention 的 denoise `anchor_working_current` 分支中，对 history / working KV 生成更短的 contiguous compacted K/V，然后让 attention 读取：

```text
input = sink(anchor) + compacted_history + current
```

主 `kv_cache` 仍保留完整 K/V，用于 fallback、cache 写入和 eviction。5C 的核心验证点是：在不改训练、不改 checkpoint、不改 cross-attn、不改 eviction 主逻辑的前提下，能否稳定减少 attention 可见 history KV 长度。

## Round 5C 不做什么

本轮不做 importance / redundancy scoring，不做 output reuse，不做 L1rel threshold，不压缩 current_only，不默认压缩 clean_cache_update，不压缩 sink，不压缩 current denoising KV，不改 cross-attn，不改训练逻辑，不改 checkpoint，不重写 scheduler，也不重写 rolling forcing 主循环。

## 为什么 5B sidecar 不是性能收益版本

Round 5B 的 sidecar 能证明 compressed history 表示可以持久化，但 sidecar 本身额外保存 compressed history tensor，而主 `kv_cache` 仍是固定预分配 tensor。长视频下 sidecar 又会被 eviction 打断，81-frame single-prompt 已经显示 reuse 为 0。因此 5B 正确性稳定，但不适合作为性能收益路径继续深挖。

## 5C compacted KV buffer 设计

5C 使用默认关闭的 `compacted_kv_*` 配置。启用后，每层 self-attention 在构造 denoise attention input 前：

1. 读取原始 sink / history / current；
2. 保护 sink 和 current；
3. 对 history 使用 uniform / linspace selection；
4. 构造更短的 compacted history K/V；
5. 拼接 `sink + compacted_history + current`；
6. attention 只看到 compacted visible KV；
7. 如果任一检查失败，fallback 到原始 history；
8. 记录 `[FlowCache][compacted_kv]` 和 JSONL；
9. inference 末尾输出 `[FlowCache][compacted_kv_summary]`。

本轮不修改主 `kv_cache["k"]` / `kv_cache["v"]`，因此不保证 peak allocated / reserved 立刻下降。它比 5B 更接近收益路径，是因为它不依赖长期 sidecar reuse，而是直接缩短当前 attention 的 visible history。

## sink/current 保护策略

`compacted_kv_protect_sink=true` 和 `compacted_kv_protect_current=true` 是必需保护项。sink(anchor) 原样保留；当前 denoising KV 原样保留；只对中间 history / working KV 做 uniform selection。

## current_only / clean_cache_update

`current_only` 永远不 applied，会记录 `current_only_protected`。`clean_cache_update` 默认不压缩，会记录 `clean_cache_update_disabled`。这样可以避免影响 clean cache rerun 和 cache 写入路径。

## fallback 条件

以下情况会 fallback 或 skipped：

1. `compacted_kv_real_enabled=false`；
2. branch 是 `current_only`；
3. clean cache update 默认关闭；
4. history tokens 小于等于 `compacted_kv_min_history_tokens`；
5. strategy 非 `uniform` / `linspace`；
6. key/value history 长度不一致；
7. keep token 数非法；
8. compacted visible token 公式不匹配；
9. device / dtype / finite tensor 检查失败；
10. attention 执行异常。

fallback 会记录 `fallback=true` 和 `fallback_reason`，不会 silent failure。

## 日志字段

`[FlowCache][compacted_kv]` 和 JSONL 中包含：

- `event`
- `flowcache_run_id`
- `prompt_idx`
- `sample_idx`
- `window_index`
- `layer_idx`
- `branch`
- `applied`
- `fallback`
- `fallback_reason`
- `original_history_tokens`
- `compacted_history_tokens`
- `protected_sink_tokens`
- `protected_current_tokens`
- `original_visible_tokens`
- `compacted_visible_tokens`
- `saved_visible_tokens`
- `target_ratio`
- `strategy`
- `dtype`
- `device`
- `elapsed_ms`

`[FlowCache][compacted_kv_summary]` 包含：

- `total_events`
- `applied_events`
- `skipped_events`
- `fallback_count`
- `warning_count`
- `current_only_applied`
- `clean_cache_update_applied`
- `applied_weighted_visible_saving_ratio`
- `overall_weighted_visible_saving_ratio`
- `avg_compacted_history_tokens`
- `avg_original_history_tokens`
- `runtime_sec`
- `peak_cuda_allocated_gb`
- `peak_cuda_reserved_gb`
- `target_ratio`
- `strategy`

## 服务器实验命令

进入服务器目录：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py scripts/summarize_round5c.py
```

创建 ratio050 config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml <<'YAML'

flowcache:
  enabled: true
  debug: false

  metadata_enabled: true
  metadata_output_path: logs/round5c_21_ratio050.jsonl
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

  persistent_cache_compress_enabled: false
  persistent_cache_compress_real_enabled: false

  compacted_kv_enabled: true
  compacted_kv_real_enabled: true
  compacted_kv_target_ratio: 0.5
  compacted_kv_strategy: uniform
  compacted_kv_min_history_tokens: 4680
  compacted_kv_apply_to_denoise: true
  compacted_kv_apply_to_current_only: false
  compacted_kv_apply_to_clean_cache_update: false
  compacted_kv_protect_sink: true
  compacted_kv_protect_current: true
  compacted_kv_debug_verify: true
  compacted_kv_log_events: true
  compacted_kv_max_events: null

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0

  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
```

创建 ratio075 config：

```bash
cp configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml configs/rolling_forcing_dmd_flowcache_round5c_ratio075.yaml
python - <<'PY'
from pathlib import Path
p = Path("configs/rolling_forcing_dmd_flowcache_round5c_ratio075.yaml")
s = p.read_text()
s = s.replace("metadata_output_path: logs/round5c_21_ratio050.jsonl", "metadata_output_path: logs/round5c_21_ratio075.jsonl")
s = s.replace("compacted_kv_target_ratio: 0.5", "compacted_kv_target_ratio: 0.75")
p.write_text(s)
PY
```

第一阶段，按顺序跑：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5c_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5c_disabled_smoke.log
```

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5c_21_ratio050 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5c_21_ratio050.log
```

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5c_ratio075.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5c_21_ratio075 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5c_21_ratio075.log
```

第一阶段检查：

```bash
grep -n "\[FlowCache\]" logs/round5c_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[compacted_kv_summary\]" logs/round5c_21_ratio050.log logs/round5c_21_ratio075.log
grep -n "\[FlowCache\]\[warning\]" logs/round5c_21_ratio050.log logs/round5c_21_ratio075.log | head -n 80
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round5c_*.log
wc -l logs/round5c_21_ratio050.jsonl logs/round5c_21_ratio075.jsonl
ls -lh videos/round5c_disabled_smoke videos/round5c_21_ratio050 videos/round5c_21_ratio075
```

JSONL 检查：

```bash
python - <<'PY'
import json
from pathlib import Path
for path in [Path("logs/round5c_21_ratio050.jsonl"), Path("logs/round5c_21_ratio075.jsonl")]:
    parse_errors = tensor_like = formula_errors = 0
    summaries = []
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        if "tensor(" in raw:
            tensor_like += 1
        try:
            obj = json.loads(raw)
        except Exception:
            parse_errors += 1
            continue
        rows.append(obj)
        if obj.get("event") == "compacted_kv_summary":
            summaries.append(obj)
        if obj.get("event") == "compacted_kv":
            ov = obj.get("original_visible_tokens", 0)
            cv = obj.get("compacted_visible_tokens", 0)
            sv = obj.get("saved_visible_tokens", 0)
            if ov - cv != sv or cv > ov or sv < 0:
                formula_errors += 1
            if obj.get("compacted_history_tokens", 0) > obj.get("original_history_tokens", 0):
                formula_errors += 1
    summary = summaries[-1] if summaries else {}
    print(path)
    print("parse_errors", parse_errors)
    print("tensor_like", tensor_like)
    print("formula_errors", formula_errors)
    print("total_events", summary.get("total_events"))
    print("applied_events", summary.get("applied_events"))
    print("fallback_count", summary.get("fallback_count"))
    print("warning_count", summary.get("warning_count"))
    print("current_only_applied", summary.get("current_only_applied"))
    print("clean_cache_update_applied", summary.get("clean_cache_update_applied"))
    print("overall_weighted_visible_saving_ratio", summary.get("overall_weighted_visible_saving_ratio"))
PY
```

如果第一阶段通过，将 config 的 JSONL 路径切到 81-frame：

```bash
python - <<'PY'
from pathlib import Path
repls = {
    "configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml": (
        "logs/round5c_21_ratio050.jsonl", "logs/round5c_81_ratio050.jsonl"),
    "configs/rolling_forcing_dmd_flowcache_round5c_ratio075.yaml": (
        "logs/round5c_21_ratio075.jsonl", "logs/round5c_81_ratio075.jsonl"),
}
for file_name, (old, new) in repls.items():
    p = Path(file_name)
    p.write_text(p.read_text().replace(old, new))
PY
```

第二阶段可以并行跑：

```bash
CUDA_VISIBLE_DEVICES=5 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round5c_81_baseline \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5c_81_baseline.log
```

```bash
CUDA_VISIBLE_DEVICES=6 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round5c_81_ratio050 \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5c_81_ratio050.log
```

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5c_ratio075.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round5c_81_ratio075 \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5c_81_ratio075.log
```

第二阶段汇总：

```bash
python scripts/summarize_round5c.py | tee logs/round5c_summary_stdout.log
grep -n "\[FlowCache\]\[compacted_kv_summary\]" logs/round5c_81_ratio050.log logs/round5c_81_ratio075.log
grep -n "\[FlowCache\]\[warning\]" logs/round5c_81_ratio050.log logs/round5c_81_ratio075.log | head -n 80
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round5c_81_*.log
ls -lh videos/round5c_81_baseline videos/round5c_81_ratio050 videos/round5c_81_ratio075
```

## 实验先后顺序

第一阶段必须按顺序执行：`py_compile`、21-frame disabled smoke、21-frame ratio050、21-frame ratio075。只有第一阶段通过，才进入第二阶段。

## 哪些实验可以并行

第二阶段的 81-frame baseline、ratio050、ratio075 可以分配到不同 GPU 并行跑。第一阶段不要并行，便于快速定位 smoke 问题。

## 成功标准

21-frame 成功标准：

1. disabled smoke 生成视频；
2. disabled 日志没有 `[FlowCache]`；
3. ratio050 / ratio075 生成视频；
4. ratio050 / ratio075 有 `[FlowCache][compacted_kv_summary]`；
5. `warning_count=0`；
6. `fallback_count=0` 或很少且原因清楚；
7. `current_only_applied=0`；
8. `clean_cache_update_applied=0`；
9. `compacted_history_tokens <= original_history_tokens`；
10. `saved_visible_tokens >= 0`；
11. JSONL `parse_errors=0`；
12. `tensor_like=0`；
13. `formula_errors=0`；
14. 无 Traceback / RuntimeError / OOM；
15. 没有改训练。

81-frame 成功标准：

1. baseline / ratio050 / ratio075 都生成 3 个视频；
2. baseline 无 FlowCache；
3. ratio050 / ratio075 日志干净；
4. `warning_count=0`；
5. `fallback_count=0` 或很少且原因清楚；
6. `current_only_applied=0`；
7. `clean_cache_update_applied=0`；
8. 记录 runtime；
9. 记录 peak allocated / reserved；
10. 记录 visible saving；
11. 对比 baseline / 5B.2 的 runtime 和 memory；
12. 判断 5C 是否比 5B 更接近性能收益路径。

## 需要回传

第一阶段如果失败，只回传：

1. `logs/round5c_disabled_smoke.log`
2. `logs/round5c_21_ratio050.log`
3. `logs/round5c_21_ratio075.log`
4. `logs/round5c_21_ratio050.jsonl`
5. `logs/round5c_21_ratio075.jsonl`
6. `configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml`
7. `configs/rolling_forcing_dmd_flowcache_round5c_ratio075.yaml`
8. `docs/round5c_compacted_kv_buffer.md`
9. `scripts/summarize_round5c.py`
10. 三个视频目录 `ls -lh` 输出
11. summary grep 输出
12. warning/error grep 输出
13. JSONL 检查输出

第一阶段通过并跑了第二阶段，则回传：

1. `logs/round5c_81_baseline.log`
2. `logs/round5c_81_ratio050.log`
3. `logs/round5c_81_ratio075.log`
4. `logs/round5c_81_ratio050.jsonl`
5. `logs/round5c_81_ratio075.jsonl`
6. `logs/round5c_summary.txt`
7. `logs/round5c_summary.json`
8. `configs/rolling_forcing_dmd_flowcache_round5c_ratio050.yaml`
9. `configs/rolling_forcing_dmd_flowcache_round5c_ratio075.yaml`
10. `docs/round5c_compacted_kv_buffer.md`
11. `scripts/summarize_round5c.py`
12. `videos/round5c_81_baseline` 的 `ls -lh` 输出
13. `videos/round5c_81_ratio050` 的 `ls -lh` 输出
14. `videos/round5c_81_ratio075` 的 `ls -lh` 输出
15. warning/error grep 输出
16. compacted_kv_summary grep 输出
17. py_compile 输出

## 下一阶段：Round 5D

如果 5C 能稳定生成视频并减少 visible history length，但 runtime / memory 仍无收益，Round 5D 应该考虑真正的 compacted cache allocation 或 attention kernel 侧优化：减少主 cache buffer、减少写入/拷贝、或让 attention 后端避免为未使用历史 token 做额外工作。importance / redundancy scoring 仍应放在 compacted path 稳定之后再做。
