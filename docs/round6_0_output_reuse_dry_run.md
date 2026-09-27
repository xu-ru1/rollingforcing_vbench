# Round 6.0 Output Reuse Dry-Run 诊断

## 1. 本轮做什么

Round 6.0 只做 output reuse / L1rel gating 的 dry-run 诊断。在 RollingForcing inference 的 denoise 输出写回 `output` 之前，对同一个 block 在相邻 rolling timestep 上的输出做只读对比，记录 block/chunk 级别的 relative L1：

```text
l1rel = mean(abs(current_output - previous_output)) / max(mean(abs(previous_output)), 1e-8)
```

每个候选会按阈值 `[0.03, 0.05, 0.08, 0.10]` 统计理论可复用比例，并写入日志和 JSONL。记录只包含 shape、id、标量 metric、threshold、fallback reason，不写 tensor 内容。

## 2. 本轮不做什么

- 不直接复用 denoise 输出。
- 不跳过 generator forward。
- 不改 attention 输出、hidden states、noisy cache 更新语义或最终视频生成结果。
- 不改训练、checkpoint、cross-attn。
- 不新增 KV compression 算法，不继续调 ratio，不继续深挖 sidecar/5C telemetry。
- 不做 importance scoring 或 redundancy scoring。

## 3. 为什么 Round 5 可以关闭

Round 5A/5B/5C 已经把 KV compression/read-path 的关键问题走完：5A 证明 logical read path 正确但不省显存；5B 证明 persistent sidecar 在 21-frame reuse 有效但 81-frame 会被 eviction 打断；5C 证明 compacted KV read buffer 在 21-frame 和 81-frame single-prompt 下干净，没有 warning/fallback/current_only/clean_cache_update 问题。5C 能减少 visible history length，但当前 runtime 约慢 15%，不是性能收益版本。

因此 Round 6.0 转向 output reuse 的可行性诊断，先确认哪些 block/chunk 在相邻 timestep 上足够接近，再决定后续是否值得做真实复用。

## 4. L1rel Dry-Run 如何统计

RollingForcing 中同一个 block 会在多个 window 中以不同 denoise timestep 被访问。当前实现利用已有 `output` buffer：在 `output[:, current_start:current_end] = denoised_pred` 覆盖之前，`output` 里仍保留该 block 上一次 visit 的 denoise 输出。dry-run 只读取这两个 tensor 计算 L1rel，随后原路径照常覆盖。

JSONL 事件为 `output_reuse_dry_run`，至少包含：

- `flowcache_run_id`
- `prompt_idx`
- `sample_idx`
- `window_idx` / `window_index`
- `step_idx` / `timestep`
- `block_idx` / `chunk_idx` / `block_id` / `chunk_id`
- `l1rel`
- `threshold`
- `reusable_by_threshold`
- `tensor_shape`
- `tensor_like=0`
- `fallback` / `fallback_reason` / `warning_reason`

summary 事件为 `output_reuse_summary`，包含 `total_candidates`、`valid_metric_count`、不同 threshold 下的 count/ratio、`min/avg/max l1rel`、`warning_count`、`parse_errors`、runtime 和 CUDA peak memory。

## 5. 为什么本轮不直接复用输出

Output reuse 会直接改变 denoise 轨迹，比 KV read-path 观察更靠近最终生成质量。Round 6.0 先做 dry-run，是为了确认 L1rel 分布、阈值敏感性、长视频稳定性和日志完整性；只有当 21-frame 与 81-frame 都显示足够候选、无 warning/OOM/Traceback，并且 JSONL 统计干净，才进入真实复用实现。

## 6. 服务器实验命令

以下命令从仓库根目录运行。`CUDA_VISIBLE_DEVICES` 可按服务器实际 GPU 调整。

先确保 Round 4.2 的三条 smoke prompt 文件存在；如果服务器已有同名文件，不覆盖：

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

### 第一阶段：必须串行先跑

```bash
python -m py_compile \
  inference.py \
  pipeline/rolling_forcing_inference.py \
  utils/flowcache.py \
  utils/wan_wrapper.py \
  wan/modules/causal_model.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round6_0_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round6_0_disabled_smoke.log
```

```bash
rm -f logs/round6_0_output_reuse_dryrun.jsonl logs/round6_0_dryrun_21.jsonl
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round6_0_dryrun.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round6_0_dryrun_21 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round6_0_dryrun_21.log
mv logs/round6_0_output_reuse_dryrun.jsonl logs/round6_0_dryrun_21.jsonl
```

### 第二阶段：第一阶段通过后可并行

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round6_0_81_baseline \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round6_0_81_baseline.log
```

```bash
rm -f logs/round6_0_output_reuse_dryrun.jsonl logs/round6_0_81_dryrun.jsonl
CUDA_VISIBLE_DEVICES=1 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round6_0_dryrun.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round6_0_81_dryrun \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round6_0_81_dryrun.log
mv logs/round6_0_output_reuse_dryrun.jsonl logs/round6_0_81_dryrun.jsonl
```

如果只有一张 GPU，第二阶段按 baseline、dry-run 串行跑。

## 7. 实验先后顺序

1. `py_compile`
2. 21-frame disabled smoke
3. 21-frame output_reuse_dry_run smoke
4. 81-frame baseline
5. 81-frame output_reuse_dry_run

第一阶段全部通过后，才进入第二阶段。

## 8. 哪些可以并行

第一阶段不并行。第二阶段的 `81-frame baseline` 和 `81-frame output_reuse_dry_run` 可以在不同 GPU 上并行；它们输出目录、日志文件不同，只有 dry-run 会写临时 `logs/round6_0_output_reuse_dryrun.jsonl`，baseline 不写 FlowCache JSONL。

## 9. 成功标准

1. disabled log 中没有 `[FlowCache]`。
2. dry-run 成功生成视频。
3. dry-run 不改变视频输出路径与保存逻辑。
4. JSONL 非空。
5. `tensor_like=0`。
6. `warning_count=0`。
7. 没有 Traceback / RuntimeError / OOM。
8. 有 L1rel 分布。
9. 有不同 threshold 下 theoretical reusable ratio。
10. 没有实现真实 output reuse。
11. 没有改训练、checkpoint、cross-attn。

## 10. 检查与回传

统一 grep：

```bash
grep -n "\[FlowCache\]" logs/round6_0_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[output_reuse_summary\]" logs/round6_0_dryrun_21.log
grep -n "\[FlowCache\]\[output_reuse_summary\]" logs/round6_0_81_dryrun.log
grep -n "\[FlowCache\]\[warning\]" logs/round6_0_dryrun_21.log logs/round6_0_81_dryrun.log | head -n 40
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round6_0_*.log
```

JSONL 统计：

```bash
python - <<'PY'
import json
from collections import Counter

for path in ["logs/round6_0_dryrun_21.jsonl", "logs/round6_0_81_dryrun.jsonl"]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    summaries = [r for r in rows if r.get("event") == "output_reuse_summary"]
    events = [r for r in rows if r.get("event") == "output_reuse_dry_run"]
    tensor_like = sum(1 for r in rows if r.get("tensor_like", 0) != 0)
    reusable = Counter(str(r.get("threshold")) for r in events if r.get("reusable_by_threshold"))
    l1rel = [r["l1rel"] for r in events if r.get("l1rel") is not None]
    print(path)
    print("  rows=", len(rows), "events=", len(events), "summaries=", len(summaries), "tensor_like=", tensor_like)
    print("  l1rel_count=", len(l1rel), "min=", min(l1rel) if l1rel else None, "avg=", sum(l1rel)/len(l1rel) if l1rel else None, "max=", max(l1rel) if l1rel else None)
    print("  reusable=", dict(sorted(reusable.items())))
    if summaries:
        print("  summary=", summaries[-1])
PY
```

视频目录：

```bash
ls -lh videos/round6_0_disabled_smoke
ls -lh videos/round6_0_dryrun_21
ls -lh videos/round6_0_81_baseline
ls -lh videos/round6_0_81_dryrun
```

需要回传：

- `logs/round6_0_disabled_smoke.log`
- `logs/round6_0_dryrun_21.log`
- `logs/round6_0_dryrun_21.jsonl`
- `logs/round6_0_81_baseline.log`
- `logs/round6_0_81_dryrun.log`
- `logs/round6_0_81_dryrun.jsonl`
- `configs/rolling_forcing_dmd_flowcache_round6_0_dryrun.yaml`
- `docs/round6_0_output_reuse_dry_run.md`
- summary grep 输出
- warning/error grep 输出
- JSONL 统计输出
- 视频目录 `ls -lh` 输出
