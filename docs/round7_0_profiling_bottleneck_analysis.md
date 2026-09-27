# Round 7.0 profiling-driven bottleneck analysis

## 1. 本轮做什么

Round 7.0 只做 profiling-driven bottleneck analysis：用默认关闭、低风险的 timing 统计回答 81-frame RollingForcing inference 的时间主要花在哪里。

本轮新增两层 profiling：

1. coarse profiling：默认推荐使用，覆盖总耗时、单 prompt 总耗时、denoise loop、model forward、clean cache update、VAE decode、video save、Python pipeline overhead、FlowCache logging overhead、eval metrics 等阶段。
2. sampled fine profiling：代码支持但默认不启用，只有 `profiler_mode: fine` / `sampled_fine` / `detailed` / `full` / `all` 时才记录 attention/block/FFN/KV write/eviction 等细粒度阶段。

## 2. 本轮不做什么

本轮不做 KV compression 新算法、不修 sidecar reuse、不优化 compacted KV、不做真实 output reuse、不做 L1rel threshold 真实复用，也不做 importance/redundancy scoring。

代码要求上，本轮不改变训练逻辑、不改变 checkpoint、不改变 attention 输出、不改变 kv_cache 主体算法、不改变 eviction 策略、不改变 baseline 默认配置路径。

## 3. 为什么 Round 5 / Round 6 可以关闭

Round 5 已证明 KV compression/read-path 在工程上可行，但当前不是性能收益版本：5A 正确但不省显存，5B reuse 被 eviction 打断，5C 能减少 visible history length 但 runtime 仍慢约 15%。

Round 6 已证明 output reuse / L1rel dry-run 链路正确且 dry-run 不改变生成结果，但 81-frame 的最小 l1rel 约 0.115，threshold <= 0.10 时 reusable ratio 为 0，threshold 0.20 后才出现大量候选，质量风险较高。

所以 Round 7 先关闭这些方向，把问题收敛成：真实 81-frame inference 的时间瓶颈到底在哪里。

## 4. Profiling 设计

新增配置在 `flowcache` 下，默认配置中 `profiler_enabled: false`。Round 7 profile 配置文件为：

```yaml
configs/rolling_forcing_dmd_flowcache_round7_0_profile.yaml
```

该配置里 `flowcache.enabled: false`，只打开 `profiler_enabled: true`。这样不会触发 Round 5/6 的 compression、metadata、dry-run 或 output reuse 逻辑，只记录 timing。

JSONL event 至少包含：

- `event`
- `flowcache_run_id`
- `prompt_idx`
- `sample_idx`
- `phase`
- `elapsed_sec` / `elapsed_ms`
- `count`
- `mode`
- `cuda_synchronize`
- `device`
- `warning_count`
- `fallback_reason`

summary 至少包含：

- `total_runtime_sec`
- `prompt_count`
- `sample_count`
- `saved_videos`
- `phase_time_sec`
- `phase_time_ratio`
- `event_count_by_phase`
- `warning_count`
- `profiler_overhead_estimate`
- `peak_cuda_allocated_gb`
- `peak_cuda_reserved_gb`
- config mode 和 `cuda_synchronize`

JSONL 不写 tensor 内容。summary 聚合 phase time，stdout 只打印 `[FlowCache][profiler]` 和 `[FlowCache][profiler_summary]`，避免 per-layer 刷屏。

## 5. Coarse 和 fine 的区别

coarse profiling 默认启用，推荐用于 Round 7.0 正式 81-frame 结果。它低风险、输出量小，主要回答：

- `total_inference`
- `prompt_total`
- `text_encode_total`
- `denoise_loop_total`
- `model_forward_total`
- `clean_cache_update_total`
- `noisy_cache_update_total`
- `kv_cache_init_total`
- `vae_decode_total`
- `video_save_total`
- `python_pipeline_overhead`
- `flowcache_logging_total`
- `eval_metrics_total`

fine profiling 默认不启用。只有显式修改 `profiler_mode` 后才采样：

- `attention_forward_total`
- `attention_qkv_or_cache_read_total`
- `attention_compute_total`
- `attention_output_total`
- `cross_attention_total`
- `mlp_or_ffn_total`
- `block_forward_total`
- `kv_cache_write_total`
- `kv_eviction_total`

fine profiling 的计时包在模块 forward 周围，不改变输入输出 tensor。注意默认不强制 CUDA synchronize，因此 fine 默认更适合看 CPU launch/调度侧趋势；若要 GPU 精确阶段耗时，需要显式开启 `profiler_cuda_synchronize: true`，但这会明显改变 runtime。

## 6. 为什么默认不强制 CUDA synchronize

高频位置强制 `torch.cuda.synchronize()` 会改变推理时间结构，尤其是 attention/block/FFN 这种频繁调用点。Round 7 默认使用 CPU wall timer，保持最低风险和较小扰动。

如果后续需要精确 GPU kernel 时间，可以在单独 fine run 中显式设置 `profiler_cuda_synchronize: true`，并把它当作“诊断实验”，不要和 baseline runtime 直接混为优化收益。

## 7. 服务器实验命令

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

准备目录和 21-frame smoke prompt：

```bash
mkdir -p logs videos docs scripts
if [ ! -f logs/round4_2_prompts.txt ]; then
  cat > logs/round4_2_prompts.txt <<'EOF'
A cinematic shot of a red sports car driving through rain at night.
A corgi wearing sunglasses rides a skateboard through a sunny park.
A wide aerial view of snowy mountains under a golden sunrise.
EOF
fi
head -n 1 logs/round4_2_prompts.txt > logs/round7_0_smoke_prompt.txt
```

第一阶段必须按顺序跑。

1. py_compile：

```bash
python -m py_compile \
  inference.py \
  pipeline/rolling_forcing_inference.py \
  utils/flowcache.py \
  utils/wan_wrapper.py \
  wan/modules/causal_model.py \
  scripts/summarize_round7_profile.py 2>&1 | tee logs/round7_py_compile.log
```

2. 21-frame disabled smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round7_0_smoke_prompt.txt \
  --output_folder videos/round7_0_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_0_disabled_smoke.log
```

3. 21-frame profiler smoke：

```bash
: > logs/round7_0_profile_21.jsonl
FLOWCACHE_PROFILER_OUTPUT_PATH=logs/round7_0_profile_21.jsonl \
FLOWCACHE_PROFILER_SUMMARY_PATH=logs/round7_0_profile_21_summary.json \
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round7_0_profile.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round7_0_smoke_prompt.txt \
  --output_folder videos/round7_0_profile_21 \
  --num_output_frames 21 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_0_profile_21.log
```

第一阶段检查：

```bash
grep -n "\[FlowCache\]" logs/round7_0_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[profiler_summary\]" logs/round7_0_profile_21.log
grep -n "\[FlowCache\]\[warning\]" logs/round7_0_profile_21.log | head -n 40
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round7_0_disabled_smoke.log logs/round7_0_profile_21.log
ls -lh videos/round7_0_disabled_smoke
ls -lh videos/round7_0_profile_21
python - <<'PY' | tee logs/round7_0_profile_21_jsonl_check.txt
import json
path = "logs/round7_0_profile_21.jsonl"
parse_errors = 0
tensor_like = 0
total = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        if "tensor(" in line or "Tensor" in line or "<tensor_like" in line:
            tensor_like += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
print({"total": total, "parse_errors": parse_errors, "tensor_like": tensor_like})
PY
```

如果第一阶段出现以下任一情况，不跑第二阶段：

1. disabled smoke 有 `[FlowCache]`。
2. profiler smoke 有 Traceback / RuntimeError / OOM。
3. `logs/round7_0_profile_21.jsonl` 为空。
4. JSONL `parse_errors > 0`。
5. JSONL `tensor_like > 0`。
6. 视频失败。
7. baseline 默认路径被影响。

第二阶段在第一阶段通过后再跑。baseline 和 profiler coarse 可以在不同 GPU 上并行：

81-frame baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round7_81_baseline \
  --num_output_frames 81 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_81_baseline.log
```

81-frame profiler coarse：

```bash
: > logs/round7_81_profile.jsonl
FLOWCACHE_PROFILER_OUTPUT_PATH=logs/round7_81_profile.jsonl \
FLOWCACHE_PROFILER_SUMMARY_PATH=logs/round7_81_profile_summary.json \
CUDA_VISIBLE_DEVICES=1 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round7_0_profile.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round7_81_profile \
  --num_output_frames 81 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_81_profile.log
```

汇总报告：

```bash
python scripts/summarize_round7_profile.py 2>&1 | tee logs/round7_profile_report_stdout.log
```

第二阶段检查：

```bash
grep -n "\[FlowCache\]" logs/round7_81_baseline.log | head -n 40
grep -n "\[FlowCache\]\[profiler_summary\]" logs/round7_81_profile.log
grep -n "\[FlowCache\]\[warning\]" logs/round7_81_profile.log | head -n 40
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round7_81_baseline.log logs/round7_81_profile.log
ls -lh videos/round7_81_baseline
ls -lh videos/round7_81_profile
python - <<'PY' | tee logs/round7_81_profile_jsonl_check.txt
import json
path = "logs/round7_81_profile.jsonl"
parse_errors = 0
tensor_like = 0
total = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        if "tensor(" in line or "Tensor" in line or "<tensor_like" in line:
            tensor_like += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
print({"total": total, "parse_errors": parse_errors, "tensor_like": tensor_like})
PY
```

## 8. 实验先后顺序

第一阶段严格串行：

1. py_compile。
2. 21-frame disabled smoke。
3. 21-frame profiler smoke。
4. 检查 disabled 无 `[FlowCache]`、profile 有 summary、JSONL 可解析且不含 tensor 内容、视频存在。

第二阶段只有在第一阶段通过后执行：

1. 81-frame baseline。
2. 81-frame profiler coarse。
3. 跑 `scripts/summarize_round7_profile.py` 生成报告。

## 9. 哪些实验可以并行

第一阶段不要并行，因为它是安全闸门。

第二阶段中 `81-frame baseline` 和 `81-frame profiler coarse` 可以在不同 GPU 上并行。它们写不同的 logs 和 videos 目录；baseline 不写 FlowCache JSONL，profile 只写 `logs/round7_81_profile.jsonl` 和 `logs/round7_81_profile_summary.json`。

汇总脚本必须等第二阶段两个任务都结束后再跑。

## 10. 成功标准

21-frame 成功标准：

1. disabled smoke 生成视频。
2. disabled 日志没有 `[FlowCache]`。
3. profiler smoke 生成视频。
4. profiler 日志有 `[FlowCache][profiler_summary]`。
5. profiler JSONL 非空。
6. JSONL `parse_errors=0`。
7. JSONL `tensor_like=0`。
8. `warning_count=0` 或很少且可解释。
9. 无 Traceback / RuntimeError / OOM。
10. baseline 默认路径未改变。

81-frame 成功标准：

1. baseline / profiler 都生成 3 个视频。
2. baseline 无 `[FlowCache]`。
3. profiler 有 summary。
4. summary 能给出 phase time。
5. profiler overhead 可计算。
6. peak allocated / reserved 可记录。
7. report 能指出 top bottleneck phases。
8. 无 Traceback / RuntimeError / OOM。
9. 没有算法改动。
10. 没有训练/checkpoint 改动。

## 11. 需要回传哪些文件/日志

如果第一阶段失败，只回传：

1. `logs/round7_0_disabled_smoke.log`
2. `logs/round7_0_profile_21.log`
3. `logs/round7_0_profile_21.jsonl`
4. `logs/round7_0_profile_21_summary.json`
5. `configs/rolling_forcing_dmd_flowcache_round7_0_profile.yaml`
6. `docs/round7_0_profiling_bottleneck_analysis.md`
7. `scripts/summarize_round7_profile.py`
8. 两个视频目录 `ls -lh` 输出
9. warning/error grep 输出
10. JSONL parse/tensor_like 检查输出

如果第一阶段通过并跑了第二阶段，回传：

1. `logs/round7_81_baseline.log`
2. `logs/round7_81_profile.log`
3. `logs/round7_81_profile.jsonl`
4. `logs/round7_81_profile_summary.json`
5. `logs/round7_profile_report.txt`
6. `logs/round7_profile_report.json`
7. `configs/rolling_forcing_dmd_flowcache_round7_0_profile.yaml`
8. `docs/round7_0_profiling_bottleneck_analysis.md`
9. `scripts/summarize_round7_profile.py`
10. `videos/round7_81_baseline` 的 `ls -lh` 输出
11. `videos/round7_81_profile` 的 `ls -lh` 输出
12. warning/error grep 输出
13. profiler summary grep 输出
14. py_compile 输出

## 12. 发现瓶颈后的下一阶段选择

如果瓶颈主要在 `model_forward_total`，下一阶段再打开 sampled fine profiling，看 `attention_forward_total`、`mlp_or_ffn_total`、`block_forward_total` 谁占主导。

如果瓶颈主要在 attention，下一阶段转 attention kernel、attention mask、KV read layout 或 sampled layer 级别诊断。

如果瓶颈主要在 MLP/block，下一阶段转 FFN、block-level scheduling、torch compile/算子层诊断。

如果瓶颈主要在 `vae_decode_total`，下一阶段转 VAE decode batch/chunk、decode cache、precision 或后处理路径。

如果瓶颈主要在 `video_save_total` 或 `python_pipeline_overhead`，下一阶段转视频编码、CPU transfer、I/O、pipeline overlap。

如果 `kv_cache_write_total` / `kv_eviction_total` 占比不高，不建议继续 KV/cache 优化；如果它们占比高，再进入 Round 7.x 的细粒度 KV read/write/eviction profiling。
