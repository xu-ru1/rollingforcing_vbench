# Round 7.1 sampled fine model_forward profiling

## 1. Round 7.1 做什么

Round 7.1 只做 model_forward 细粒度 profiling。目标是在不改变生成结果的前提下，把 Round 7.0 中最大的 `model_forward_total` 拆开，判断主要耗时来自 attention、MLP/FFN、block forward、KV cache read/write/eviction，还是仍然主要由 VAE decode / video save 等非 model_forward 阶段主导。

本轮新增配置文件：

```bash
configs/rolling_forcing_dmd_flowcache_round7_1_fine_profile.yaml
```

该配置中 `flowcache.enabled: false`，不启用 Round 5/6 的 KV compression、output reuse、metadata dry-run 等功能，只启用 profiler。

## 2. Round 7.1 不做什么

本轮不做任何算法优化，不实现 output reuse，不实现新的 KV compression，不使用 `torch.compile`，不改 VAE 算法，不改训练逻辑，不改 checkpoint，不改 attention 输出，不改 kv_cache 主体逻辑，也不改 cache eviction 主逻辑。

## 3. Round 7.0 的瓶颈结论

Round 7.0 coarse profiling 已通过，81-frame 结果如下：

- baseline runtime: 198.286s
- profiling runtime: 197.741s
- profiling overhead 约 -0.27%，属于运行波动
- warning_count=0
- baseline 无 `[FlowCache]`
- baseline/profile 各生成 3 个视频
- 无 Traceback / RuntimeError / OOM

主要 phase：

- `denoise_loop_total`: 132.32s, 66.9%
- `model_forward_total`: 129.10s, 65.3%
- `vae_decode_total`: 40.25s, 20.4%
- `clean_cache_update_total`: 25.80s, 13.0%
- `video_save_total`: 20.34s, 10.3%

结论：瓶颈主要在 model_forward / denoise_loop。KV compression 和 output reuse 暂时关闭，不继续小修。

## 4. 为什么只做 sampled fine profiling

完整 layer × step profiling 会在高频路径中写入大量计时事件，可能改变 runtime 结构，也会让 JSONL 膨胀。Round 7.1 因此只采样少量层和步：

```yaml
profiler_sample_layers: [0, 10, 20, 29]
profiler_sample_every_n_steps: 4
```

这能回答“attention 和 MLP 谁更重”这类方向性问题，同时避免把 profiling 本身变成新的瓶颈。

## 5. Profiling 覆盖哪些 phase

粗粒度 phase：

- `total_inference`
- `prompt_total`
- `denoise_loop_total`
- `model_forward_total`
- `clean_cache_update_total`
- `vae_decode_total`
- `video_save_total`

sampled fine phase：

- `block_forward_sampled`
- `attention_forward_sampled`
- `attention_qkv_sampled`
- `attention_compute_sampled`
- `attention_output_sampled`
- `cross_attention_sampled`
- `mlp_forward_sampled`
- `kv_cache_read_sampled`
- `kv_cache_write_sampled`
- `kv_eviction_sampled`

summary 会额外聚合：

- `sampled_phase_time_sec`
- `sampled_phase_count`
- `sampled_avg_ms_by_phase`
- `sampled_layer_breakdown`
- `sampled_step_breakdown`
- `top_bottleneck_phases`
- `next_step_recommendation`

## 6. Profiling 不覆盖哪些 phase

默认不记录每一层、每一步，也不拆 GPU kernel 级别的 flash attention / GEMM / memory copy。`kv_cache_read_sampled` 覆盖安全可包裹的 cache slice/anchor 读取位置；更细的单个 slice、clone、rope 子步骤本轮不拆。

如果某个 sampled phase 没有出现，优先检查该 phase 是否在采样层和采样 window 内触发，而不是直接认为该阶段不存在。

## 7. 为什么默认不启用 CUDA synchronize

默认 `profiler_cuda_synchronize: false`。这是为了避免在 attention、block、MLP 等高频位置强制同步，改变真实推理行为。

因此默认 timing 是 coarse/relative wall-time，不是严格 CUDA kernel 时间。若后续需要严格 GPU kernel 级别诊断，可以单独开 `profiler_cuda_synchronize: true` 跑很小样本，但不能和默认 baseline runtime 直接比较。

## 8. 服务器命令

进入环境：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
mkdir -p logs videos docs scripts
```

准备 prompt：

```bash
if [ ! -f logs/round4_2_prompts.txt ]; then
  cat > logs/round4_2_prompts.txt <<'EOF'
A cinematic shot of a red sports car driving through rain at night.
A corgi wearing sunglasses rides a skateboard through a sunny park.
A wide aerial view of snowy mountains under a golden sunrise.
EOF
fi
head -n 1 logs/round4_2_prompts.txt > logs/round7_1_single_prompt.txt
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
  scripts/summarize_round7_1_profile.py 2>&1 | tee logs/round7_1_py_compile.log
```

2. 21-frame disabled smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round7_1_single_prompt.txt \
  --output_folder videos/round7_1_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_1_disabled_smoke.log
```

3. 21-frame fine profiler smoke：

```bash
: > logs/round7_1_21_profile.jsonl
FLOWCACHE_PROFILER_OUTPUT_PATH=logs/round7_1_21_profile.jsonl \
FLOWCACHE_PROFILER_SUMMARY_PATH=logs/round7_1_21_profile_summary.json \
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round7_1_fine_profile.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round7_1_single_prompt.txt \
  --output_folder videos/round7_1_21_profile \
  --num_output_frames 21 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_1_21_profile.log
```

第一阶段检查：

```bash
grep -n "\[FlowCache\]" logs/round7_1_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[profiler_summary\]" logs/round7_1_21_profile.log
grep -n "\[FlowCache\]\[warning\]" logs/round7_1_21_profile.log | head -n 40
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round7_1_disabled_smoke.log logs/round7_1_21_profile.log
ls -lh videos/round7_1_disabled_smoke | tee logs/round7_1_disabled_smoke_ls.txt
ls -lh videos/round7_1_21_profile | tee logs/round7_1_21_profile_ls.txt
python - <<'PY' | tee logs/round7_1_21_profile_jsonl_check.txt
import json
path = "logs/round7_1_21_profile.jsonl"
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

第一阶段通过后，跑 81-frame single-prompt fine profiler：

```bash
: > logs/round7_1_81_single_profile.jsonl
FLOWCACHE_PROFILER_OUTPUT_PATH=logs/round7_1_81_single_profile.jsonl \
FLOWCACHE_PROFILER_SUMMARY_PATH=logs/round7_1_81_single_profile_summary.json \
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round7_1_fine_profile.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round7_1_single_prompt.txt \
  --output_folder videos/round7_1_81_single_profile \
  --num_output_frames 81 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_1_81_single_profile.log
```

生成 Round 7.1 report：

```bash
python scripts/summarize_round7_1_profile.py 2>&1 | tee logs/round7_1_profile_report_stdout.log
```

如果 81 single 正常，再考虑跑 81-frame 3-prompt fine profiler，不强制立即跑：

```bash
: > logs/round7_1_81_3prompt_profile.jsonl
FLOWCACHE_PROFILER_OUTPUT_PATH=logs/round7_1_81_3prompt_profile.jsonl \
FLOWCACHE_PROFILER_SUMMARY_PATH=logs/round7_1_81_3prompt_profile_summary.json \
CUDA_VISIBLE_DEVICES=1 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round7_1_fine_profile.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/round4_2_prompts.txt \
  --output_folder videos/round7_1_81_3prompt_profile \
  --num_output_frames 81 \
  --num_samples 1 \
  --save_with_index \
  --eval_metrics 2>&1 | tee logs/round7_1_81_3prompt_profile.log
```

## 9. 实验先后顺序

第一阶段串行：

1. py_compile
2. 21-frame disabled smoke
3. 21-frame fine profiler smoke
4. JSONL / warning / error / video 检查

第一阶段通过后：

1. 81-frame single-prompt fine profiler
2. `scripts/summarize_round7_1_profile.py`
3. 根据结果决定是否跑 81-frame 3-prompt fine profiler

## 10. 哪些实验可以并行

第一阶段不要并行，它是安全闸门。

第二阶段当前只强制跑 81 single，因此不需要并行。可选的 81-frame 3-prompt fine profiler 可以在 81 single 通过后单独使用另一张 GPU 跑，但不要和第一阶段并行。

## 11. 成功标准

1. disabled 无 `[FlowCache]`。
2. profiler smoke 成功生成视频。
3. profiler JSONL 非空。
4. JSONL `parse_errors=0`。
5. JSONL `tensor_like=0`。
6. `warning_count=0` 或很少且可解释。
7. 无 Traceback / RuntimeError / OOM。
8. summary 有 attention / MLP / block，或明确说明无法安全采集。
9. 不改变生成结果。
10. 不改训练。
11. 不改 checkpoint。
12. 不改 attention 输出。
13. 不改 kv_cache 主体逻辑。

## 12. 需要回传哪些文件/日志

如果第一阶段失败，回传：

1. `logs/round7_1_disabled_smoke.log`
2. `logs/round7_1_21_profile.log`
3. `logs/round7_1_21_profile.jsonl`
4. `logs/round7_1_21_profile_summary.json`
5. `configs/rolling_forcing_dmd_flowcache_round7_1_fine_profile.yaml`
6. `docs/round7_1_fine_model_forward_profiling.md`
7. `scripts/summarize_round7_1_profile.py`
8. 视频目录 `ls -lh` 输出
9. warning/error grep 输出
10. JSONL 检查输出

如果第二阶段完成，回传：

1. `logs/round7_1_81_single_profile.log`
2. `logs/round7_1_81_single_profile.jsonl`
3. `logs/round7_1_81_single_profile_summary.json`
4. `logs/round7_1_profile_report.txt`
5. `logs/round7_1_profile_report.json`
6. `configs/rolling_forcing_dmd_flowcache_round7_1_fine_profile.yaml`
7. `docs/round7_1_fine_model_forward_profiling.md`
8. `scripts/summarize_round7_1_profile.py`
9. `videos/round7_1_81_single_profile` 的 `ls -lh` 输出
10. warning/error grep 输出
11. profiler summary grep 输出
12. py_compile 输出

## 13. 如何根据结果决定 Round 8

如果 `attention_forward_sampled` / `attention_compute_sampled` / `cross_attention_sampled` 在 sampled block 中占主导，Round 8 转向 attention profiling 和 attention kernel / mask / memory layout 诊断。

如果 `mlp_forward_sampled` 占主导，Round 8 转向 MLP/FFN profiling，重点看 linear/GELU 和 block scheduling。

如果 `kv_cache_read_sampled` / `kv_cache_write_sampled` / `kv_eviction_sampled` 占比稳定较高，再考虑回到 KV/cache 方向；否则继续关闭 Round 5 KV compression 路线。

如果 sampled model forward 内部没有明显单点，但 `vae_decode_total` 仍接近 20%，Round 8 可以转向 VAE decode。

如果 `video_save_total` 接近或超过 10%，且目标是端到端 wall time，Round 8 可以转向 video save / CPU transfer / I-O 路径。

如果所有 sampled fine phase 都显示很低或噪声很大，先增加 sampled layers 或单独打开 CUDA synchronize 的极小样本诊断，不直接做优化。
