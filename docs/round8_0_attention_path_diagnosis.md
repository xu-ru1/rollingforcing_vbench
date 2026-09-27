# Round 8.0 Self-Attention Path Diagnosis

## 1. 本轮目标

Round 8.0 只做 self-attention path diagnosis，不做算法优化。目标是把 `CausalWanSelfAttention.forward` 的耗时拆到更细的阶段，确认 Round 7.1 里 `attention_forward_sampled` 偏高到底来自 attention kernel 本体，还是来自 RoPE、QKV projection、KV cache assembly、padding/mask/cu_seqlens、output projection、KV write 或 clean cache update attention path。

本轮不改训练、不改 checkpoint、不改 attention 输出、不改 KV cache 主体逻辑、不改 eviction、不做新的 KV compression、不做 output reuse、不换 kernel、不启用 `torch.compile`。

## 2. 先读这个：避免 shell 报错

这份文档所有可运行命令都从服务器仓库根目录执行：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
```

不要把下面这种参数单独贴到 shell 里：

```bash
--output_folder videos/xxx
```

如果看到：

```text
--output_folder: command not found
```

说明上一条多行命令被复制断了，shell 把 `--output_folder` 当成了一条新命令。为避免这个问题，推荐直接用脚本：

```bash
bash scripts/run_round8_0_attention_profile.sh smoke
```

脚本内部已经写死 `--data_path` 和 `--output_folder`，不会再出现 prompt 文件缺失或 output folder 为 `None` 的问题。

前面的 `torch.load(... weights_only=False)` 是 `FutureWarning`，不是本轮失败原因。真正需要先排除的是 `FileNotFoundError`、`--output_folder: command not found`、`TypeError: output_folder None`、`RuntimeError`、OOM。

## 3. 推荐运行方式：脚本

先进入环境：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

如果你的 checkpoint 是默认文件，直接跑：

```bash
bash scripts/run_round8_0_attention_profile.sh smoke
```

`smoke` 会依次执行：

1. `py_compile`
2. 自动创建 `logs/round8_0_single_prompt.txt`
3. 自动创建 `logs/round8_0_3prompts.txt`
4. 21-frame disabled smoke
5. 21-frame attention profiler smoke
6. grep warning/error
7. 检查 JSONL parse errors 和 tensor-like 内容
8. 记录视频目录 `ls -lh`

如果服务器 checkpoint 不是 `checkpoints/rolling_forcing_dmd.pt`，用环境变量覆盖：

```bash
CKPT=checkpoints/ode_init.pt EMA_FLAG= bash scripts/run_round8_0_attention_profile.sh smoke
```

如果要指定 GPU：

```bash
GPU=0 bash scripts/run_round8_0_attention_profile.sh smoke
```

21-frame smoke 全部通过后，再跑 81-frame single-prompt：

```bash
GPU=0 bash scripts/run_round8_0_attention_profile.sh 81
```

81-frame single-prompt 正常后，才考虑 81-frame 3-prompt：

```bash
GPU_3PROMPT=1 bash scripts/run_round8_0_attention_profile.sh 3prompt
```

只重新生成汇总报告：

```bash
bash scripts/run_round8_0_attention_profile.sh summary
```

## 4. 手动命令：只在不用脚本时复制

下面是脚本的等价手动命令。每条 `python inference.py` 都是一整行，故意不使用反斜杠续行，避免复制断行后出现 `--output_folder: command not found`。

### 4.1 准备目录和 prompt

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
mkdir -p logs videos
cat > logs/round8_0_single_prompt.txt <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
EOF
cat > logs/round8_0_3prompts.txt <<'EOF'
A calm lake at sunrise, cinematic, gentle camera movement.
A futuristic city street at night, neon lights, slow camera pan.
A small dog running through a flower field, bright daylight, smooth motion.
EOF
```

### 4.2 必要文件检查

```bash
test -f inference.py
test -f configs/rolling_forcing_dmd.yaml
test -f configs/rolling_forcing_dmd_flowcache_round8_0_attention_profile.yaml
test -f checkpoints/rolling_forcing_dmd.pt
test -f logs/round8_0_single_prompt.txt
test -f logs/round8_0_3prompts.txt
```

如果 checkpoint 不是 `checkpoints/rolling_forcing_dmd.pt`，把下面命令中的 `CKPT=...` 改成实际路径。

```bash
GPU=0
CKPT=checkpoints/rolling_forcing_dmd.pt
EMA_FLAG=--use_ema
```

如果使用 `checkpoints/ode_init.pt` 且不需要 EMA：

```bash
GPU=0
CKPT=checkpoints/ode_init.pt
EMA_FLAG=
```

### 4.3 py_compile

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py scripts/summarize_round8_0_attention_profile.py 2>&1 | tee logs/round8_0_py_compile.log
```

### 4.4 21-frame disabled smoke

```bash
FLOWCACHE_ATTENTION_PROFILER_ENABLED=false CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path configs/rolling_forcing_dmd.yaml --checkpoint_path "$CKPT" --data_path logs/round8_0_single_prompt.txt --output_folder videos/round8_0_disabled_smoke --num_output_frames 21 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics 2>&1 | tee logs/round8_0_disabled_smoke.log
```

### 4.5 21-frame attention profiler smoke

```bash
: > logs/round8_0_21_attention.jsonl
FLOWCACHE_ATTENTION_PROFILER_OUTPUT_PATH=logs/round8_0_21_attention.jsonl FLOWCACHE_ATTENTION_PROFILER_SUMMARY_PATH=logs/round8_0_21_attention_summary.json CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path configs/rolling_forcing_dmd_flowcache_round8_0_attention_profile.yaml --checkpoint_path "$CKPT" --data_path logs/round8_0_single_prompt.txt --output_folder videos/round8_0_21_attention --num_output_frames 21 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics 2>&1 | tee logs/round8_0_21_attention.log
```

### 4.6 第一阶段检查

```bash
grep -n "\[FlowCache\]" logs/round8_0_disabled_smoke.log | head -n 40 || true
grep -n "\[FlowCache\]\[attention_profiler_summary\]" logs/round8_0_21_attention.log || true
grep -n "\[FlowCache\]\[warning\]" logs/round8_0_21_attention.log | head -n 40 || true
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round8_0_disabled_smoke.log logs/round8_0_21_attention.log || true
ls -lh videos/round8_0_disabled_smoke | tee logs/round8_0_disabled_smoke_ls.txt
ls -lh videos/round8_0_21_attention | tee logs/round8_0_21_attention_ls.txt
python - <<'PY' | tee logs/round8_0_21_attention_jsonl_check.txt
import json
path = "logs/round8_0_21_attention.jsonl"
total = parse_errors = tensor_like = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        if "tensor(" in line or "Tensor" in line or "<tensor_like" in line:
            tensor_like += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
print({"path": path, "total": total, "parse_errors": parse_errors, "tensor_like": tensor_like})
PY
```

### 4.7 81-frame single-prompt attention profiler

只在 21-frame disabled smoke 和 21-frame attention profiler 都通过后运行。

```bash
: > logs/round8_0_81_single_attention.jsonl
FLOWCACHE_ATTENTION_PROFILER_OUTPUT_PATH=logs/round8_0_81_single_attention.jsonl FLOWCACHE_ATTENTION_PROFILER_SUMMARY_PATH=logs/round8_0_81_single_attention_summary.json CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path configs/rolling_forcing_dmd_flowcache_round8_0_attention_profile.yaml --checkpoint_path "$CKPT" --data_path logs/round8_0_single_prompt.txt --output_folder videos/round8_0_81_single_attention --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics 2>&1 | tee logs/round8_0_81_single_attention.log
```

### 4.8 81-frame 3-prompt attention profiler

只在 81-frame single-prompt 稳定后运行。

```bash
: > logs/round8_0_81_3prompt_attention.jsonl
FLOWCACHE_ATTENTION_PROFILER_OUTPUT_PATH=logs/round8_0_81_3prompt_attention.jsonl FLOWCACHE_ATTENTION_PROFILER_SUMMARY_PATH=logs/round8_0_81_3prompt_attention_summary.json CUDA_VISIBLE_DEVICES="$GPU" python inference.py --config_path configs/rolling_forcing_dmd_flowcache_round8_0_attention_profile.yaml --checkpoint_path "$CKPT" --data_path logs/round8_0_3prompts.txt --output_folder videos/round8_0_81_3prompt_attention --num_output_frames 81 --num_samples 1 $EMA_FLAG --save_with_index --eval_metrics 2>&1 | tee logs/round8_0_81_3prompt_attention.log
```

### 4.9 生成汇总报告

```bash
python scripts/summarize_round8_0_attention_profile.py 2>&1 | tee logs/round8_0_attention_report_stdout.log
```

## 5. 成功标准

1. `logs/round8_0_disabled_smoke.log` 中没有异常 Traceback/OOM。
2. disabled smoke 生成视频。
3. 21-frame attention profiler 生成视频。
4. `logs/round8_0_disabled_smoke.log` 和 `logs/round8_0_21_attention.log` 中都能看到 `[Inference] Loaded checkpoint weights`。
5. `logs/round8_0_21_attention.jsonl` 非空。
6. JSONL check 中 `parse_errors=0`。
7. JSONL check 中 `tensor_like=0`。
8. summary 中能看到 attention phase breakdown。
9. 所有 `python inference.py` 命令都带有 `--data_path`、`--output_folder` 和 `--checkpoint_path`。
10. 81-frame single-prompt 只在 21-frame smoke 通过后运行。
11. 81-frame 3-prompt 只在 81-frame single-prompt 稳定后运行。

## 6. 需要回传的文件

第一阶段失败时回传：

- `logs/round8_0_py_compile.log`
- `logs/round8_0_disabled_smoke.log`
- `logs/round8_0_21_attention.log`
- `logs/round8_0_21_attention.jsonl`
- `logs/round8_0_21_attention_summary.json`
- `logs/round8_0_21_attention_jsonl_check.txt`
- `logs/round8_0_disabled_smoke_ls.txt`
- `logs/round8_0_21_attention_ls.txt`
- `docs/round8_0_attention_path_diagnosis.md`
- `scripts/run_round8_0_attention_profile.sh`
- `scripts/summarize_round8_0_attention_profile.py`
- `configs/rolling_forcing_dmd_flowcache_round8_0_attention_profile.yaml`

81-frame 完成后回传：

- `logs/round8_0_81_single_attention.log`
- `logs/round8_0_81_single_attention.jsonl`
- `logs/round8_0_81_single_attention_summary.json`
- `logs/round8_0_attention_report.txt`
- `logs/round8_0_attention_report.json`
- `videos/round8_0_81_single_attention` 的 `ls -lh` 输出

## 7. 根据结果决定 Round 8.1

如果 `attention_kernel` 是稳定 top subphase，并且 flash-attn 已确认使用，Round 8.1 才考虑 kernel/path 级优化。如果 `kv_cache_read_or_assembly`、`padding_or_mask_setup`、RoPE、QKV 或 output projection 更高，Round 8.1 优先拆对应数据准备或 layout 路径。如果 attention subphase 都不是主因，而 VAE decode 或 video save 仍占端到端高比例，就转向 VAE decode / video save，不回到 KV compression ratio 微调。
