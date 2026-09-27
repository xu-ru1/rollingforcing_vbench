# Round 3A Compression Candidate Analysis

Round 3A 只做 compression candidate analysis。它在 Round 2 的 KV metadata tracker 基础上增加 attention 内部只读 hook，用真实 attention 输入长度标记哪些 token 未来可能作为 compression candidate。

## 1. Round 3A 做了什么

| 文件 | 修改内容 | 目的 |
| --- | --- | --- |
| `configs/default_config.yaml` | 新增 `compression_candidate_enabled`、`log_attention_parts`、`log_compression_candidates`，默认关闭 | 控制 Round 3A 的 attention parts 和 candidate 日志 |
| `utils/flowcache.py` | 新增 attention parts、compression candidate 记录和 `[FlowCache][candidate_summary]` | 只保存长度、range、比例、timestamp，不保存 tensor |
| `pipeline/rolling_forcing_inference.py` | 将 `flowcache_manager` 只读透传给两次 generator 调用，并在结束时调用 `summarize_candidates()` | 让 attention 内部能在 debug config 下记录真实 KV 组成 |
| `utils/wan_wrapper.py` | `WanDiffusionWrapper.forward` 新增默认 `None` 的 FlowCache 透传参数 | 默认调用者不受影响 |
| `wan/modules/causal_model.py` | 在 `CausalWanSelfAttention.forward` 的 attention 调用前增加只读 hook | 记录真实 `anchor / working / current / input` token 长度，不改变 attention 输入 |
| `docs/round3a_compression_candidate_analysis.md` | 本文档 | 记录字段、验证方式和下一轮建议 |

同时补齐 Round 2 小问题：JSONL 中现在同时写 `total_visible_kv_tokens` 和兼容 alias `total_kv_tokens`。

## 2. Round 3A 没做什么

1. 没有实现 KV compression。
2. 没有实现 output reuse。
3. 没有实现 L1rel threshold。
4. 没有改变 Q/K/V tensor 内容。
5. 没有改变 attention 输入拼接顺序。
6. 没有改变 attention 计算。
7. 没有改变 cache 写入或 eviction。
8. 没有改变 checkpoint 读取。
9. 没有改变训练逻辑。
10. 没有删除 Self Forcing、DMD、Wan 相关代码。

## 3. Attention Parts 字段

`[FlowCache][attention_parts]` 和 JSONL 中的 `event="attention_parts"` 记录真实 attention 输入组成：

| 字段 | 含义 |
| --- | --- |
| `source_event` | `denoise_attention` 或 `clean_cache_update_attention` |
| `window_index` | rolling forcing window 下标 |
| `layer_idx` | Transformer layer 下标 |
| `attention_branch` | `current_only`、`anchor_working_current` 或 `clean_cache_update` |
| `block_length` | 一个 causal block 的 token 数，当前为 `3 * 1560 = 4680` |
| `sink_tokens` | 代码中的 `sink_tokens = 1 * block_length` |
| `anchor_tokens` / `anchor_value_tokens` | attention sink / anchor cache 的 key/value 长度 |
| `working_tokens` / `working_value_tokens` | history/working cache 的 key/value 长度 |
| `current_tokens` / `current_value_tokens` | current `roped_key` / `v` 的长度 |
| `input_tokens` / `input_value_tokens` | attention 输入 key/value 拼接后的总长度 |
| `total_visible_kv_tokens` | 可见 KV token 总数 |
| `total_kv_tokens` | `total_visible_kv_tokens` 的兼容 alias |
| `current_start` | 当前调用传入的 global token 起点 |
| `cache_start` / `cache_end` | 当前 cache 写入区间或等价范围 |
| `global_end_index` / `local_end_index` | 当前层 KV cache end index |

## 4. Compression Candidate 字段

`[FlowCache][compression_candidate]` 和 JSONL 中的 `event="compression_candidate"` 按保守策略标记候选：

| 字段 | 含义 |
| --- | --- |
| `protected_sink_tokens` | 受保护的 sink/anchor token 数 |
| `protected_current_tokens` | 受保护的 current token 数 |
| `compressible_history_tokens` | 暂定可压缩的 history/working token 数 |
| `total_kv_tokens` | attention 可见 KV token 总数 |
| `candidate_ratio` | `compressible_history_tokens / total_kv_tokens` |
| `candidate_region_start` / `candidate_region_end` | 在 `[anchor, working, current]` 拼接坐标中的候选区间 |

Round 3A 的候选定义：

```text
protected_sink_tokens = anchor_tokens
protected_current_tokens = current_tokens
compressible_history_tokens = working_tokens
candidate_region = [anchor_tokens, anchor_tokens + working_tokens)
candidate_ratio = working_tokens / input_tokens
```

如果某个 branch 没有显式 anchor/current 拼接，hook 会按真实 attention 输入长度保守估计，并通过 `attention_branch` 标注来源。

## 5. 为什么 Sink / Current 必须保护

attention sink 是 causal cache 中长期常驻的 anchor block，用来稳定长时序注意力坐标和早期上下文。压缩它会改变后续 window 对全局起点的参照。

current KV 是当前 denoising window 的新计算内容，直接对应当前 step 的 query/key/value。压缩 current 会立刻改变当前 denoising 的 self-attention 结果，风险最高。

因此 Round 3A 将 sink 和 current 标记为 protected，只把 working/history 区间标为 candidate。

## 6. 为什么本轮只标记 Candidate

当前目标是验证真实 attention 内部长度是否和 Round 2 pipeline-level range 一致。直接压缩会同时引入信息损失、range 映射、mask 兼容和 cache 更新风险，不利于定位问题。Round 3A 只做只读记录，为 Round 3B 的 dry-run 或 prototype 准备可靠候选区间。

## 7. 服务器验证命令

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
mkdir -p logs videos/round3a_disabled_smoke videos/round3a_candidate_smoke
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

新建 Round 3A config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round3a.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round3a.yaml <<'YAML'

flowcache:
  enabled: true
  debug: true
  metadata_enabled: true
  metadata_output_path: logs/round3a_kv_candidates.jsonl

  compression_candidate_enabled: true
  log_attention_parts: true
  log_compression_candidates: true

  output_reuse_enabled: false
  kv_compress_enabled: false
  l1rel_threshold: 0.0
  protect_sink_tokens: true
  log_every_window: true
  log_kv_ranges: true
  log_summary: true
YAML
```

disabled smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round3a_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round3a_disabled_smoke.log
```

candidate debug smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round3a.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round3a_candidate_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round3a_candidate_smoke.log
```

检查结果：

```bash
grep -n "\[FlowCache\]" logs/round3a_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[attention_parts\]" logs/round3a_candidate_smoke.log | head -n 80
grep -n "\[FlowCache\]\[compression_candidate\]" logs/round3a_candidate_smoke.log | head -n 80
grep -n "\[FlowCache\]\[candidate_summary\]" logs/round3a_candidate_smoke.log
grep -n "\[FlowCache\]\[warning\]" logs/round3a_candidate_smoke.log | head -n 40

head -n 30 logs/round3a_kv_candidates.jsonl
wc -l logs/round3a_kv_candidates.jsonl

ls -lh videos/round3a_disabled_smoke
ls -lh videos/round3a_candidate_smoke
```

## 8. Round 3A 成功条件

1. disabled smoke 成功生成视频。
2. candidate debug smoke 成功生成视频。
3. disabled 日志没有 `[FlowCache]`。
4. candidate 日志有 `[FlowCache][attention_parts]`。
5. candidate 日志有 `[FlowCache][compression_candidate]`。
6. candidate 日志有 `[FlowCache][candidate_summary]`。
7. JSONL 文件存在且非空。
8. JSONL 同时包含 `total_visible_kv_tokens` 和 `total_kv_tokens`。
9. JSONL 不包含 tensor 内容。
10. `protected_sink_tokens` 稳定等于 sink/anchor 长度。
11. `protected_current_tokens` 对应 current `roped_key/value` 长度。
12. `compressible_history_tokens` 对应 working/history 长度。
13. 没有实现 compression。
14. 没有实现 reuse。
15. 没有改变 attention 结果。
16. 没有改变训练逻辑。

## 9. 下一轮 Round 3B 建议

Round 3B 建议先做 clean/history KV compression 的 dry-run：

1. 保持真实 KV 不变，只计算如果压缩 candidate region 后的 token 数和比例。
2. 继续保护 sink/current。
3. 对比 attention 内部真实 candidate 和 pipeline-level range 是否一致。
4. dry-run 稳定后，再做真实 compression prototype。
5. 真实 compression 仍必须默认关闭，且先只作用于 clean/history KV。
