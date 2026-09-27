# Round 2 KV Metadata Tracker

本轮目标是给 FlowCache 增加只观察、不修改 tensor 的 KV metadata tracker。它用于后续 FlowCache 适配前确认 rolling window、attention sink、history KV、current denoising KV 的范围关系。

## 1. Round 2 做了什么

| 文件 | 修改内容 | 目的 |
| --- | --- | --- |
| `configs/default_config.yaml` | 在 `flowcache` 下新增 `metadata_enabled`、`metadata_output_path`、`log_kv_ranges`、`log_summary` | 默认关闭 metadata tracker，debug config 中可单独打开 |
| `utils/flowcache.py` | 扩展 `FlowCacheManager`，新增 window record、KV range record、event、summary、jsonl 输出 | 只保存 int、float、shape、range、timestamp 等轻量 metadata |
| `pipeline/rolling_forcing_inference.py` | 在 rolling window loop 中调用 `begin_window`、`log_kv_range`、`log_event`、`end_window`、`summarize` | 记录每个 window/layer 的 KV cache end index、sink/history/current range 和 window 耗时 |
| `docs/round2_kv_metadata_tracker.md` | 本文档 | 记录字段含义、验证命令和下一轮建议 |

本轮没有把 `FlowCacheManager` 传入 `wan/modules/causal_model.py`。原因是 Round 2 需要尽量降低风险，先在 pipeline 层基于真实 `kv_cache` metadata 做观察。attention 内部变量已定位，后续 Round 3 前再决定是否增加默认 `None` 的只读 hook。

## 2. Round 2 没做什么

1. 没有实现 KV compression。
2. 没有实现 output reuse。
3. 没有实现 L1rel threshold。
4. 没有改变 self-attention 数学计算。
5. 没有改变 Q/K/V tensor 内容。
6. 没有改变 attention 输入拼接顺序。
7. 没有改变 cache eviction 逻辑。
8. 没有改变 checkpoint 读取。
9. 没有改变训练逻辑。
10. 没有删除 Self Forcing、DMD、Wan 相关代码。

## 3. 新增配置

默认配置仍然关闭：

```yaml
flowcache:
  enabled: false
  debug: false
  output_reuse_enabled: false
  kv_compress_enabled: false
  metadata_enabled: false
  metadata_output_path: null
  l1rel_threshold: 0.0
  protect_sink_tokens: true
  log_every_window: true
  log_kv_ranges: false
  log_summary: true
```

字段说明：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `enabled` | `false` | FlowCache 总开关，默认关闭 |
| `metadata_enabled` | `false` | Round 2 metadata tracker 开关 |
| `metadata_output_path` | `null` | JSONL 输出路径；为空则不写文件 |
| `debug` | `false` | 打印详细日志 |
| `log_kv_ranges` | `false` | 打印每层 `[FlowCache][kv_range]` |
| `log_summary` | `true` | inference 结束后打印 `[FlowCache][summary]` |
| `output_reuse_enabled` | `false` | 预留，Round 2 不使用 |
| `kv_compress_enabled` | `false` | 预留，Round 2 不使用 |
| `l1rel_threshold` | `0.0` | 预留，Round 2 不使用 |
| `protect_sink_tokens` | `true` | 预留，Round 2 不修改 KV |

## 4. Metadata 字段说明

JSONL 每条 KV range 记录包含：

| 字段 | 含义 |
| --- | --- |
| `event` | 记录阶段，例如 `before_denoise`、`after_denoise`、`after_clean_cache_update` |
| `timestamp` | 写记录时的 wall-clock timestamp |
| `window_index` | rolling forcing window 下标 |
| `start_block` / `end_block` | 当前 rolling window 覆盖的 block 范围 |
| `num_frame_per_block` | 每个 block 的 latent frame 数 |
| `rolling_window_length_blocks` | rolling window 的 block 长度 |
| `layer_idx` | Transformer layer 下标 |
| `global_end_index` | 该层 `kv_cache["global_end_index"]` |
| `local_end_index` | 该层 `kv_cache["local_end_index"]` |
| `sink_start` / `sink_end` | pipeline 层估计的 attention sink 可见区间 |
| `history_start` / `history_end` | pipeline 层估计的 history/working KV 可见区间 |
| `current_start` / `current_end` | pipeline 层估计的 current denoising KV 可见区间 |
| `cache_tokens` | 当前 local KV cache token 数 |
| `working_history_tokens` | 估计参与 self-attention 的 history/working token 数 |
| `current_tokens` | 当前 denoising query/current KV token 数 |
| `total_visible_kv_tokens` | 估计可见 self-attention KV token 总数 |
| `block_length` | `num_frame_per_block * frame_seq_length` |
| `frame_seq_length` | 每帧 latent token 数，当前为 1560 |
| `current_num_frames` | 当前记录对应的 current frame 数 |
| `max_attention_tokens` | attention 最大可见 token 数，当前取 `generator.seq_len` |

JSONL 不保存 tensor 内容，不保存 GPU tensor 引用。

## 5. Sink / History / Current Range 定义

真实 attention 内部位于 `wan/modules/causal_model.py` 的 `CausalWanSelfAttention.forward`：

1. `self.frame_length = 1560`。
2. `self.block_length = 3 * self.frame_length`。
3. `sink_tokens = 1 * self.block_length`，对应第一个 sink block。
4. 普通 denoising 路径中，`input_key` 按 `[anchor_cache_key, working_cache_key, roped_key]` 拼接。
5. `anchor_cache_key` 对应 attention sink。
6. `working_cache_key` 对应 history/working cache。
7. `roped_key` 对应 current denoising window 的当前 KV。

Round 2 的 pipeline-level tracker 按这个拼接顺序估计可见区间：

```text
sink_range    = [0, sink_tokens)
history_range = [sink_tokens, sink_tokens + working_history_tokens)
current_range = [sink_tokens + working_history_tokens,
                 sink_tokens + working_history_tokens + current_tokens)
```

注意：本轮没有进入 attention forward 内部打点，因此 `history_start/history_end` 更准确地说是 `history_or_working_range`。clean history KV 和 working history KV 的更细拆分建议放在 Round 3 前确认。

## 6. 服务器验证命令

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
mkdir -p logs videos/round2_disabled_smoke videos/round2_metadata_smoke
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

新建 Round 2 debug config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round2.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round2.yaml <<'YAML'

flowcache:
  enabled: true
  debug: true
  metadata_enabled: true
  metadata_output_path: logs/round2_kv_metadata.jsonl
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
  --output_folder videos/round2_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round2_disabled_smoke.log
```

metadata debug smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round2.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round2_metadata_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round2_metadata_smoke.log
```

结果检查：

```bash
grep -n "\[FlowCache\]" logs/round2_disabled_smoke.log | head
grep -n "\[FlowCache\]\[kv_range\]" logs/round2_metadata_smoke.log | head -n 80
grep -n "\[FlowCache\]\[summary\]" logs/round2_metadata_smoke.log
head -n 20 logs/round2_kv_metadata.jsonl
wc -l logs/round2_kv_metadata.jsonl
ls -lh videos/round2_disabled_smoke videos/round2_metadata_smoke
```

## 7. 如何判断 Round 2 成功

Round 2 成功条件：

1. disabled smoke 成功生成视频。
2. metadata smoke 成功生成视频。
3. disabled 日志无 `[FlowCache]`，不影响默认行为。
4. metadata 日志有 `[FlowCache][window]`。
5. metadata 日志有 `[FlowCache][kv_range]`。
6. metadata 日志有 `[FlowCache][summary]`。
7. `logs/round2_kv_metadata.jsonl` 存在且非空。
8. JSONL 中只包含数字、字符串、range、shape，不包含 tensor。
9. 没有实现 compression。
10. 没有实现 reuse。
11. 没有改变 attention 结果。

## 8. 下一轮 Round 3 建议

Round 3 建议只对 clean history KV 做 compression，继续保护 sink/current：

1. 增加 attention 内部只读 metadata hook，默认 `None`，确认 `anchor_cache_key`、`working_cache_key`、`roped_key` 的真实长度。
2. 明确区分 attention sink、clean history KV、working history KV、current denoising KV。
3. compression 只作用于 clean history KV candidate region。
4. `sink_range` 和 `current_range` 必须常驻，不压缩。
5. 继续保持 `flowcache.enabled=false` 时完全无影响。
