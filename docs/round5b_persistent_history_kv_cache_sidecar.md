# Round 5B: 持久化 history KV cache sidecar prototype

## 1. Round 5B 做什么

Round 5B 实现一个非常保守的 persistent compacted history KV cache prototype。

本轮新增 `persistent_cache_compress_*` 配置，默认关闭。显式开启后，在 inference 的 causal self-attention cache read path 中，对 history/working KV 做 uniform/linspace token selection，并把压缩后的 history K/V 保存到每层 `kv_cache[layer_idx]` 的 FlowCache sidecar 字段中。

本轮验证的重点是：

- history KV 的压缩表示能否被持久保存到 sidecar；
- 后续读取 history 时能否优先使用合法 sidecar；
- compressed history length 是否被记录；
- sink/current 是否保持完整保护；
- current_only 是否始终不压缩；
- clean_cache_update 默认不压缩；
- 不破坏 `global_end_index` / `local_end_index` / cache eviction；
- 视频生成仍然正常。

## 2. Round 5B 不做什么

Round 5B 不做以下内容：

- 不做 importance / redundancy scoring；
- 不做 output reuse；
- 不做 L1rel threshold recomputation；
- 不压缩 current_only；
- 默认不压缩 clean_cache_update；
- 不压缩 attention sink；
- 不压缩 current denoising KV；
- 不改训练逻辑；
- 不改 checkpoint；
- 不自动下载模型；
- 不改 cross-attention cache。

## 3. Round 5A logical_read_path 的局限

Round 5A 已经证明 logical read path 可以降低本次 attention 可见的 history KV 长度，但它每次 attention 都从完整 history 重新做 selection。

Round 5A 的局限是：

- 压缩结果不持久；
- 后续 window / 后续 attention 默认仍从完整 history 重新读；
- `kv_cache["k"]` / `kv_cache["v"]` 主体不变；
- 固定预分配 cache buffer 不变；
- cache 写入、搬移、eviction 不变；
- peak allocated / reserved 不一定下降。

## 4. 为什么 Round 5B 使用 sidecar 而不是直接 in-place compaction

当前 RollingForcing 的 causal KV cache 是固定预分配 tensor，每层在 `_initialize_kv_cache` 中一次性分配：

```python
kv_cache["k"]: [batch_size, 1560 * 24, 12, 128]
kv_cache["v"]: [batch_size, 1560 * 24, 12, 128]
```

主 cache 的写入、局部索引、全局索引和 eviction 都依赖 `global_end_index` / `local_end_index` 与固定位置语义。直接 in-place compact history 会影响未来读写位置，风险较高。

因此 Round 5B 采用 sidecar：

- 保留主 `kv_cache["k"]` / `kv_cache["v"]` 不动；
- 不修改 `global_end_index` / `local_end_index`；
- 不修改 eviction 逻辑；
- 在每层 `kv_cache` dict 中新增 FlowCache sidecar 字段；
- sidecar 只保存 compressed history，不保存 sink/current。

## 5. sidecar 的生命周期

每层 sidecar 挂在该层的 `kv_cache` dict 上，字段包括：

- `flowcache_compressed_history_key`
- `flowcache_compressed_history_value`
- `flowcache_compressed_history_start`
- `flowcache_compressed_history_end`
- `flowcache_compressed_history_source_start`
- `flowcache_compressed_history_source_end`
- `flowcache_compressed_history_window_index`
- `flowcache_compressed_history_layer_idx`
- `flowcache_compressed_history_target_ratio`
- `flowcache_compressed_history_strategy`
- `flowcache_compressed_history_tokens`
- `flowcache_compressed_history_global_end_index`
- `flowcache_compressed_history_local_end_index`
- `flowcache_compressed_history_valid`

生命周期策略：

1. denoise `anchor_working_current` branch 读取 history/working range；
2. 如果 sidecar 不存在或无效，则从当前 history range 创建 compressed sidecar；
3. 本次 attention 可以直接使用 newly-created sidecar；
4. 如果 sidecar metadata 与当前 history range 完全匹配，则复用 sidecar；
5. 如果 source range、strategy、ratio、dtype/device、end index 等不匹配，则失效并重建；
6. 每次主 cache 写入或 eviction 后，旧 sidecar 会被标记 invalid，避免 stale history。

第一版允许 create 多、reuse 少。重点是验证持久化表示和合法性检查路径，而不是立刻获得稳定加速。

## 6. sidecar 有效性判断

使用 sidecar 前必须验证：

- layer_idx 匹配；
- source_start / source_end 匹配当前 history range；
- strategy 匹配；
- target_ratio 匹配；
- compressed_tokens 不超过 original_history_tokens；
- global_end_index / local_end_index 与创建时一致；
- key/value tensor 存在；
- key/value token length 一致；
- key/value device 与当前 KV 一致；
- key/value dtype 与当前 KV 一致；
- debug verify 开启时检查 NaN/Inf。

任一条件失败时，fallback 到原始 history KV，并记录 sidecar invalidate 或 fallback。

## 7. sink/current 保护策略

Round 5B sidecar 只保存 history/working region。

- sink/anchor KV 从原始 cache 读取并完整保留；
- current denoising KV 使用当前 forward 计算出的 `roped_key` / `v`，完整保留；
- sidecar 不包含 sink；
- sidecar 不包含 current；
- `original_visible_tokens = sink + history + current`；
- `compressed_visible_tokens = sink + compressed_history + current`。

## 8. clean_cache_update 为什么默认不压

clean_cache_update 会在 window 后用 clean frame 更新 cache。它与 cache 写入/局部索引更新关系更近，错误压缩会更容易影响后续窗口的 cache 内容。

因此本轮默认：

```yaml
persistent_cache_compress_apply_to_clean_cache_update: false
```

如果未来显式开启，需要单独验证 clean branch 的 sidecar create/reuse/fallback 与输出质量。

## 9. 为什么本轮仍不保证 peak memory 下降

当前 `kv_cache` 是固定预分配 tensor。Round 5B 不缩小主 cache buffer，也不改动态分配策略。

因此：

- 主 `kv_cache["k"]` / `kv_cache["v"]` 的分配大小不变；
- PyTorch peak allocated / reserved 不一定下降；
- sidecar 作为额外压缩副本，短期甚至可能增加少量峰值；
- 本轮目标不是 memory-shrinking dynamic cache，而是验证 persistent compressed-history read path 是否正确。

真正降低 peak memory 需要 Round 5C 引入 dynamic compacted KV cache buffer 或更小 cache allocation。

## 10. fallback 策略

出现以下情况时 fallback：

- sidecar metadata 不匹配；
- source range 不匹配；
- strategy/ratio 不匹配；
- dtype/device 不匹配；
- key/value length 不一致；
- compressed length 非法；
- 出现 NaN/Inf；
- sidecar 创建或 attention 使用期间抛异常。

fallback 时：

- 使用原始 history KV；
- 不修改 sink/current；
- 不修改主 `kv_cache["k"]` / `kv_cache["v"]`；
- 记录 `[FlowCache][persistent_cache_sidecar] action=fallback`；
- 记录 `[FlowCache][persistent_cache_compression] fallback=True`；
- summary 中统计 fallback_count / sidecar_fallback_count。

## 11. 服务器验证命令与需要回传的文件

服务器环境：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

创建 Round 5B config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round5b.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round5b.yaml <<'YAML'

flowcache:
  enabled: true
  debug: false

  metadata_enabled: true
  metadata_output_path: logs/round5b_persistent_cache.jsonl
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

  persistent_cache_compress_enabled: true
  persistent_cache_compress_real_enabled: true
  persistent_cache_compress_target_ratio: 0.5
  persistent_cache_compress_strategy: uniform
  persistent_cache_compress_min_candidate_tokens: 4680
  persistent_cache_compress_apply_to_denoise: true
  persistent_cache_compress_apply_to_clean_cache_update: false
  persistent_cache_compress_apply_to_current_only: false
  persistent_cache_compress_protect_sink: true
  persistent_cache_compress_protect_current: true
  persistent_cache_compress_sidecar_enabled: true
  persistent_cache_compress_debug_verify: true
  persistent_cache_compress_max_windows: null
  persistent_cache_compress_max_layers: null

  kv_compress_shadow_compare: false
  kv_compress_shadow_max_events: 0

  output_reuse_enabled: false
  l1rel_threshold: 0.0
YAML
```

disabled baseline 21-frame smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_disabled_smoke.log
```

Round 5B 21-frame smoke：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5b.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_persistent_cache_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_persistent_cache_smoke.log
```

先只跑 21-frame smoke。81-frame 三 prompt 等 21-frame 结果回传后再判断。

检查命令：

```bash
grep -n "\[FlowCache\]" logs/round5b_disabled_smoke.log | head -n 40
grep -n "\[FlowCache\]\[persistent_cache_compression\]" logs/round5b_persistent_cache_smoke.log | head -n 120
grep -n "\[FlowCache\]\[persistent_cache_sidecar\]" logs/round5b_persistent_cache_smoke.log | head -n 120
grep -n "\[FlowCache\]\[persistent_cache_summary\]" logs/round5b_persistent_cache_smoke.log
grep -n "\[FlowCache\]\[warning\]" logs/round5b_persistent_cache_smoke.log | head -n 80
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round5b_*.log

head -n 80 logs/round5b_persistent_cache.jsonl
wc -l logs/round5b_persistent_cache.jsonl

ls -lh videos/round5b_disabled_smoke
ls -lh videos/round5b_persistent_cache_smoke
```

JSONL 检查脚本：

```bash
python - <<'PY'
import json
from collections import Counter, defaultdict

path = "logs/round5b_persistent_cache.jsonl"
event_counts = Counter()
applied_by_branch = Counter()
skipped_by_reason = Counter()
sidecar_counts = Counter()
fallback_count = 0
warning_count = 0
current_only_applied = 0
clean_cache_update_applied = 0
formula_errors = []
tensor_like_count = 0
parse_errors = 0

def looks_tensor_like(value):
    if isinstance(value, str):
        lowered = value.lower()
        return "tensor(" in lowered or "device=" in lowered or "dtype=" in lowered
    if isinstance(value, dict):
        return any(looks_tensor_like(v) for v in value.values())
    if isinstance(value, list):
        return any(looks_tensor_like(v) for v in value)
    return False

with open(path, "r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception as exc:
            parse_errors += 1
            print("parse_error", line_no, exc)
            continue

        if looks_tensor_like(item):
            tensor_like_count += 1

        event = item.get("event")
        event_counts[event] += 1

        if event == "persistent_cache_sidecar":
            sidecar_counts[item.get("action", "unknown")] += 1
            continue

        if event == "persistent_cache_compression":
            branch = item.get("branch")
            applied = bool(item.get("applied"))
            if applied:
                applied_by_branch[branch] += 1
            else:
                skipped_by_reason[item.get("skipped_reason") or "unknown"] += 1
            fallback_count += int(bool(item.get("fallback")))
            current_only_applied += int(branch == "current_only" and applied)
            clean_cache_update_applied += int(branch == "clean_cache_update" and applied)

            expected_original = (
                int(item.get("protected_sink_tokens", 0)) +
                int(item.get("original_history_tokens", 0)) +
                int(item.get("protected_current_tokens", 0))
            )
            expected_compressed = (
                int(item.get("protected_sink_tokens", 0)) +
                int(item.get("compressed_history_tokens", 0)) +
                int(item.get("protected_current_tokens", 0))
            )
            if int(item.get("original_visible_tokens", -1)) != expected_original:
                formula_errors.append((line_no, "original_visible"))
            if int(item.get("compressed_visible_tokens", -1)) != expected_compressed:
                formula_errors.append((line_no, "compressed_visible"))
            if int(item.get("compressed_history_tokens", 0)) > int(item.get("original_history_tokens", 0)):
                formula_errors.append((line_no, "history_order"))
            if int(item.get("compressed_visible_tokens", 0)) > int(item.get("original_visible_tokens", 0)):
                formula_errors.append((line_no, "visible_order"))
            if int(item.get("saved_visible_tokens", 0)) < 0:
                formula_errors.append((line_no, "negative_saved"))

        if event == "persistent_cache_summary":
            warning_count = int(item.get("warning_count", warning_count))

print("event_counts", dict(event_counts))
print("applied_by_branch", dict(applied_by_branch))
print("skipped_by_reason", dict(skipped_by_reason))
print("current_only_applied", current_only_applied)
print("clean_cache_update_applied", clean_cache_update_applied)
print("fallback_count", fallback_count)
print("warning_count", warning_count)
print("sidecar_create_count", sidecar_counts.get("create", 0))
print("sidecar_reuse_count", sidecar_counts.get("reuse", 0))
print("sidecar_invalidate_count", sidecar_counts.get("invalidate", 0))
print("sidecar_fallback_count", sidecar_counts.get("fallback", 0))
print("formula_errors_count", len(formula_errors))
print("formula_errors_head", formula_errors[:10])
print("tensor_like_count", tensor_like_count)
print("parse_errors", parse_errors)
PY
```

请回传：

- `logs/round5b_disabled_smoke.log` 中 FlowCache grep 结果；
- `logs/round5b_persistent_cache_smoke.log` 中 compression / sidecar / summary / warning grep 结果；
- Traceback/OOM grep 结果；
- JSONL 检查脚本输出；
- `wc -l logs/round5b_persistent_cache.jsonl`；
- 两个输出视频目录 `ls -lh`；
- 如果方便，回传 `[FlowCache][persistent_cache_summary]` 完整行。

## 12. Round 5B 21-frame smoke 成功标准

1. disabled smoke 成功生成视频；
2. persistent-cache smoke 成功生成视频；
3. disabled 日志没有 `[FlowCache]`；
4. persistent-cache 日志有 `[FlowCache][persistent_cache_compression]`；
5. persistent-cache 日志有 `[FlowCache][persistent_cache_summary]`；
6. JSONL 非空；
7. 没有 Traceback / RuntimeError / OOM；
8. warning_count = 0 或很少且可解释；
9. fallback_count = 0 或很少且可解释；
10. current_only applied = 0；
11. clean_cache_update applied = 0；
12. protected_sink_tokens 不变；
13. protected_current_tokens 不变；
14. compressed_history_tokens <= original_history_tokens；
15. compressed_visible_tokens <= original_visible_tokens；
16. saved_visible_tokens >= 0；
17. sidecar_create_count > 0；
18. sidecar metadata 有效；
19. JSONL 不包含 tensor 内容；
20. 输出视频存在且非空；
21. 没有实现 importance/redundancy scoring；
22. 没有实现 output reuse；
23. 没有实现 L1rel；
24. 没有改训练逻辑。

## 13. 下一轮建议

如果 Round 5B smoke 通过，下一步建议先评估 sidecar create/reuse/invalidate 分布：

- 如果 create 多、reuse 少，先分析 source range 和 cache write invalidation 的节奏；
- 如果 fallback 或 warning 多，先修 metadata 校验或 source range 边界；
- 如果视频稳定但显存不降，符合固定预分配 cache 预期；
- Round 5C 再考虑 dynamic compacted KV cache buffer 或更小 cache allocation；
- 暂时仍不要进入 importance/redundancy scoring、output reuse 或 L1rel。
