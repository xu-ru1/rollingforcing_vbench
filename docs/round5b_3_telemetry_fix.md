# Round 5B.3：persistent sidecar telemetry 修复

## 本轮做什么

Round 5B.3 只修 persistent cache sidecar 的日志和 summary 统计口径。核心改动是给 FlowCacheManager 生命周期生成 `flowcache_run_id`，并让 JSONL 记录和 `[FlowCache][persistent_cache_summary]` 使用同一个 run-level 标识。验证时应按最新 summary 的 `flowcache_run_id` 过滤 `persistent_cache_sidecar` action，再和 summary 中的 sidecar count 对齐。

## 本轮不做什么

本轮不实现 importance / redundancy scoring，不实现 output reuse，不实现 L1rel threshold，不新增 sidecar reuse 策略，不做 dynamic compacted KV cache，不改 `kv_cache["k"]` / `kv_cache["v"]` 主 tensor，不改 cache eviction 主逻辑，不改训练逻辑，也不改 checkpoint。

## 为什么 5B.2 会出现 JSONL reuse=90 但 summary reuse=0

5B.2 的回传 JSONL 里混入了不止一个实验段：81-frame ratio050 的 cumulative summary 显示 reuse 为 0，但同一个 JSONL 后面又追加了一个 21-frame summary 和 21-frame sidecar reuse events。直接统计整份 JSONL 的 `action="reuse"` 会把不同实验/run 的记录混在一起，因此会得到 reuse=90，而这并不等价于某条 81-frame summary 的口径。

## summary 统计口径

Round 5B.3 后，`persistent_cache_summary` 的口径是 `summary_scope=manager_lifetime_cumulative`。它表示当前 Python 进程中当前 `FlowCacheManager` 自创建以来的累计统计；如果同一进程内多次调用 inference，会输出多条累计 summary，`summary_index` 递增。它不是“整个 JSONL 文件”的统计，因为 JSONL 文件可能被多次实验追加。

JSONL 中所有写入事件都会带 `flowcache_run_id`。因此检查 sidecar action 时，必须用 latest summary 的 `flowcache_run_id` 过滤同一 run 的 `persistent_cache_sidecar` 事件。

## 如何验证 summary 和 JSONL action 对齐

检查脚本应执行以下逻辑：

1. 读取 JSONL；
2. 找到最后一条 `event == "persistent_cache_summary"`；
3. 读取该 summary 的 `flowcache_run_id`；
4. 只统计同一个 `flowcache_run_id` 下的 `event == "persistent_cache_sidecar"`；
5. 比较 create / reuse / keep_valid / invalidate / fallback 与 summary 字段是否一致；
6. 同时检查 parse error、tensor-like 字符串、visible token 公式、warning/fallback/current_only/clean_cache_update。

## 为什么本轮不追求 runtime / memory 收益

当前 kv_cache 主体仍是固定预分配 tensor。persistent sidecar 会额外保存 compressed history tensor，用来证明持久化压缩表示和 read path 能工作；它不会缩小主 kv_cache buffer，因此不保证降低 peak allocated / reserved，甚至可能增加显存。本轮只保证 telemetry 不误导，为后续 Round 5C 判断 dynamic compacted KV cache buffer 做准备。

## 服务器验证命令

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
```

语法检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/wan_wrapper.py wan/modules/causal_model.py
```

创建 21-frame ratio050 config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050.yaml
cat >> configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050.yaml <<'YAML'

flowcache:
  enabled: true
  debug: false

  metadata_enabled: true
  metadata_output_path: logs/round5b_3_21_ratio050.jsonl
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

创建 81-frame single-prompt config：

```bash
sed 's#logs/round5b_3_21_ratio050.jsonl#logs/round5b_3_81_single_ratio050.jsonl#' \
  configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050.yaml \
  > configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050_81.yaml
```

21-frame ratio050 telemetry sanity：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_3_21_ratio050 \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_3_21_ratio050.log
```

81-frame single-prompt ratio050 telemetry sanity：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050_81.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round5b_3_81_single_ratio050 \
  --num_output_frames 81 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round5b_3_81_single_ratio050.log
```

## 检查命令

```bash
grep -n "\[FlowCache\]\[persistent_cache_summary\]" logs/round5b_3_21_ratio050.log
grep -n "\[FlowCache\]\[persistent_cache_summary\]" logs/round5b_3_81_single_ratio050.log
grep -n "\[FlowCache\]\[warning\]" logs/round5b_3_*ratio050.log | head -n 80
grep -n "Traceback\|RuntimeError\|CUDA out of memory\|OOM" logs/round5b_3_*ratio050.log
wc -l logs/round5b_3_21_ratio050.jsonl logs/round5b_3_81_single_ratio050.jsonl
ls -lh videos/round5b_3_21_ratio050
ls -lh videos/round5b_3_81_single_ratio050
```

JSONL 对齐检查：

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

paths = [
    Path("logs/round5b_3_21_ratio050.jsonl"),
    Path("logs/round5b_3_81_single_ratio050.jsonl"),
]
for path in paths:
    parse_errors = 0
    tensor_like = 0
    formula_errors = 0
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
        if obj.get("event") == "persistent_cache_summary":
            summaries.append(obj)
        if obj.get("event") == "persistent_cache_compression":
            ov = obj.get("original_visible_tokens", 0)
            cv = obj.get("compressed_visible_tokens", 0)
            sv = obj.get("saved_visible_tokens", 0)
            if ov - cv != sv or cv > ov or sv < 0:
                formula_errors += 1
    if not summaries:
        print(path, "missing persistent_cache_summary")
        continue
    summary = summaries[-1]
    run_id = summary.get("flowcache_run_id")
    actions = Counter(
        obj.get("action")
        for obj in rows
        if obj.get("event") == "persistent_cache_sidecar"
        and obj.get("flowcache_run_id") == run_id
    )
    expected = {
        "create": summary.get("sidecar_create_count", 0),
        "reuse": summary.get("sidecar_reuse_count", 0),
        "keep_valid": summary.get("sidecar_keep_valid_count", 0),
        "invalidate": summary.get("sidecar_invalidate_count", 0),
        "fallback": summary.get("sidecar_fallback_count", 0),
    }
    actual = {key: actions.get(key, 0) for key in expected}
    print("path", path)
    print("run_id", run_id)
    print("summary_scope", summary.get("summary_scope"))
    print("summary_index", summary.get("summary_index"))
    print("expected", expected)
    print("actual", actual)
    print("aligned", expected == actual)
    print("parse_errors", parse_errors)
    print("tensor_like", tensor_like)
    print("formula_errors", formula_errors)
    print("warning_count", summary.get("warning_count"))
    print("fallback_count", summary.get("fallback_count"))
    print("current_only_applied", summary.get("current_only_applied"))
    print("clean_cache_update_applied", summary.get("clean_cache_update_applied"))
PY
```

## 必须先跑的实验

必须先跑 `py_compile`，然后跑 21-frame ratio050 telemetry sanity，再跑 81-frame single-prompt ratio050 telemetry sanity。两项都只用于验证 telemetry，不用于重新判断性能。

## 可以跳过的实验

本轮可以跳过 disabled baseline、ratio075、三 prompt 81-frame、大规模 benchmark、视觉质量对比和 runtime/memory 结论。

## 成功标准

1. 21-frame 和 81-frame single-prompt 都能生成视频；
2. 两个日志都有 `[FlowCache][persistent_cache_summary]`；
3. summary 行包含 `flowcache_run_id`、`summary_scope=manager_lifetime_cumulative`、`summary_index`；
4. JSONL latest summary 的 `flowcache_run_id` 与同 run sidecar action 统计对齐；
5. `warning_count=0`；
6. `fallback_count=0`；
7. `current_only_applied=0`；
8. `clean_cache_update_applied=0`；
9. `parse_errors=0`；
10. `tensor_like=0`；
11. `formula_errors=0`；
12. 没有 Traceback / RuntimeError / OOM；
13. 没有算法改动；
14. 没有训练改动。

## 需要回传

第一组必须回传：

1. `logs/round5b_3_21_ratio050.log`
2. `logs/round5b_3_21_ratio050.jsonl`
3. `logs/round5b_3_81_single_ratio050.log`
4. `logs/round5b_3_81_single_ratio050.jsonl`
5. `configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050.yaml`
6. `configs/rolling_forcing_dmd_flowcache_round5b_3_ratio050_81.yaml`
7. `docs/round5b_3_telemetry_fix.md`
8. summary grep 输出
9. JSONL action/count 对齐检查输出
10. warning/error grep 输出
11. 两个视频目录的 `ls -lh` 输出
