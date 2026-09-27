# Round 1 FlowCache 默认关闭框架

本轮只加入 FlowCache 的配置、轻量 manager 和 debug logging。它不实现 output reuse，不实现 KV compression，不修改 attention kernel，不修改 checkpoint 格式，也不改变训练逻辑。

## 1. 修改文件

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `configs/default_config.yaml` | 新增 `flowcache` 配置块，所有功能默认关闭 | 让所有现有 inference config 在 merge default config 后都具备 FlowCache 开关，同时默认行为保持原样 |
| `utils/flowcache.py` | 新增 `FlowCacheConfig` 和 `FlowCacheManager` | 提供 Round 1 的默认关闭框架和 gated debug logging 入口 |
| `pipeline/rolling_forcing_inference.py` | 在 `CausalInferencePipeline.__init__` 中创建 `self.flowcache_manager`，在 rolling forcing 主循环中加入可选日志 | 记录 rolling window、denoising step、noisy cache shape、KV cache end index 等 metadata，供下一轮定位 sink/history/current KV range |
| `docs/round1_flowcache_skeleton.md` | 本文档 | 记录本轮改动、验证方法和下一轮建议 |

## 2. 新增配置项

默认配置位于 `configs/default_config.yaml`：

```yaml
flowcache:
  enabled: false
  debug: false
  output_reuse_enabled: false
  kv_compress_enabled: false
  l1rel_threshold: 0.0
  protect_sink_tokens: true
  log_every_window: true
```

字段含义：

| 字段 | 默认值 | Round 1 含义 |
| --- | --- | --- |
| `enabled` | `false` | FlowCache 总开关。默认关闭时 manager 全部 no-op |
| `debug` | `false` | debug 日志开关。只有 `enabled=true` 且 `debug=true` 时打印 |
| `output_reuse_enabled` | `false` | 预留 output reuse 开关，本轮不使用 |
| `kv_compress_enabled` | `false` | 预留 KV compression 开关，本轮不使用 |
| `l1rel_threshold` | `0.0` | 预留 L1rel threshold，本轮不参与判断 |
| `protect_sink_tokens` | `true` | 预留 sink token 保护策略，本轮不修改 KV |
| `log_every_window` | `true` | debug 模式下是否每个 rolling window 打印窗口 metadata |

## 3. FlowCacheManager 当前做什么

`utils/flowcache.py` 中的 `FlowCacheManager` 当前只做三件事：

1. 从 config 中读取 FlowCache 相关字段。
2. 在 `enabled=true` 且 `debug=true` 时打印 rolling window metadata。
3. 在 `enabled=true` 且 `debug=true` 时打印 clean KV cache 每层的 `global_end_index` 和 `local_end_index`。

当前不会做的事：

1. 不保存 GPU tensor。
2. 不修改 latent、noise、KV cache 或 attention 输出。
3. 不实现 chunk-wise output reuse。
4. 不实现 history KV 或 clean-chunk KV compression。
5. 不做 L1rel threshold 判断。
6. 不改变 checkpoint 读取、模型结构或 attention kernel。

## 4. 默认关闭时为什么不影响原始 inference

`FlowCacheManager.should_log` 的条件是：

```python
self.config.enabled and self.config.debug
```

默认配置中 `flowcache.enabled=false` 且 `flowcache.debug=false`，因此所有 `log_window(...)` 和 `log_kv_cache_summary(...)` 调用会直接返回。接入点只读取 shape 和 cache metadata，不参与任何 tensor 计算；默认关闭时也不会新增日志。因此默认配置下，原始 RollingForcing inference 的计算路径、checkpoint 加载和输出保存逻辑保持不变。

## 5. 服务器验证命令

服务器路径：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
mkdir -p logs videos/round1_disabled_smoke videos/round1_debug_smoke
```

先做轻量检查：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py
python -c "from utils.flowcache import FlowCacheManager; m = FlowCacheManager(); print(m.enabled, m.debug)"
```

默认关闭模式 smoke test：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round1_disabled_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round1_disabled_smoke.log
```

确认默认关闭模式没有 FlowCache 日志：

```bash
grep -n "\[FlowCache\]" logs/round1_disabled_smoke.log
```

debug 模式建议复制一个临时 config，不覆盖原始 config：

```bash
cp configs/rolling_forcing_dmd.yaml configs/rolling_forcing_dmd_flowcache_debug.yaml
cat >> configs/rolling_forcing_dmd_flowcache_debug.yaml <<'YAML'

flowcache:
  enabled: true
  debug: true
  output_reuse_enabled: false
  kv_compress_enabled: false
  l1rel_threshold: 0.0
  protect_sink_tokens: true
  log_every_window: true
YAML
```

debug smoke test：

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path configs/rolling_forcing_dmd_flowcache_debug.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/round1_debug_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/round1_debug_smoke.log
```

查看 debug 日志：

```bash
grep -n "\[FlowCache\]" logs/round1_debug_smoke.log | head -n 40
```

预期结果：

1. `round1_disabled_smoke.log` 中不出现 `[FlowCache]`。
2. `round1_debug_smoke.log` 中出现 `[FlowCache][window]`。
3. `round1_debug_smoke.log` 中出现 `[FlowCache][kv_cache]`，能看到每层 clean KV cache 的 `global_end_index` 和 `local_end_index`。
4. 两个 smoke test 都能正常生成视频。

## 6. 跑完后需要回传的日志

请回传或贴出以下内容：

```bash
logs/round1_disabled_smoke.log
logs/round1_debug_smoke.log
configs/rolling_forcing_dmd_flowcache_debug.yaml
```

以及以下命令输出：

```bash
grep -n "\[FlowCache\]" logs/round1_disabled_smoke.log
grep -n "\[FlowCache\]" logs/round1_debug_smoke.log | head -n 80
ls -lh videos/round1_disabled_smoke videos/round1_debug_smoke
```

## 7. 下一轮建议

Round 2 建议只做 KV metadata tracker，仍然不压缩、不复用：

1. 在 clean KV cache 更新前后记录每层 `global_start/global_end/local_start/local_end`。
2. 根据 rolling window 的 `start_block/end_block` 和 `num_frame_per_block` 标注 attention sink、history KV、current denoising window KV。
3. 只输出 range metadata，不移动、不裁剪、不拼接 KV。
4. 保持 `flowcache.enabled=false` 时完全无影响。
5. 为 Round 3 的 clean-chunk KV compression 预留只读统计接口。
