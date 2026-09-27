# R0 服务器更新清单（2026-09-15）

目标目录：`/mnt/42_store/zxz2/xr/RollingForcing-main`。

以下文件应保持相对路径上传并覆盖服务器同名文件：

```text
pipeline/rolling_forcing_inference.py
utils/flowcache.py
utils/step_cache_r0.py
tests/test_step_cache_r0.py
configs/rolling_forcing_dmd_step_cache_r0_trace.yaml
configs/step_cache_r0_protocol_v2.json
prompts/step_cache_r0_single.txt
scripts/run_step_cache_r0.sh
scripts/collect_step_cache_r0_manifest.py
scripts/summarize_step_cache_r0.py
scripts/inspect_step_cache_r0_videos.py
scripts/assess_step_cache_r0.py
```

建议同时上传本说明和 `docs/round0_step_cache_baseline.md`，以便服务器侧保留本轮协议与回传标准；它们不参与推理。

上传完成后，从服务器运行：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
RF_ROOT="$PWD" \
PYTHON_BIN=/data/zxz2/condaenv/rollingforcing/bin/python \
GPU=0 \
CKPT="$PWD/checkpoints/rolling_forcing_dmd.pt" \
WAN_MODEL=/mnt/82_store/LLM-weights/Wan-AI/Wan2.1-T2V-1.3B \
bash scripts/run_step_cache_r0.sh all
```

结果会写入 `outputs/step_cache_r0/r0_*/`，本轮需要回传其中的 `reports/`、`logs/`、四个 MP4 和 shell 退出码。
