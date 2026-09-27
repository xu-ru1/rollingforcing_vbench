# R1：第一层 feature 观察任务

本轮保持 `step_cache.enabled=false`。模型仍执行完整 RollingForcing；新增代码只在第一层 `norm1` 加 timestep modulation 后、调用 self-attention 前，对每个 video block 做 2×2 空间池化、FP32 relative-L1 距离计算并输出标量 JSONL。

上传更新文件：

```text
pipeline/rolling_forcing_inference.py
utils/wan_wrapper.py
wan/modules/causal_model.py
utils/step_cache_policy.py
utils/step_cache_state.py
utils/step_cache_metric.py
tests/test_step_cache_policy.py
tests/test_step_cache_state.py
tests/test_step_cache_metric.py
configs/default_config.yaml
configs/rolling_forcing_dmd_step_cache_r1_observe.yaml
scripts/run_step_cache_r1_observe.sh
scripts/summarize_step_cache_r1.py
```

服务器运行：

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
RF_ROOT="$PWD" \
PYTHON_BIN=/data/zxz2/condaenv/rollingforcing/bin/python \
GPU=0 \
CKPT="$PWD/checkpoints/rolling_forcing_dmd.pt" \
bash scripts/run_step_cache_r1_observe.sh
```

回传 `reports/r1_metric_summary.json`、`reports/r1_metrics.jsonl`、`logs/`、`videos/observe_81/0-0_ema.mp4` 与退出码。通过条件：135 条 metric record、每个 stage 0–4 各 27 条、108 条有效相邻 stage 距离、JSONL 不含原始 feature 张量。
