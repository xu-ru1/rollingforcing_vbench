# R0：基线与阶段轨迹运行说明

本轮只冻结并测量当前 RollingForcing 基线。`rolling_forcing_dmd_step_cache_r0_trace.yaml` 仅记录实际 scheduler stage 和 RNG SHA256；output reuse、KV compaction、cache-body compaction 均关闭。因此 R0 不会创建或使用 step cache，也不会改变主推理的随机数调用。

运行合同位于 `configs/step_cache_r0_protocol_v2.json`。运行器 `scripts/run_step_cache_r0.sh` 会创建一个全新的输出目录；同一目录禁止续写，避免将不同源码、checkpoint 或环境的结果混在一起。

## 服务器执行

在服务器中将仓库放在任意绝对路径，激活已有可用推理环境，然后一次性执行：

```bash
cd /absolute/path/RollingForcing-main
RF_ROOT="$PWD" \
GPU=0 \
CKPT=/absolute/path/rolling_forcing_dmd.pt \
WAN_MODEL=/absolute/path/Wan2.1-T2V-1.3B \
bash scripts/run_step_cache_r0.sh all
```

`GPU` 是物理卡号；脚本通过 `CUDA_VISIBLE_DEVICES` 将它交给单一推理进程。默认使用 EMA。若 checkpoint 没有 EMA 参数，可显式设置 `USE_EMA=0`；运行器会自动检查 `_regular.mp4` 文件。

如需先定位环境或模型问题，使用同样的环境变量执行：

```bash
bash scripts/run_step_cache_r0.sh preflight
```

`all` 的输出目录形如 `outputs/step_cache_r0/r0_YYYYMMDDTHHMMSSZ/`，含四次串行视频生成：B0 两次、21 latent 的 trace、81 latent 的 trace。请勿手工修改其内部 YAML 或复用已存在的输出目录。

## 需要回传的材料

打包该次输出目录中的 `reports/`、`logs/`、四个 `videos/*/*.mp4`，并附上完整命令和 shell 退出码。重点文件为：

- `reports/preflight_manifest.json`：源码 SHA256、checkpoint hash、模型资产、环境与 GPU。
- `reports/trace_21_trace_summary.json`、`reports/trace_81_trace_summary.json`：stage/RNG 自动验收结果。
- `reports/*_profile_summary.json`：粗粒度 profile。
- `reports/*_video_check.json`：实际 MP4 解码帧数。
- `reports/r0_assessment.json`：四次运行的统一自动验收结论。

通过的 `trace_81_trace_summary.json` 必须报告 27 个 video block、135 个 stage event、31 个 window；每个 block 恰好有 stage 0–4，且主 forward 前后 CPU/CUDA RNG hash 一致。21 latent 的视频必须解码为 81 帧，81 latent 的视频必须解码为 321 帧。

R0 的本地代码检查与 CPU 单测可在没有 GPU 或模型的机器上执行：

```bash
python -m py_compile inference.py pipeline/rolling_forcing_inference.py utils/flowcache.py utils/step_cache_r0.py
python -m unittest discover -s tests -p 'test_step_cache_r0.py' -v
```
