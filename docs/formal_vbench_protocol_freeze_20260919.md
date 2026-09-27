# RollingForcing step-position cache：正式实验协议冻结（2026-09-19）

本文件覆盖旧计划中关于正式生成长度、随机种子、方法矩阵和 VBench
口径的暂定描述。R0--R5 的开发与正确性结果不因本次协议冻结而失效；旧的
`scripts/run_step_cache_r6.sh` 和对应 R6 压缩包只保留为开发历史，不作为正式
校准或 VBench 输入。

## 1. 官方依据与可复现解释

- RollingForcing 论文明确规定定量评测视频为 30 秒、16 fps、832x480：
  <https://openreview.net/pdf?id=IAyzXjbfwo>。
- 官方仓库当前公开的唯一 CLI 推理示例使用 `--num_output_frames 126`：
  <https://github.com/TencentARC/RollingForcing>。
- 官方 `rolling_forcing_dmd.yaml` 设置 `num_frame_per_block: 3` 和 `seed: 0`：
  <https://raw.githubusercontent.com/TencentARC/RollingForcing/main/configs/rolling_forcing_dmd.yaml>。
- 官方推理入口把 `num_output_frames` 直接用作 T2V latent 长度，并以 16 fps
  写出视频：
  <https://raw.githubusercontent.com/TencentARC/RollingForcing/main/inference.py>。
- VBench 的 `trimmed30` 路径在视频超过 30 秒时使用
  `vlen = int(30 * fps)`，即 16 fps 时取前 480 帧：
  <https://github.com/Vchitect/VBench/blob/master/vbench/utils.py>。

公开论文没有逐字写出“正式 30 秒运行使用 126 latent”，也没有在正文中说明
501 到 480 帧的裁剪步骤。因此论文中只能声称我们遵循官方发布代码的
126-latent 推理规格，并按照论文与 VBench 的 30 秒口径评测，不能把未公开的
作者内部命令写成已获证实的事实。

## 2. 已冻结的生成与评测口径

1. 六种方法全部使用 126 latent frames、3 latent/block、42 个完整 block。
2. 六种方法统一使用 seed 0；同一 prompt 的初始噪声及后续完整随机数轨迹
   必须一致，并保存可核验 hash。
3. Wan VAE 对 126 latent 解码得到 501 个 RGB 帧。原始视频统一保存为
   501 帧、16 fps、832x480，作为生成完整性和运行审计材料。
4. VBench 输入由原始视频确定性截取前 480 帧，得到 30.0 秒。不得按方法、
   prompt 或结果选择不同区间，不得进行中心裁剪或质量驱动裁剪。
5. 原始视频与 480 帧评测视频分别保存 SHA256、帧数、fps、分辨率以及源文件
   映射。裁剪过程只能改变时间长度，禁止重新缩放、插帧或改变 fps。
6. diffusion latency、DiT PFLOPs、算子 token 数和复用统计均按实际执行的
   完整 126-latent 推理计算，不按 480 帧比例折算。
7. diffusion latency 覆盖完整 rolling denoising loop，包括 main DiT、
   `updating_cache=True` clean-cache forward、再加噪与必要缓存管理；排除文本
   编码、VAE 解码和视频写盘。端到端时间另列。

采用 126 而不是 121 的原因是 126 能整除每 block 的 3 latent，保持完整的
rolling window、stage、cache 生命周期和 PFLOPs 统计语义；六种方法共享完全
相同的 backbone 生成规格。

## 3. 正式方法矩阵

| 论文展示名 | 含义 | 角色 |
| --- | --- | --- |
| RF-Vanilla | 原生 RollingForcing，不启用新增跨阶段缓存 | 基线 |
| RF+Fixed-slow | 固定阈值缓存，慢速档 | 直接基线 |
| RF+Front-slow | Front Protect，匹配 slow 成本 | 主方法 |
| RF+Fixed-fast | 固定阈值缓存，快速档 | 直接基线 |
| RF+Front-fast | Front Protect，匹配 fast 成本 | 主方法 |
| RF+U-shape-fast | U-shape 调度，匹配 fast 成本 | 调度形状消融 |

Front 是主方法，Fixed 是直接基线，U-shape-fast 是快速档调度消融。本轮不增加
U-shape-slow。

## 4. 成本校准与冻结顺序

1. 在 126-latent 校准集上扫描 RF+Fixed 的真实运行曲线。
2. 只根据运行有效性、diffusion latency 和计算成本提出 slow/fast 候选，
   不查看 VBench 或其他质量结果。
3. 分别扫描 Front 的 slow/fast 以及 U-shape 的 fast base threshold。
4. 匹配优先级固定为：平均 diffusion latency、DiT PFLOPs、按实际窗口/token
   加权的成本、原始 reuse count。目标平均 diffusion latency 误差不超过 2%。
5. 单卡串行运行，CUDA 计时边界同步；先预热，再用 3 个校准 prompt 重复计时。
   若变异系数超过 2%，补到最多 5 次并先排查机器波动。
6. 将六组最终配置、源码 hash、prompt/seed、资产 hash 和计时结果写入冻结
   manifest。冻结后不得依据正式质量结果回调阈值。

slow/fast 的具体阈值和目标 speedup 尚未冻结；它们必须来自 126-latent 实际
曲线，并在批量 VBench 前单独提交确认。

## 5. VBench 与后续 Fidelity

- 正式质量评测采用 VBench 1.0 官方 16 维、946 条 prompt。六种方法共生成
  `946 x 6 = 5676` 个原始视频，并产生一一对应的 480 帧评测视频。
- 当前 81-latent/321-frame 结果仅作为开发、正确性和方向验证材料。
- 先完成 VBench；PSNR、SSIM、LPIPS 等 paired Fidelity 后续再运行。
- Fidelity 可以直接复用上述 VBench 原始生成结果中的 110 个 prompt，无需
  重生成。110 个 prompt ID 必须在查看各方法质量分数前冻结，所有方法使用
  相同 prompt、seed/noise 和帧区间。优先沿用既有 SkyReels 110 条冻结清单；
  若无法取得，则在看结果前用固定、可复现且与质量无关的规则建立清单。

## 6. 当前状态

- R0--R5B：服务器验收通过。
- 旧 R6：取消正式资格，不运行；若已经运行，只能记为 81-latent 开发 pilot。
- 下一门槛：126-latent RF+Fixed 成本曲线与计时校准。
- 在 slow/fast 成本点经用户确认前，不生成正式 946-prompt VBench 视频。
