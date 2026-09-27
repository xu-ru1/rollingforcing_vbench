# R5B 服务器验收与当前进度

日期：2026-09-19

## R5B 结论

回传 `step_cache_r5b_result_r5b_20260919T081154Z.tar.gz` 通过全部验收，记为 **R5 SERVER_VALIDATED PASS**。

Fixed、Front、U-shape 都产生 43 次 decision reuse、1,290 次 layer reuse，算子统计完全一致：

- `full_window_tokens = 31,590,000`
- `K tokens = V tokens = 31,590,000`
- `Q/attention-output/cross-attention/MLP tokens = 25,552,800`
- `reuse_tokens = 6,037,200`
- 目标部分算子 token 削减比例为 19.111%
- 峰值 residual cache 为 2,156,544,000 bytes
- 所有末端状态清零，6 个新增视频均为 177 帧、832×480、16 fps

同预算下相对 disabled baseline 的三 prompt 平均 latent 结果：

| 方法 | 阈值 | reuse stage 分布 | MSE | relative-L1 | PSNR |
|---|---:|---|---:|---:|---:|
| Fixed | 0.2785 | S1=40, S2=1, S3=2 | 0.377527 | 0.496187 | 22.287 dB |
| Front | 0.3610 | S1=0, S2=42, S3=1 | **0.221451** | **0.355403** | **24.794 dB** |
| U-shape | 0.4585 | S1=1, S2=42, S3=0 | 0.263610 | 0.396094 | 24.357 dB |

Front 的平均 MSE 比 Fixed 低 41.34%，比 U-shape 低 15.99%。对三条 prompt，Front 均不差于 Fixed。Front 与 U-shape 的 43 个 reuse 中有 42 个相同；在 rainy-car prompt 上，Front 唯一的 stage-3 事件替代了 U-shape 唯一的 stage-1 事件，MSE 从 0.484764 降至 0.358288。这与 R4 的等剂量位置实验一致：较早的 stage-1 reuse 对最终 latent 更敏感。

本结论来自 3 条校准 prompt、单 seed 的 latent 诊断。它足以关闭成本校准和确认方法方向，但不能替代未见 prompt、不同 seed、实际时延和 VBench。

## 总体进度

| 阶段 | 状态 | 已完成内容 |
|---|---|---|
| R0 | SERVER_VALIDATED PASS | 基线、stage、RNG、81 latent |
| R1 | SERVER_VALIDATED PASS | FP32 2×2 modulation observer |
| R2 | SERVER_VALIDATED PASS | 真实 partial compute、dense oracle、算子守恒 |
| R3/M0 | SERVER_VALIDATED PASS | 数值等价、生命周期、eviction、reset、动态策略接线 |
| R4/M1 | SERVER_VALIDATED PASS | stage 1/2/3 等剂量位置敏感性 |
| R5/M2 | SERVER_VALIDATED PASS | 闭环阈值校准、精确成本匹配、Front 优势 |
| R6 | LOCAL_IMPLEMENTED / GPU PENDING | 冻结配置泛化和重复计时 |
| R7 | PENDING | 干净环境安装与单卡 smoke |
| R8 | PENDING | VBench 前资产、计划和最终 bundle |

## 精简 R6

R6 使用 frozen Fixed 0.2785 与 Front 0.3610，不再调参。运行 disabled、Fixed、Front 三种方法，每种在同一进程顺序生成 4 条 81-latent 视频：第 1 条作为 CUDA/model 预热并排除，后 3 条是未参与 R4/R5 的 prompt，同时用于 seed 1 泛化和配对 diffusion timing。

总计 12 个视频、3 次模型启动。该设计同时覆盖 81-latent KV eviction、初始噪声配对、真实在线 reuse 分布、latent 偏差、CUDA diffusion 时间和峰值显存。U-shape 已在 R5 完成同预算对照；243-latent 长视频不属于当前 VBench 前置的必要门槛，因此取消。

R6 的实用门槛是 Front 的平均 diffusion speedup 至少 5%。质量方向按未见 prompt 的 Front vs Fixed latent MSE 报告；无论正负都保留，不再回到阈值搜索。
