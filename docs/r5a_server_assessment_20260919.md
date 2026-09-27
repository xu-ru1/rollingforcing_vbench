# R5A 服务器验收

日期：2026-09-19

R5A 的结构与执行正确，记为 **R5A EXECUTION PASS / COST MATCH PENDING**。

- observe-only 为完整计算，225 个决策且没有真实跳算。
- Fixed、Front、U-shape 都保持完整 K/V，满足 `Q + reuse_tokens = full_window_tokens`，末端状态清零。
- 12 个视频均成功解码为 177 帧、832×480、16 fps。
- 质量盲 dense-trace 预测分别为 44/45/45 次 reuse；真实闭环执行为 Fixed 35、Front 55、U-shape 43。真实 reuse 会改变后续 feature，因此 dense trace 只能用于初始阈值，不能直接声称成本匹配。

当前不同预算下的平均 latent MSE 为 Fixed 0.3422、Front 0.2433、U-shape 0.2636。Front 同时使用了最多计算复用，不能据此宣称其同预算质量最好。

R5B 仅补跑 Fixed 和 Front：用各自真实 trace 自回放，目标统一到 U-shape 已达到的 43 次 reuse。预选阈值为 Fixed 0.2785、Front 0.3610；服务器脚本会从回传 trace 重新计算，而不是依赖手工常数。新增 6 个视频。若三者实际 reuse 数均在 41–45 且最大差不超过 2，则冻结该成本档并进入精简 R6。
