# R6E：Front-fast 成本重新匹配

R6D 显示 Front 0.46 在 seed-0 cat workload 下复用不足，不能满足 fast 成本匹配。
R6E 将在同一运行批次重新测量 Fixed-fast 0.40，并扫描 Front 0.50、0.54、0.58。

每个 case 仍包含一个 warmup 和 cat、rainy-car、robot 各三次重复；每条 prompt 前将
CPU/CUDA RNG 重置为 seed 0。候选只根据 prompt-balanced diffusion latency、每 prompt
重复稳定性、完整 K/V、Q/reuse 守恒和 residual lifecycle 选择。目标是与 Fixed-fast
平均 latency 的绝对误差不超过 2%。R6E 不访问质量指标，不自动冻结阈值。
