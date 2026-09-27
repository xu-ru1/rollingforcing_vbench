# R6D：六组候选的重复计时确认

R6D 只解决配置冻结前剩余的计时不确定性，不进入 VBench、Fidelity 或任何质量读取。
所有生成保持 126 latent、3 latent/block、seed 0；实际输出仍是 501 RGB frames。
每个 case 先运行一个独立 warmup，再对 cat、rainy-car、robot 各重复三次。

`--reset_seed_per_prompt` 使每一条测量在相同的 CPU/CUDA RNG 状态和同一初始噪声
下运行；audit hashes 验证这个条件。计时仍只取 rolling diffusion 的 CUDA 事件时间，
不包括文本编码、VAE 解码和写盘。重复间视频会生成并检查，随后删除，以控制回传包大小。

待确认的六组为：Vanilla、Fixed-slow 0.26、Front-slow 0.24、Fixed-fast 0.40、
Front-fast 0.46、U-shape-fast 0.58。slow/fast 只同对应 Fixed 锚点匹配：要求
prompt-balanced mean 的误差不超过 2%，并报告每个 prompt 三次重复的 CV。若重复
CV 本身超过 2%，不会宣称已完成 2% 时延匹配。

本轮的 operator token 统计只包含 main rolling DiT forward；clean refresh 没有
step-cache skip，必须在后续 PFLOPs 报告中以完整、统一的实际 clean forward 成本加入。
