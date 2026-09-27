# R6C：126-latent 匹配成本微调

R6B验证了实现与统计路径，但尚未满足全部正式匹配门槛：Front-slow候选偏快约
3.6%，U-shape-fast 0.56偏慢2.29%。R6C在同一批次重新运行两个Fixed锚点，并只
补充必要的阈值。

- Fixed-slow：0.26；Fixed-fast：0.40。
- Front-slow候选：0.20、0.22、0.24、0.26。
- U-shape-fast候选：0.57、0.58、0.60。

不再运行U-shape-slow，也不再重复R6B已经证实的Front-fast 0.46。每个case包含
warmup和三条校准prompt，总共36个126-latent视频。它只依据mean diffusion
latency与同批Fixed锚点的相对误差、operator cost、CV和正确性检查报告候选，
不会自动冻结配置或读取质量。
