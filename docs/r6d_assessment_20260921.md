# R6D：重复计时评估（2026-09-21）

R6D assessment 为 `ok`：35/35 CPU tests 通过，六组共 60 个视频均通过 126 latent、
501 decoded frames、16 fps、832x480、跨方法 initial-noise/RNG hash 对齐、同 prompt
重复的 final-latent hash 对齐、K/V 完整、Q/reuse 守恒及 terminal state 清空检查。

| 组别 | prompt-balanced diffusion latency | 结果 |
| --- | ---: | --- |
| Vanilla | 70,869.32 ms | 基线 |
| Fixed-slow 0.26 | 65,444.31 ms | slow 锚点 |
| Front-slow 0.24 | 65,194.77 ms | 对 Fixed-slow `-0.38%`，三条 prompt 均在 1% 内 |
| Fixed-fast 0.40 | 52,872.04 ms | fast 锚点 |
| Front-fast 0.46 | 56,443.87 ms | 对 Fixed-fast `+6.76%`，不合格 |
| U-shape-fast 0.58 | 53,351.26 ms | 对 Fixed-fast `+0.91%` |

Front-fast 的不匹配不是一次性随机波动：cat prompt 的三次重复均约 62.3 s，较
Fixed-fast 的 53.0 s 慢 17.4%；另外两条 prompt 接近锚点。这是固定 seed 0 下真实的
prompt/workload 差异，不能沿用 R6B 的 `0.46` 候选。R6D 因此确认 Fixed-slow 0.26、
Front-slow 0.24、Fixed-fast 0.40 和 U-shape-fast 0.58；仅需重扫 Front-fast。

R6E 只重跑 Fixed-fast 锚点与 Front 的 0.50、0.54、0.58，仍使用同一预热和三 prompt
各三次的 seed-0 协议。它不重跑已经达标的 Vanilla、slow 组或 U-shape，也不读取质量。
