# R6C：126-latent 成本匹配微调评估（2026-09-20）

R6C assessment 为 `ok`，35/35 CPU tests 通过。9 个 case 均生成 126 latent 和
501 decoded frames（16 fps、832x480），并通过 noise/RNG hash、完整 K/V、Q/reuse
守恒和 terminal state 清空检查。该轮未读取任何质量指标。

| 方法 | threshold | mean diffusion (ms) | CV | token saving | 相对对应 Fixed 的误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-slow | 0.26 | 64,715.02 | 3.56% | 20.95% | -- |
| Front-slow 候选 | 0.24 | 65,237.00 | 5.97% | 15.24% | +0.81% |
| Front-slow 候选 | 0.26 | 63,159.99 | 0.29% | -- | -2.40% |
| Fixed-fast | 0.40 | 53,288.98 | 0.39% | 39.05% | -- |
| U-shape-fast 候选 | 0.57 | 53,499.60 | 0.19% | -- | +0.40% |
| U-shape-fast 候选 | 0.58 | 53,463.35 | 0.16% | -- | +0.33% |
| U-shape-fast 候选 | 0.60 | 52,861.14 | 0.48% | -- | -0.80% |

`Front 0.24` 是唯一处于 slow 目标 2% 匹配范围内的候选；`U-shape 0.58` 保持在
fast 目标 0.33% 范围内，且相比 0.60 更保守。R6B 已确认 `Front 0.46` 与
`Fixed 0.40` 的误差为 -0.16%。但 R6C 的每个测量 prompt 只出现一次，且 slow
组 aggregate CV 偏高，因此这些阈值仍是待重复计时确认的候选，不是正式冻结结果。
