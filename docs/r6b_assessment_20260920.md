# R6B Front/U-shape 成本曲线评估（2026-09-20）

R6B的assessment为`ok`，CPU tests 35/35 PASS。11个case全部通过126 latent、
501 decoded frames、16 fps、832x480、noise/RNG hash、K/V完整计算、Q/reuse守恒、
terminal state清空等检查。

Vanilla平均diffusion latency为71,551.57 ms，CV 0.09%。

| 方法 | threshold | mean diffusion (ms) | CV | token saving | 与对应Fixed目标的时延误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-slow | 0.26 | 65,517.32 | 3.75% | 20.95% | -- |
| Front-slow候选 | 0.28 | 63,170.43 | 0.98% | 19.52% | -3.58% |
| Front-slow候选 | 0.34 | 63,155.81 | 0.40% | 21.90% | -3.60% |
| Fixed-fast | 0.40 | 53,433.56 | 0.55% | 39.05% | -- |
| Front-fast候选 | 0.46 | 53,345.72 | 0.16% | 38.93% | -0.16% |
| U-shape-fast候选 | 0.56 | 54,654.35 | 0.56% | 38.69% | +2.28% |

其余中间点（Front 0.40、U-shape 0.50）CV超过7%，不纳入匹配候选；U-shape
0.38/0.44均停留在同一低复用平台。

结论：Front-fast 0.46已满足与Fixed-fast 0.40的2%平均latency匹配；
Front-slow与U-shape-fast仍需微调。Fixed-slow的三prompt离散较大，最终冻结前
必须以每prompt重复测量区分机器抖动和prompt相关的真实工作量差异。
