# R6A Fixed 成本曲线评估（2026-09-20）

R6A 使用官方发布的126-latent规格、3 latent/block、seed 0，以及R4/R5的
cat、rainy-car、robot校准prompt。每个case包含一条独立warmup和三条测量prompt，
并在前后各运行一次Vanilla。

## 验收结果

- assessment status：`ok`。
- CPU step-cache tests：35/35 PASS。
- 7个case全部完成；28个视频全部通过501 decoded frames、16 fps、832x480检查。
- 各方法同prompt的initial noise、推理前后CPU/CUDA RNG hash完全一致。
- K/V始终为完整窗口，`Q + reuse_tokens = full_window_tokens`。
- 每个Fixed case均为184次main forward、5520次layer call，terminal residual/state为0。
- Vanilla start/end diffusion均值相对漂移：0.4909%。
- 峰值residual bytes：2,156,544,000（约2.01 GiB），与R5B一致。

## 实测曲线

以Vanilla start/end均值的平均值70,544.065 ms为参考：

| Fixed threshold | mean diffusion (ms) | CV | reuse decisions | partial operator token saving | latency change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 71057.55 | 0.46% | 0 | 0.00% | -0.72% |
| 0.26 | 64621.95 | 3.39% | 176 | 20.95% | +9.16% |
| 0.32 | 56559.71 | 8.86% | 290 | 34.52% | +24.72% |
| 0.40 | 53199.74 | 0.70% | 328 | 39.05% | +32.60% |
| 0.50 | 53568.51 | 0.35% | 328 | 39.05% | +31.69% |

## 解释与下一步

0.26可以作为slow锚点，但其三条prompt的CV为3.39%，正式冻结前需要重复计时；
0.40是当前最清晰的fast锚点。0.32的时延波动超过2%，先不采用；0.50与0.40
产生相同复用mask且更慢，删除为重复平台。

下一轮R6B只扫描Front/U-shape的真实运行成本，并重新包含Vanilla及Fixed 0.26、
0.40锚点。阈值选择只看时延、算子统计和运行稳定性，不读取质量结果。
