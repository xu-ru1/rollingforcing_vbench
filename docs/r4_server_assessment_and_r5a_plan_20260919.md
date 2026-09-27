# R4 服务器验收与精简 R5A 方案

日期：2026-09-19

## R4 结论

服务器回传 `step_cache_r4_result_r4_position_20260919T072128Z.tar.gz` 通过全部结构验收，状态记为 **R4 SERVER_VALIDATED PASS**。

- 3 条 prompt、45 latent、seed 0；每个方法均生成 3 个可解码视频，每个视频 177 帧、832×480、16 fps。
- stage 1/2/3 都严格在预登记的 global block 4..11 注入，每条 prompt 8 次，共 24 次决策、720 次 layer reuse。
- 三组的 K/V、Q、attention output、cross-attention、MLP 和 reuse token 统计完全相同。每组 `full_window_tokens=31,590,000`，`reuse_tokens=3,369,600`，目标部分算子的 token 削减比例为 10.667%。
- 初始噪声逐 prompt 完全一致，末端状态全部清零，峰值 residual cache 均为 2,156,544,000 bytes。

相对 disabled baseline 的 mean latent MSE：

| 注入位置 | MSE | relative-L1 | dynamic-range PSNR |
|---|---:|---:|---:|
| stage 1 | 0.245853 | 0.327844 | 24.469 dB |
| stage 2 | 0.122970 | 0.211209 | 27.645 dB |
| stage 3 | 0.105621 | 0.193365 | 28.741 dB |

三条 prompt 的 MSE 排序都为 `stage3 < stage2 < stage1`。在相同 block、相同注入次数和完全相同算子预算下，stage 1 的平均 MSE 是 stage 3 的 2.33 倍，因此本轮证据支持“较早的可复用位置更敏感”，也支持 Front 对 stage 1 使用更严格阈值的方向。

该结论仅是配对 latent 扰动诊断，不等价于感知质量或 VBench 结论。R1 中 stage 3 的 modulation feature relative-L1 更大，而 R4 中 stage 1 的最终 latent 扰动更大，说明原始 feature distance 不能直接当作位置敏感性；阈值需要同时考虑在线距离和 stage 位置。

## R5A 精简方案

旧方案的 5–9 个阈值/方法盲扫最多需要 102 个视频，现取消。R5A 只做一次质量盲、成本匹配的在线筛选：

1. 对 R4 的同 3 prompt、seed 0、45 latent 跑一次 dense observe-only，共 3 个视频，记录 225 个在线距离事件。
2. 选择器只读取 decision trace，不读取 latent 或视频。它分别回放 Fixed、Front、U-shape，在 0.0005 的阈值网格上选择最接近 45 次 reuse 的 base threshold；目标约对应 20% 的 Q/attention-output/cross-attention/MLP token 削减。
3. 每种策略只运行一个真实 sparse 候选，共 9 个视频；复用 R4 disabled baseline，不重复生成 baseline。
4. 验证实际在线 reuse 数、stage 分配、完整 K/V、token 守恒、状态释放、视频解码和配对 latent 误差。

总新增量为 12 个视频、4 次模型启动。若三个真实候选的 reuse 数彼此相差不超过 2 且各自距目标不超过 2，直接冻结这个成本档；否则只对偏离的方法补跑一个阈值，不恢复大规模扫描。

R5A 结果用于选择 R6 的方法与成本档，不作正式稳定 speedup 声明。R6 再对冻结候选重复计时。
