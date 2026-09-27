# FlowCache / RollingForcing 实验阶段摘要

## 做了什么

本阶段把 FlowCache 思路系统接入 RollingForcing inference 路径，按 Round 0 到 Round 8 逐步推进：先跑通 baseline，再做默认关闭的 FlowCache skeleton、KV metadata、compression candidate dry-run，随后进入真实 attention-input compression、cache-body logical read、persistent sidecar、compacted KV read buffer、output reuse dry-run，最后用 coarse/fine profiling 定位当前真正瓶颈。

核心原则一直是 inference-only：不改训练、不改 checkpoint，baseline 默认路径保持干净；只有实验配置打开时才记录 telemetry 或进入对应实验分支。

## 发现了什么

| Round | 目标 | 是否改 tensor | 是否默认关闭 | 是否通过 | 关键结论 |
| --- | --- | --- | --- | --- | --- |
| Round 0 | baseline 跑通，建立比较基准 | 否 | 是 | 通过 | 原始 RollingForcing inference 可生成视频。 |
| Round 1 | FlowCache skeleton 与默认关闭配置 | 否 | 是 | 通过 | 默认关闭路径干净，不改 attention/训练/checkpoint。 |
| Round 2 | KV metadata：sink/history/current range | 否 | 是 | 通过 | Rolling window 内 KV 区域划分可观测。 |
| Round 3A | compression candidate analysis | 否 | 是 | 通过 | 只读识别 history compression candidate。 |
| Round 3B | dry-run saving projection | 否 | 是 | 通过 | anchor_working_current 有理论 compression 潜力。 |
| Round 4 | attention-input real compression | 是，本次 attention input | 是 | 通过但无收益 | sink/current 保护有效，runtime/memory 未改善。 |
| Round 4.4-lite | no-jsonl overhead check | 是，本次 attention input | 是 | 通过但无收益 | 关闭 JSONL 后仍无明显收益。 |
| Round 5A | cache-body logical read path | 是，logical visible read | 是 | 通过但无显存收益 | visible length 降低，但主 kv_cache 固定预分配。 |
| Round 5B | persistent sidecar | 是，sidecar/read path | 是 | 通过但无收益 | 21-frame reuse 可用；81-frame reuse 被 eviction 打断。 |
| Round 5C | compacted KV read buffer | 是，read buffer | 是 | 通过但变慢 | JSONL 干净，runtime 仍约慢，重组开销抵消收益。 |
| Round 6.0 | output reuse / L1rel dry-run | 否 | 是 | 通过但负结果 | threshold <=0.10 无 candidate，不进入真实 reuse。 |
| Round 7.0 | coarse profiling | 否 | 是 | 通过 | 瓶颈主要在 denoise_loop/model_forward。 |
| Round 7.1 | sampled fine profiling | 否 | 是 | 通过 | self-attention path 成为第一嫌疑；GPU async 口径需谨慎。 |
| Round 8.0 | attention path diagnosis 准备 | 否 | 是 | 进行中/未完成 | 缺少 21/81 服务器 profile，不能下最终结论。 |

最重要的发现有三点。第一，KV compression/read-path 在 RollingForcing 上工程可行，sink/current 保护与 `current_only`/`clean_cache_update` 分支保护有效，JSONL telemetry 经过多轮修正后可信。第二，KV visible length reduction 没有转化成 runtime/peak memory 收益，主要受固定预分配 kv_cache、compact buffer 构造、index_select 和 attention 输入重组开销影响。第三，Round 7/7.1 profiling 指向 model_forward/self-attention，而不是 FlowCache logging、KV init、MLP、KV read-write 或 eviction。

## 哪些方向不值得继续

KV compression ratio 不建议继续盲调：Round 4/5 已经把 correctness、branch guard、sidecar、compacted read buffer 走通，但没有端到端收益。Output reuse 不建议进入真实实现：Round 6 低阈值下无候选，81-frame min L1rel 约 0.115，threshold <=0.10 reusable ratio 为 0，高阈值会有质量风险。

## 关键数据

| Round | 方法 | correctness | visible saving | runtime/memory 表现 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Round 4 | 本次 attention input_key/input_value 真实压缩 | attention_output_diff=0；fallback/warning=0 | 81 ratio050 overall 1.90%，applied 25.0% | 81 baseline 205.595s；ratio050 228.218s；peak 未降 | 工程可行但不是收益版本 |
| Round 5A | cache-body logical read path | warning=0 fallback=0 | overall 1.9%，applied 25.0% | runtime 239.177s；peak reserved 31.525GB | 主 kv_cache 固定预分配，显存收益不出现 |
| Round 5B | persistent sidecar | telemetry action 对齐；21-frame reuse 路径有效 | 约 overall 1.90%，applied 25.0% | 81-frame reuse 被 eviction 打断；runtime 无收益 | 正确性稳定，但不是收益版本 |
| Round 5C | compacted KV read buffer | parse_errors=0 tensor_like=0 warning/fallback=0 | ratio050 overall 1.9%，applied 25.0% | ratio050 runtime 70.716s；ratio075 runtime 70.315s；阶段结论约慢 15% | index_select/compact buffer 构造/attention 输入重组抵消收益 |

| 实验 | valid_l1rel_count | min L1rel | avg L1rel | max L1rel | threshold <=0.10 reusable ratio | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 21-frame | 84 | 0.111358 | 0.273274 | 0.818068 | 0.03:0.0, 0.05:0.0, 0.08:0.0, 0.1:0.0 | 低阈值无 candidate，不进入真实 output reuse |
| 81-frame | 324 | 0.115446 | 0.232134 | 0.824088 | 0.03:0.0, 0.05:0.0, 0.08:0.0, 0.1:0.0 | 低阈值无 candidate，不进入真实 output reuse |

| phase | runtime | ratio | 结论 |
| --- | --- | --- | --- |
| total_inference | 204.813s | 100.0% | 不是瓶颈 |
| denoise_loop_total | 139.039s | 67.9% | 主要瓶颈 |
| model_forward_total | 135.856s | 66.3% | 主要瓶颈 |
| vae_decode_total | 41.933s | 20.5% | 端到端不可忽略 |
| clean_cache_update_total | 27.376s | 13.4% | 端到端不可忽略 |
| video_save_total | 18.810s | 9.2% | 端到端不可忽略 |
| flowcache_logging_total | 0.000s | 0.0% | 不是瓶颈 |
| kv_cache_init_total | 0.018s | 0.0% | 不是瓶颈 |
| sampled attention_forward | avg 23.78ms | NA | sampled fine profiling 的第一嫌疑 |
| sampled block_forward | avg 28.87ms | NA | attention 占 block 大头 |
| sampled MLP/KV/eviction | MLP 0.24ms; KV read 0.18ms; KV write 0.58ms; eviction 0.08ms | NA | 不是首要瓶颈 |

## 下一步做什么

下一阶段应先完成 attention path diagnosis，而不是继续 KV 小修。Round 8 应把 self-attention forward 拆成 RoPE/QKV/cache assembly/padding-mask/attention kernel/output projection/cache write/clean_cache_update attention path，并用 CUDA event 或清晰 fallback timer 记录。根据结果再决定 Round 8.1 优先优化 kernel、cache assembly、padding/mask、clean_cache_update，还是转向 VAE decode/video save。

当前阶段不要表述为“显著加速成功”；更准确的表述是：完成了多轮稳定工程适配和负结果筛选，已经把下一阶段优化方向从 KV compression/output reuse 收敛到 attention path diagnosis 与端到端 profiling。
