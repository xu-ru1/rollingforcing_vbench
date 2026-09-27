# FlowCache / RollingForcing 阶段性交付报告

## 1. 项目背景

本阶段工作的目标是把 FlowCache 思路以 inference-only 的方式适配到 RollingForcing 视频生成流程中，验证 KV compression、cache read path、persistent sidecar、compacted KV buffer、output reuse dry-run 以及 profiling 诊断链路是否能稳定接入，并据此判断下一阶段该继续优化哪里。

需要强调：当前结果是阶段性研究结果，不是最终加速成功声明。多轮实验已经证明工程接入可行、默认路径可保持干净，但主要压缩/reuse 方向暂未得到端到端 runtime 或 peak memory 收益。

## 2. 范围与设置

本报告只基于当前仓库已有 `docs/`、`logs/`、`configs/`、`scripts/` 文件整理；没有重新跑 inference 长实验，没有继续实现新算法，也没有全量重读代码。日志主要来自 21-frame smoke、81-frame single/3-prompt benchmark、Round 5C summary、Round 6 JSONL、Round 7/7.1 profiling report。当前目录不是 git repository，`git status` 无法作为最终变更依据，需要以服务器提交包或文件清单交叉确认。

## 3. 整体技术路线

路线从“只读观测”逐步推进到“局部真实读路径改造”，再回到 profiling 定位瓶颈：Round 1-3 先保证默认关闭路径和 metadata/candidate telemetry；Round 4-5 验证 KV compression 家族在 RollingForcing 里的正确性和边界；Round 6 验证 output reuse 的候选质量；Round 7 定位端到端瓶颈；Round 8 只作为 attention path diagnosis 的准备阶段记录。

## 4. Round 总览

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

## 5. Round-by-Round 结果

### Round 0：baseline

原始 RollingForcing inference 能跑通并生成 baseline 视频，建立后续比较基准。早期 `logs/baseline_smoke.log` 没有完整 EvalMetrics，因此 runtime/peak memory 无法严格确认，只能确认 smoke 路径成功且无 FlowCache 依赖。

### Round 1：FlowCache skeleton

新增默认关闭的 FlowCache 配置与 manager 接入，目标是让后续 telemetry 和实验都能通过 config 开关控制。核心结论是 baseline 默认路径保持干净，不改 attention、训练或 checkpoint。

### Round 2：KV metadata

记录 RollingForcing 中 sink/history/current KV range，验证每个 window/layer 的 KV 可见区间划分。该阶段只读 metadata，不改 tensor，为后续 compression candidate 和分支保护提供依据。

### Round 3A / 3B：candidate analysis 与 dry-run saving projection

Round 3A 只读 attention/KV 信息并识别 history compression candidate；Round 3B 进一步估算 projected visible saving。结论是 anchor_working_current 分支确有 history compression 潜力，但这仍只是理论 saving，不代表 runtime 或 memory 会下降。

### Round 4：attention-input real compression

Round 4 第一次真实压缩本次 attention 的 `input_key/input_value`，保护 sink/current，并默认不压缩 `current_only` 与 `clean_cache_update`。81-frame ratio050 的整体 visible saving 约 1.9%，applied saving 25.0%，warning/fallback 为 0，attention output diff 记录为 0。但 81-frame baseline runtime 205.595s，ratio050 runtime 228.218s，未得到端到端收益。

### Round 4.4-lite：no-jsonl check

关闭 JSONL 后，81-frame ratio050 no-jsonl runtime 为 209.238s，仍无明显性能收益。因此 logging 不是唯一原因，attention-input 临时压缩不是性能收益版本。

### Round 5A：cache-body logical read path

Round 5A 将 visible history length 降低到 logical read path 中，81-frame 3-prompt runtime 239.177s，overall visible saving 1.9%，warning/fallback 为 0。但主 `kv_cache` 仍固定预分配，peak reserved 31.525GB 未下降，因此显存收益无法兑现。

### Round 5B：persistent sidecar

Persistent sidecar create/reuse/invalidate 路径稳定，21-frame prefix reuse 被验证，5B.2 修复了 prefix reuse，5B.3 修复了 telemetry summary 与 JSONL action 对齐。81-frame single 中已有 summary 显示 create/reuse/invalidate 事件可追踪，但长视频 reuse 会被 eviction 打断，不能作为收益版本。

### Round 5C：compacted KV read buffer

Round 5C 的 ratio050/ratio075 JSONL 均 `parse_errors=0`、`tensor_like=0`、`warning_count=0`、`fallback_count=0`，`current_only_applied=0`、`clean_cache_update_applied=0`。ratio050 overall visible saving 1.9%，applied saving 25.0%；ratio075 overall visible saving 1.0%，applied saving 12.5%。但阶段结论显示 runtime 仍约慢 15%，说明 `index_select`、compact buffer 构造和 attention 输入重组开销抵消了 token reduction 收益。

### Round 6.0：output reuse / L1rel dry-run

Round 6 只做 dry-run，不改变生成结果。81-frame JSONL summary：valid L1rel count=324，min/avg/max L1rel=0.115446/0.232134/0.824088，threshold 0.03/0.05/0.08/0.10 下 reusable ratio 全为 0。因此低阈值 output reuse 暂不成立，高阈值虽可能出现候选但质量风险大，不进入真实 output reuse。

### Round 7.0：coarse profiling

Round 7 证明主要 wall-clock 时间在 denoise/model forward。81-frame 3-prompt profiling runtime 197.741s，VAE decode 约 20%，video save 约 10%。FlowCache logging 与 KV cache init 很小，不是瓶颈。

### Round 7.1：sampled fine profiling

81-frame 3-prompt 结果：total runtime 204.813s，model_forward_total 135.856s，占 66.3%；denoise_loop_total 139.039s，占 67.9%；VAE decode 41.933s，占 20.5%；clean_cache_update 27.376s，占 13.4%；video save 18.810s，占 9.2%。

Sampled block_forward avg 28.87ms，attention_forward avg 23.78ms，MLP avg 0.24ms，KV read/write/eviction 均远小于 attention_forward。注意 Round 7.1 未启用全局 CUDA synchronize，因此这些 fine timing 是方向判断，不是 kernel 级最终结论。

### Round 8.0：attention path diagnosis 准备

当前仓库已有 Round 8.0 attention path diagnosis 的文档、配置、脚本和本地检查报告，但缺少 `logs/round8_0_21_attention.*` 与 `logs/round8_0_81_single_attention.*` 服务器实验结果。因此本报告只把 Round 8 写为“正在准备/部分中断/未形成最终结论”，不能声称已经证明 attention kernel 本体慢。

## 6. KV Compression 系列对比

| Round | 方法 | correctness | visible saving | runtime/memory 表现 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Round 4 | 本次 attention input_key/input_value 真实压缩 | attention_output_diff=0；fallback/warning=0 | 81 ratio050 overall 1.90%，applied 25.0% | 81 baseline 205.595s；ratio050 228.218s；peak 未降 | 工程可行但不是收益版本 |
| Round 5A | cache-body logical read path | warning=0 fallback=0 | overall 1.9%，applied 25.0% | runtime 239.177s；peak reserved 31.525GB | 主 kv_cache 固定预分配，显存收益不出现 |
| Round 5B | persistent sidecar | telemetry action 对齐；21-frame reuse 路径有效 | 约 overall 1.90%，applied 25.0% | 81-frame reuse 被 eviction 打断；runtime 无收益 | 正确性稳定，但不是收益版本 |
| Round 5C | compacted KV read buffer | parse_errors=0 tensor_like=0 warning/fallback=0 | ratio050 overall 1.9%，applied 25.0% | ratio050 runtime 70.716s；ratio075 runtime 70.315s；阶段结论约慢 15% | index_select/compact buffer 构造/attention 输入重组抵消收益 |

## 7. Output Reuse 结果

| 实验 | valid_l1rel_count | min L1rel | avg L1rel | max L1rel | threshold <=0.10 reusable ratio | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 21-frame | 84 | 0.111358 | 0.273274 | 0.818068 | 0.03:0.0, 0.05:0.0, 0.08:0.0, 0.1:0.0 | 低阈值无 candidate，不进入真实 output reuse |
| 81-frame | 324 | 0.115446 | 0.232134 | 0.824088 | 0.03:0.0, 0.05:0.0, 0.08:0.0, 0.1:0.0 | 低阈值无 candidate，不进入真实 output reuse |

## 8. Profiling 结果

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

## 9. 负结果与原因分析

1. KV compression 工程可行但当前版本无性能收益：attention input 临时压缩、logical read、sidecar、compacted read buffer 都能保护 sink/current 并保持 telemetry 干净，但固定预分配 kv_cache、read buffer 构造、index_select、reshape/assembly 开销会抵消 token reduction。
2. Output reuse 低阈值不成立：threshold <=0.10 下 reusable ratio 为 0；提升阈值会直接带来生成质量风险。
3. 继续盲目调 compression ratio 的边际价值低：Round 4/5 已经证明 correctness 与 branch guard，不缺更多小修，而缺真正瓶颈定位。
4. Round 7 将瓶颈指向 model_forward/self-attention，但现有 sampled timing 受 GPU async 影响，需要 Round 8 用 CUDA event/更细 phase 继续诊断。

## 10. 当前阶段结论

本项目不是只停留在想法层面，已经完成多轮 inference-only 工程适配。baseline 默认路径始终保持干净；FlowCache 的 KV compression 思路可以稳定接入 RollingForcing；sink/current 保护有效；`current_only` / `clean_cache_update` 默认不压缩的分支约束有效；JSONL telemetry 经过多轮修正后可用于阶段分析。

但当前没有证据支持“FlowCache 已经带来显著加速”。更准确的结论是：KV visible length reduction 工程可行但当前实现不是性能收益版本；低阈值 output reuse 暂不成立；真正瓶颈更可能在 model_forward/self-attention，同时 VAE decode 与 video save 对端到端也有优化价值。

## 11. 下一阶段建议

| 方向 | 依据 | 风险 | 优先级 |
| --- | --- | --- | --- |
| attention path diagnosis / kernel vs assembly 拆分 | Round 7.1 attention_forward avg 23.78ms，model_forward 占 66.3% | Round 7.1 采样未全局同步，需 Round 8 CUDA event 验证 | P0 |
| cache assembly / padding/mask profiling | KV read/write 单点不慢，但 attention forward inclusive 很大 | 拆分不能改变 tensor/attention 输出 | P0 |
| clean_cache_update attention/KV 路径 | 81 3-prompt clean_cache_update 27.38s，占 13.4% | 需要区分 attention/KV 写入与其它更新开销 | P1 |
| VAE decode | 41.93s，占 20.5% | 不属于 FlowCache 主线，但端到端收益可能明显 | P1 |
| video save | 18.81s，占 9.2% | 受 I/O 环境影响，需要单独确认 | P2 |
| 继续调 KV compression ratio | Round 4/5 已证明正确但无收益 | 容易消耗时间且收益证据不足 | 暂缓 |
| 真实 output reuse | Round 6 低阈值无 candidate | 高阈值质量风险大 | 停止 |

推荐顺序是：先完成 Round 8 attention path diagnosis，把 RoPE/QKV/cache assembly/padding-mask/attention kernel/output projection/cache write/clean_cache_update attention path 拆清楚；若 kernel 本体占主导，再考虑 attention kernel 方向；若 cache assembly/padding 占主导，则优化数据准备；若 clean_cache_update 占主导，则单独拆 KV 写入与更新；如果 attention path 收益有限，再转向 VAE decode 或 video save。暂不建议继续调 KV compression ratio，也不建议进入真实 output reuse。

## 12. 风险与局限

- 部分早期日志没有 EvalMetrics，runtime/peak memory 只能标为缺失。
- Round 7.1 fine profiling 未做全局 CUDA synchronize，适合方向判断，不适合作 kernel 级结论。
- 当前本地目录不是 git repository，无法用 `git status` 严格确认改动清单。
- Round 8 服务器 attention profile 缺失，不能把 attention kernel 慢写成最终结论。
- 视频目录 `ls -lh` 与部分 peak memory 指标仍需服务器补齐。

## 13. 附录：关键文件

- Docs：`docs/round0_baseline_plan.md` 到 `docs/round8_0_attention_path_diagnosis.md`，以及本报告。
- Configs：`configs/rolling_forcing_dmd_flowcache_round4*.yaml`、`round5*.yaml`、`round6_0_dryrun.yaml`、`round7_0_profile.yaml`、`round7_1_fine_profile.yaml`、`round8_0_attention_profile.yaml`。
- Scripts：`scripts/summarize_round5c.py`、`scripts/summarize_round7_profile.py`、`scripts/summarize_round7_1_profile.py`、`scripts/summarize_round8_0_attention_profile.py`。
- Logs/JSONL：见 `logs/flowcache_stage_metrics_summary.csv` 与 `logs/flowcache_stage_report_notes.txt`。
