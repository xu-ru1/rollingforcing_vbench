# R3 / M0 审核与最小 GPU 验证矩阵（2026-09-18）

状态：`R2 SERVER_VALIDATED PASS`。本文件以
`debug-GPT/rollingforcing_server_code_after_r2_20260916.tar.gz` 解包后的源码为
R2 真值；`step_cache_r3a_result_r3a_20260917T035746Z.tar.gz` 仅作未冻结的历史
线索，不作为本轮验收证据或数值门槛来源。

## 1. 同步和本地检查

同步前的本地文件保留在 `debug-GPT/local_pre_r2_sync_20260918/`；从服务器包
解出的 R2 文件摘要记录在 `debug-GPT/server_r2_synced_file_hashes_20260918.json`。
同步后完成 Python 编译和 step-cache CPU 单测。R3 审核补丁新增 observer/reset
和 terminal-release 测试后，结果为 31/31 PASS。

## 2. R2 路径审核

| 审核项 | 结论 | 代码依据与 R3 要求 |
|---|---|---|
| active-Q 的绝对 RoPE | 通过静态审核 | 稀疏 Q 按递增 local block 组成；每个 Q chunk 用 `current_start_frame + local_block_index * num_frame_per_block`。R3 sparse/dense 必须比较同一 mask 的 latent。 |
| 稀疏 Q 的 history 可见范围 | 通过静态审核 | attention 在裁剪 Q 前保存完整 `s`，并以 `query_length=full_window_tokens` 计算历史 KV；不可用 active-Q 长度代替。 |
| 真实 reuse 时 K/V | 通过静态审核 | `qkv_fn` 只 gather Q input，K/V 始终由完整 `x` 投影。R3 必须继续核验 `k_tokens=v_tokens=full_window_tokens`。 |
| residual 刷新 | 通过静态审核及 CPU 单测 | sparse 与 dense-reference 都只循环 `recompute_blocks` 更新 residual；reuse 行读取旧 residual。 |
| clean refresh 绕过 runtime | 通过静态审核 | clean generator 调用没有传 observer/runtime；`step_cache_active` 同时要求 `not updating_cache`。 |
| terminal residual 释放 | 通过静态审核及 CPU 单测 | final-stage key 在主 forward 记录，clean 后 `finish_window_after_clean()` 释放。新增测试覆盖“clean 前保留、clean 后释放”。 |
| dense-reference 的 mask | 条件通过，待 M0 GPU 封口 | 两实现共享 config、policy、context 和 state 逻辑；M0 强制初始 noise hash 与完整 decision-mask hash 相同后才比较 latent。 |
| 跨 sample/prompt 状态 | 已修正 | R2 pipeline 只清 runtime，未清 R1 observer 的 previous-feature 表。补丁使两者在样本结束时都清理，并新增 observer reset 单测。runtime 清理也会删除未完成 clean 的 terminal keys。 |

无论 sparse/dense 是否等价，发生真实复用时均不要求其结果等于 disabled baseline；M0 的该项只验证同一近似算法的两种执行实现。

## 3. 统计和内存结论

现有 token 统计足以证明 R2 的 K/V 未跳算、Q/output/cross/MLP 已按 active blocks
减少，并可做各模块 token 量比较；不足以单独得出 PFLOPs。缺少实际每层
visible-KV token、Q×KV attention pair 数、模型维度/FFN 维度、heads/head-dim 与
cross-attention context 长度。后续任何 PFLOPs 图表必须新增这些字段并分别计算
Q/K/V/O 投影、QK/AV、cross-attention 和 MLP；在此之前只报告 operator tokens，
不将 token 比例称为 FLOPs 或 speedup。

2,156,544,000 bytes（约 2.01 GiB）是全 30 层、五 block residual adapter 的真实
峰值，属于方法成本。R3 不改为压缩或缩层，避免改变已验收 adapter；81 latent
case 必须回传 peak residual bytes 和 CUDA allocated/reserved。后续论文应把该峰值
与端到端显存一起报告。若内存成为目标 GPU 的硬限制，另起版本化 adapter 并重新
从 M0 验收，不能事后混入本版本。

## 4. R3 / M0 冻结门槛

所有 case 使用 EMA checkpoint、batch=1、固定 seed=0、关闭 legacy FlowCache
compression，且不计时、不做质量优劣结论。每个 case 保存 CPU 初始 noise 与最终
latent tensor，并解码检查视频。

| 比较 | 先决条件 | 冻结门槛 |
|---|---|---|
| disabled-a vs disabled-b | initial-noise SHA256 相同 | latent shape/dtype/hash 全同，`max_abs=0`、`relative_l1=0` |
| disabled-a vs all-recompute | initial-noise SHA256 相同 | latent shape/dtype/hash 全同，`max_abs=0`、`relative_l1=0` |
| fixed-quiet vs fixed-observed | initial-noise SHA256 与 full decision-mask hash 相同 | latent shape/dtype/hash 全同，`max_abs=0`、`relative_l1=0`；observer 只生成标量 JSONL |
| mixed-sparse vs mixed-dense-reference | initial-noise SHA256 与 full decision-mask hash 相同，且实际 reuse 大于 0 | `max_abs <= 0.05` 且 `relative_l1 <= 0.005` |

除稀疏/稠密近似比较外，前三组是相同运算顺序的确定性重跑，故采用 bitwise
门槛而非事后放宽容差。任何不同 mask、不同 noise、异常 fallback、NaN/Inf、缺失
视频或缺失 required log 均为失败。

## 5. 最小 GPU 矩阵

| 子集 | case | latent 帧 | 运行数/视频数 | 必须验收 |
|---|---|---:|---:|---|
| M0 numerical seal | disabled-a、disabled-b、all-recompute、mixed-sparse、mixed-dense-reference、fixed-quiet、fixed-observed | 21 | 7 / 7 | 81 decoded 帧；上述四组对照；mixed 有真实 reuse；K/V 全量与 Q token 合同。 |
| 短边界 | enabled all-recompute | 3、6、12 | 3 / 3 | 分别 9、21、45 decoded 帧；每个不足窗口/窗口边界无索引错误。 |
| 长序列生命周期 | diagnostic mixed sparse | 81 | 1 / 1 | 321 decoded 帧、135 decisions、31 clean refresh、KV eviction trace、terminal release 后 residual state=0、peak memory。 |
| 同进程 reset | two prompt sequential fixed | 21 each | 1 / 2 | 第二 prompt 的第一访问无 previous feature/residual；sample IDs 分离；两视频均 81 decoded 帧。 |
| 策略 smoke | dynamic Front、dynamic U-shape | 21 | 2 / 2 | 两者各至少一个合法 reuse；stage 阈值符合固定倍率；所有 stage-0/4 与 window-first 保护仍生效。 |

合计为 14 次进程运行、15 个视频。这里没有 R4 的 equal-dose position experiment，
没有阈值扫描，没有 latency/speedup 结论。

M0 的 mixed mask 使用已经在 R2 验收的九个 diagnostic points。81 latent lifecycle
case 使用一个预登记的跨后半段 blocks 的 diagnostic plan，目的仅为覆盖 residual
lifecycle/KV eviction；它不进入位置敏感性或质量分析。

## 6. R3 实施顺序

1. 服务器已应用 lifecycle 补丁并通过 35 项 step-cache CPU tests；不重跑 R0/R1/R2 GPU 结果。
2. `scripts/run_step_cache_r3_m0.sh`、`scripts/assess_step_cache_r3_m0.py` 与 R3
   configs 已实现 tensor-save、mask hash、数值评估和源码 SHA256 manifest。runner
   在 GPU 运行前重跑其所在工作区的全部 step-cache CPU tests。
3. 只先执行 M0 seven-case core。通过后再执行短边界、81 lifecycle、same-process
   reset、Front/U smoke。
4. 每个子集回传独立 tar.gz。R3/M0 全部审阅通过前，不开始 R4 或 R5。
