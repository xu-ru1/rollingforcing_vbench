# FlowCache / RollingForcing 阶段结论（截至 Round 8.4）

## 一句话结论

本阶段已经完成 FlowCache 在 RollingForcing inference 路径上的系统性工程适配与多轮筛选。当前最好的候选是 `ratio025_clean`：它在重复 81-frame 3-prompt 实验中给出 **6.25% 端到端加速**，日志与基础视觉检查均干净，但这不是强 20%+ 加速，也没有带来端到端 peak CUDA memory 下降。

因此，当前结论应表述为：

> FlowCache compacted-token reduction 在 RollingForcing 上已证明工程可行，并出现可复现的小幅端到端加速信号；最佳候选为 `ratio025_clean`。但收益幅度有限、显存峰值未下降，后续应优先减少 clean-cache-update compaction overhead，而不是继续大规模扫 compression ratio。

## 当前最佳候选

`ratio025_clean` 的配置含义：

- 历史 KV token 保留比例：25%。
- 同时作用于 `anchor_working_current` 和 `clean_cache_update`。
- 保护 sink/current，不压缩 `current_only`。
- 不改变训练、不改 checkpoint，仅在 inference 开关开启时进入实验路径。

关键稳定性：

- checkpoint 正常加载：`generator_ema`。
- `fallback_count=0`。
- `warning_count=0`。
- 无 Traceback/OOM/RuntimeError。
- 视频基础元信息正常：832x480、16fps、321 frames、约 20.06s。
- 缩略图 sanity check 未见噪声崩坏，内容与三个 prompt 对齐。

## 关键实验结果

### Round 8.0：attention path diagnosis

Round 8.0 将 self-attention path 拆到 `attention_kernel`、`clean_cache_update_attention`、QKV、RoPE、KV assembly、padding/mask、output projection、KV write 等阶段。

81-frame single-prompt attention profile 中：

| Phase | Total |
| --- | ---: |
| `attention_kernel` | 337.466ms |
| `clean_cache_update_attention` | 201.717ms |
| `qkv_or_input_projection` | 56.870ms |
| `rope` | 54.087ms |
| `kv_cache_read_or_assembly` | 15.889ms |

这说明下一步优化不应再盲目调 KV ratio，而应围绕 attention kernel 与 clean-cache-update path。

### Round 8.1 / 8.2：快速候选筛选

81-frame single-prompt baseline：

- baseline：84.679s。
- `ratio025_denoise`：69.227s，18.25% speedup。
- `ratio025_clean`：76.169s，10.05% speedup。

单 prompt 上 `ratio025_denoise` 更快，但它在 3-prompt 复验中未过 5% 门槛。

### Round 8.3：3-prompt 初次复验

3-prompt baseline：

- baseline：248.964s。

候选复验：

| Candidate | Runtime | Speedup |
| --- | ---: | ---: |
| `ratio025_denoise` | 239.389s | 3.85% |
| `ratio025_clean` | 188.044s | 24.47% |

`ratio025_clean` 初次复验非常强，但为了防止运行波动误判，又做了 Round 8.4 back-to-back repeat。

### Round 8.4：back-to-back repeat

同 GPU back-to-back 重复 81-frame 3-prompt baseline 与 `ratio025_clean`：

| Run | Runtime | Saved videos | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| baseline repeat | 241.111s | 3 | 24.746GB | 31.525GB |
| `ratio025_clean` repeat | 226.052s | 3 | 24.746GB | 31.525GB |

重复实验结论：

- speedup：6.25%。
- runtime reduction：15.06s。
- passes 5% gate：yes。
- passes 15% gate：no。

因此，上一次 24.47% 的大幅收益不能作为稳定收益宣称；更稳妥的结论是 `ratio025_clean` 有可复现的小幅加速。

## `ratio025_clean` 的 token reduction 统计

Round 8.4 clean repeat 的 compaction summary：

| Metric | Value |
| --- | ---: |
| applied events | 4500 |
| clean-cache-update applied | 2250 |
| overall weighted visible saving ratio | 0.3465 |
| applied weighted visible saving ratio | 0.3750 |
| compacted original history token ratio | 0.2500 |
| fallback count | 0 |
| warning count | 0 |

这说明 token reduction 确实发生，而且实现路径稳定。但 34.65% visible-token saving 只转化成 6.25% 端到端 runtime speedup，说明剩余瓶颈和 compaction overhead 都很重要。

## 为什么不能写成“显著加速成功”

当前证据不支持强表述，原因有三点：

1. 24.47% 的 3-prompt 加速没有在 Round 8.4 重复实验中复现。
2. 重复实验加速为 6.25%，是正向但偏小的端到端收益。
3. end-to-end peak CUDA allocated/reserved 基本不变，说明当前方式没有带来实际峰值显存下降。

推荐表述：

> `ratio025_clean` 是当前最优 FlowCache 候选，在重复 81-frame 3-prompt 实验中实现 6.25% 端到端加速，且 warning/fallback 为 0、基础视觉检查通过。该结果说明 compacted-token reduction 方向存在正向信号，但收益仍受 compaction overhead、attention kernel、clean-cache-update path 和固定预分配 KV cache 限制。

不推荐表述：

> FlowCache 已显著加速 RollingForcing。

## 负结果与已收束方向

以下方向不建议继续投入大量实验：

- 低阈值 output reuse：Round 6 中 threshold <= 0.10 reusable ratio 为 0。
- 早期 attention-input real compression / cache-body logical read / persistent sidecar：工程可行但 runtime/memory 无收益。
- 大规模 ratio sweep：Round 8.4 已经说明 token saving 与端到端 runtime 不线性，继续横向扫比例的边际价值低。

## 下一阶段建议

优先级最高的是减少 `ratio025_clean` 的 overhead，而不是换更多 ratio：

1. 拆分 clean-cache-update compaction 的开销：
   - `index_select`
   - `torch.cat`
   - contiguous/materialization
   - attention kernel 本体
2. 优化/复用 clean-cache-update 的 compacted buffer，减少每次重复构造。
3. 继续用 3-prompt back-to-back repeat 作为主要验收口径。
4. 若 overhead 降不下来，应转向 attention kernel/backend 或 VAE/video-save，而不是继续压 KV token。

建议下一阶段成功门槛：

- 3-prompt repeated speedup >= 10%。
- warning/fallback = 0。
- 视频基础视觉检查通过。
- end-to-end peak memory 不恶化。

## 最终阶段判断

本阶段可以收束为：

1. FlowCache/RollingForcing 适配成功，默认关闭路径和实验开关路径都具备可运行性。
2. 多个早期方向被有效筛掉，避免继续在无收益路径上消耗时间。
3. Round 8 attention diagnosis 将瓶颈收敛到 attention kernel 与 clean-cache-update path。
4. `ratio025_clean` 是当前唯一保留的正向候选，重复实验显示 6.25% 小幅端到端加速。
5. 当前还不是最终高收益版本，后续工作应围绕 overhead reduction 和 kernel/path 级优化推进。
