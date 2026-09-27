# RollingForcing 去噪位置缓存：方法与实验协议 v2

审核修订日期：2026-09-11。状态：R0 本地观测与服务器运行包已实现；尚未收到 GPU 实验回传。真实 step-cache 迁移仍未实现。

本文件替代 2026-09-05 版。旧版保存在 `rollingforcing_step_position_cache_migration_plan_v1_20260905.md`，仅供追溯，不再作为执行依据。
配套文件：`step_cache_method_review_20260911.md`（审核理由与代码证据）、`step_cache_work_packages.md`（分工、任务编号、交付物和验收）。

## 1. 目标、研究问题与完成口径

目标是在当前 RollingForcing 推理路径上建立一个可复现的 block 级计算复用基线，再验证按局部去噪位置调节阈值能否改善同计算预算下的质量。Front 是待验证主假设，U-shape 是固定对照，不能预先宣称 Front 最优。

研究问题分开回答：

1. 工程问题：真实跳过部分算子后，实际 diffusion 时间是否改善，额外缓存与 gather/scatter 开销有多大？
2. 位置问题：在相同 video block 集合和相同事件数下，stage 1/2/3 的复用造成何种偏差？
3. 方法问题：相近实际计算成本下，Front/U-shape 相比 Fixed 的质量如何？
4. 泛化问题：新 seed、不同 prompt 和更长视频上能否维持结果？

本轮终点为 `PRE_VBENCH_READY`：方法实现、局部实验、配置冻结、单卡运行、干净机器重建、生成/评测环境预检以及 VBench 正式任务计划都已验收。正式 VBench 视频批量生成和正式评分留到下一阶段。

分别记录四种状态，不能相互代替：

- `LOCAL_IMPLEMENTED`：代码与本地测试通过。
- `SERVER_VALIDATED`：收到并审阅服务器真实推理证据。
- `CLEAN_INSTALL_VALIDATED`：独立干净机器/全新环境按交付脚本成功。
- `PRE_VBENCH_READY`：全部前置门槛、资产、冻结配置和运行入口就绪。

Front 无显著优势是合法研究结果；不以“必须得到正结论”作为验收条件。但真实计算复用无法形成实用运行点时，应报告该迁移路线未达到加速目标，不能以 dry-run 通过冒充整体完成。

## 2. 现状与主实验边界

工作基线为 `E:\flowcache\RollingForcing-main` 的当前本地改版。它具备 KV 压缩、metadata、Round 6 output L1rel 诊断和 profiling；尚无真实 step-position cache。Round 8.4 的 6.25% 耗时下降属于 KV 压缩候选，不能计作本方法收益。

初版固定 T2V、batch=1、3 latent frames/video block、五个局部 stage、EMA checkpoint、832×480、16fps。实际模型维度、dtype、attention backend、warped timesteps、有效 guidance 和 negative prompt 处理由 R0 写入 manifest。配置中出现 guidance 字段不代表当前 wrapper 执行了双分支 CFG；当前缓存推理调用只有一次条件 forward，日志先使用实际分支 `conditional`。

源码没有 Git 时使用文件 SHA256 和 bundle hash。现有本地改版命名 `B0-local`，不称官方 upstream；只有取得可验证 commit 的未改官方源码并实际对比后才能报告 upstream 等价。`4.2副本` 不能自动充当官方基线。

主实验所有方法关闭额外 KV compaction，保留 RollingForcing 原生历史 KV、文本 KV 和 sink。`ratio025_clean` 仅进入独立组合消融，不能实验结束后将其混入主表配置。主表基线名为 `RF-Fixed`，表示同一新适配器的固定阈值版本，不声称复现了 MAGI/SkyReels 原始 FlowCache 的全部缓存语义。

## 3. 阶段、身份与状态生命周期

以实际 scheduler warp 后的 timestep 查表得到 `local_stage_index`，要求匹配唯一且同一 video block 内各帧 stage 一致；失败即终止。查表的数值容差及 timestep dtype 写入协议，不能用平均值掩盖混合 stage。

五阶段映射为 stage 0/1→early，2→middle，3/4→late；0 和 4 强制重算，所以三个可复用阶段的倍率实际为：

| 调度 | stage 1 | stage 2 | stage 3 |
|---|---:|---:|---:|
| Fixed | 1.0 | 1.0 | 1.0 |
| Front | 0.5 | 1.0 | 1.3 |
| U-shape | 0.5 | 1.4 | 0.7 |

此外，当前 window 的第一个 block 始终重算。原因是主 denoise forward 本身就会写该 block 的原生 KV，不能仅保护后续 clean forward。这个附加保护主要影响开头不足完整窗口的情形，Fixed/Front/U-shape 必须完全相同。

每次主 denoise 决策的身份包含 `(sample_id, window_index, global_video_block_id, local_stage_index, branch)`。`actual_timestep`、局部位置和源 config 是审计字段。决策一次覆盖所有缓存层；层级执行统计使用 `layer_index`，不重复计入决策总数。

持久状态 key 为 `(sample_id, global_video_block_id, branch)`，层残差再加 `layer_index`。key 不含 window/stage，否则无法跨 stage 复用。新 sample 清空；block 完成最终 stage 且当次 clean refresh 结束后释放；异常结束也清理。clean forward 从不更新 step metric、accumulator 或层残差。

默认五步正常 T2V：N 个 video blocks 共 N+4 个 window、5N 次主决策，每个 window 一次 clean forward。81 latent frames 对应 N=27，期望 135 次主决策和 31 次 clean forward；由实际 trace 验证，不硬套到 I2V/其他步数。

`--num_output_frames` 当前实为 latent 长度。预期解码长度为 4T−3：21→81 帧、81→321 帧、243→969 帧；必须以实际视频解码核验。正式命令应明确命名/记录 `num_latent_frames` 与 `num_decoded_frames`。

## 4. 首版方法：逐层残差复用，当前窗口 K/V 重算

### 4.1 缓存张量

对 Transformer layer l、video block c，每次重算后保存：

```text
R[l,c] = H_out[l,c] - H_in[l,c]
```

这个残差包含当次 self-attention、cross-attention、MLP 及其 modulation/gate 的整体更新。复用时：

```text
H_out[l,c] = H_in_current[l,c] + R_last_recompute[l,c]
```

残差只在真实重算后刷新；复用不把自身近似结果当成新真实缓存。重算行直接返回原始H_out，不为了统一接口重新执行H_in+R，以免BF16减加引入额外舍入。输入 H、各层残差和时间条件不同于纯最终 x0 复用，这是一种 RollingForcing 适配器选择，必须在方法说明里承认该增量。Fixed 与位置方法共享它，才能隔离阈值调度贡献。

### 4.2 每层执行顺序

1. 对整个当前窗口计算本层 norm1/modulation 和当前 K/V；K/V 来自本次 H_in，即使其中部分 H_in 已受前层复用影响，也不直接读取上一 stage 的 K/V。
2. 执行原生首 block KV 写入、索引推进和 eviction；按原窗口长度确定 history 范围和 anchor RoPE。
3. 对重算集合 C 计算 Q，按原 token/frame 绝对位置施加 RoPE；attention 的 K/V 仍是原路径应可见的完整集合。
4. 只对 C 执行 self-attention 输出投影、gate、cross-attention query 和 MLP；scatter 到原窗口位置。
5. 对复用集合 U 加上保存的层残差，组成下一层完整 H。
6. head、unpatchify、flow→x0 转换和 scheduler 更新照常计算；每个 clean-context forward 完整运行。

当前缓存推理通过 `attention(..., causal=False)` 默认路径读取整个当前窗口，并非只看前序 video blocks。不得为稀疏 query 新增三角 mask。优先 gather 全部重算 Q，一次调用对完整 K/V 的 attention；不能将 K/V 同时裁成各个独立 block。

特别注意 history 范围当前依赖 `query_length`。新接口要区分 `full_window_tokens` 与 `active_query_tokens`，压缩 Q 后不能因此扩大可见历史。不得用更改 `current_start/grid_sizes` 冒充非连续 Q 的真实 RoPE 坐标。

### 4.3 精确性的含义

全重算路径应与参考数值等价。发生复用后，视频轨迹会改变，原生 KV 的数值不要求与 Vanilla 相等。要求保持原有更新时机、索引、可见范围和完整 clean-forward 算法；对同一当前输入/KV 快照调用 clean forward，应与未开启 step cache 的参考函数一致。

实际省去 Q、attention query/output、cross-attention query/output 与 MLP 的部分计算，K/V、head、clean forward、scheduler、VAE 不计为省去。禁止用“复用比例×完整 DiT FLOPs”估算成本。理论计算下降不保证墙钟加速。

### 4.4 状态内存与失败路线

残差缓存上限约 `L × active_blocks × tokens_per_block × hidden_dim × dtype_bytes`。若实测模型为 30 层、5 blocks、4680 tokens/block、1536 hidden、BF16，仅残差约 2.01 GiB，另计 metric、工作区和峰值临时张量。参数来自 R0 模型，不用类构造默认值代替 checkpoint 配置。

R2 必须报告缓存 live/peak bytes、峰值 allocated/reserved、GPU 总使用和至少 81 latent frames 下的稳态内存。若全层缓存不满足单卡容量，先缩小可缓存 layer 集合并形成独立版本；若只做 MLP 复用，明确另命名 `mlp_residual_v1`，重跑 M0，不假称完成 attention skip。仅降低分辨率跑通不能验收冻结目标分辨率。

## 5. metric、累计规则与配置接口

第一候选 metric 为第一层 self-attention 输入的内容特征：patch embedding 经第一层 norm1 与当前 timestep modulation 后，按固定时空网格池化，每个 video block 保留多格特征；不先全局均值后相减，以减少相互抵消。采样/池化规则确定后冻结，使用 FP32 计算距离。

```text
d_s = mean(abs(f_s - f_prev)) / max(mean(abs(f_prev)), eps)
A_test = A_previous + d_s
reuse = eligible AND all_required_residuals_valid AND (A_test < tau_s)
A_after = A_test if reuse else 0
```

每次主访问都更新 f_prev，复用也更新；残差仅在重算时更新。首步、尾步、window 首 block、缺缓存、非有限距离、shape/device/dtype 不匹配一律重算并记录原因。计划性保护不算异常 fallback；NaN、未知 stage、损坏日志等应失败，不能默默继续统计收益。

这个 metric 为模型与内容相关的代理，需要验证相关性；不声称已经得到质量风险界限。Round 6 的 post-forward x0 L1rel 保留为诊断，不作决策器；不复制其他模型的经验多项式或阈值。

计划配置，当前尚未实现：

```yaml
step_cache:
  enabled: false
  policy: fixed                 # fixed | observe_only | force_stage | force_window | dynamic_threshold
  schedule: null                # front_protect | u_shape_protect
  base_threshold: 0.0
  warmup_local_stages: 1
  protect_last_stage: true
  protect_window_first_block: true
  adapter: layer_residual_current_kv_v1
  metric: first_layer_modulated_grid_l1rel_v1
  diagnostic_injection: false  # 单独的剂量控制实验，不属于部署策略
  decision_log: off            # off | summary | full
```

`enabled=false` 为 Vanilla/no-extra-cache；`enabled=true, policy=fixed` 为 RF-Fixed。`observe_only` 仅改变日志，不能另写决策代码。`base_threshold=0` 强制全重算。训练/I2V/双分支未支持时显式拒绝；API 与 legacy FlowCache 开关解耦，不因关闭 legacy compression manager 而静默失效。

## 6. 实验设计与数据隔离

### 6.1 诊断集与校准集

沿用 cat、rainy-city car、robot 三条完整英文 prompt，文本从交接源复制并做 SHA256；seed=0，81 latent frames。历史 MAGI/SkyReels 不同 seed 不影响这里逐方法配对，不能据此称逐视频跨模型公平比较。

正式确认在配置冻结后使用这三条 prompt 的 seed=1、2，并额外使用六条预先保存的未参与校准 prompt（seed=1）。所有 prompt、seed、长度在 R0 先登记，不能看结果后替换困难样本。新 prompt 只用于泛化检查，不用于重新选择配置。

### 6.2 M0：两个参考和混合掩码正确性

- B0-local：迁移前不可变本地快照，额外缓存关闭。
- Dense-reference：采用完全相同的残差/metric/决策，但计算所有行后仅用于构造相同近似状态的正确性 oracle；不参与性能表。
- Sparse-implementation：真实 gather/scatter 和部分计算路径。

检查 B0==disabled；强制全重算与 B0；Fixed quiet==observe；相同预设混合 mask 下 Sparse≈Dense-reference。最后一项专门验证 partial compute 实现，不要求有缓存的方法等于 Vanilla。双路径对比使用独立复制的输入/KV/state快照，不能先运行dense再让sparse读取已被修改的缓存；校验工具的RNG和计时也与生产路径隔离。

建议 FP32 小张量 oracle 使用 atol=1e-5、rtol=1e-4；BF16 GPU 的误差门槛先由 R0 重复基线测量形成 `numerics_gate.json`，在观察缓存质量前冻结。不能遇到失败后任意扩大容差。解码视频同时检查，但 MP4 文件 SHA 不作为唯一数值判据。

### 6.3 M1：限定可复用阶段的等剂量诊断

删除原计划无意义的 Force-stage-0/4 正式组。stage 0 缺少历史残差且被保护；stage 4 是最终重算保护，0 个事件无法用于阶段敏感性比较。

保留自然策略 `force_stage` 用于代码行为验证；真正的因果诊断另用 `diagnostic_injection`：提前固定 K 个内部 block，每视频仅在指定 stage 1、2 或 3 各复用一次，其他访问全部重算，不经 L1 阈值选择事件。三组使用同一 block 清单和相同缓存年龄（上一 stage 重算），K 相同。

81 latent frames、27 blocks 可取完整五阶段均处于满窗口的 block IDs 4..22；预设 K=8，取 IDs 4..11，并对实际 trace 校验。每 prompt 运行 Vanilla、Fixed 和三个 stage 注入组，共 5×3=15 个视频。保护规则冲突时计划生成失败，不自动减少 K。

阶段不同可能带来不同 history 长度/成本；报告同事件数和各自算子成本，不能将它直接写成等 PFLOPs。只有 stage 1 比 stage 3 更伤质量的配对证据成立，才能称本模型支持早期保护假设；相反或不明确也保留全结果。

### 6.4 M2：校准必须在线，离线回放仅作初选

内容 feature 和再加噪的输入随过去复用而改变。因而 Fixed 的距离轨迹只能帮助提出阈值候选，不能保证 Front/U-shape 在线 mask 重现该离线预测。

允许离线验证的是：每个候选自己的实际在线距离、保护状态、缓存可用性经同一累计规则重放，必须逐事件复现它自己的 mask。另单独验证全重算参考的原始决策。预测与实测差异保存，但不同策略离线预测不一致本身不判实现错误。

每方法先 5 个基于 R1 距离分布生成的阈值；最多追加 4 个，每方法最多 9 个阈值。每阈值跑 3 prompt×1 seed，Fixed/Front/U-shape 最多 81 次；必要时匹配细化最多再加 18 次，加 3 个 Vanilla 上限共 102 次。重复配置/hash 对齐后可复用已有同口径运行。若到上限仍不能匹配，提交未匹配结论，不无期限追加。

默认以实测算子 FLOPs 成本相差≤2%选 slow/fast 配对，再独立报告 latency；FLOPs 模型未验证时，以重复 diffusion 时间做匹配，不用未经验证的 PFLOPs。此时对最多6方法×3prompt的入围组各追加2次计时，最多增加36次（R5合计上限138），单次计时只能初筛。token 数只能叫计算代理，不叫等 FLOPs。

Fixed slow/fast 从实际存在的不同计算档选取，再为 Front/U-shape 找最近成本点；不先承诺 2×/3× 速度。两档若落同一决策平台，保留一档并说明，不能重复命名填六行。离散预算无法匹配时保留相邻上下点，不靠插值伪造匹配结果。

所有阈值选择只使用成本与运行有效性；候选质量在配置选定后揭盲。不用质量选阈值再报告同一集合的提升。Front 与 U-shape 均按预定方法保留，不因其输给 Fixed 而隐藏。

### 6.5 M3：冻结后的计时、质量与长视频

标准六方法：Vanilla、RF-Fixed-slow、RF-Fixed-fast、Front-slow、Front-fast、U-shape-fast。若只能建立一档，方法数 M 由实际冻结清单决定，所有期望数量重新生成。

- 3 prompt×3 seeds×6方法=54 次，质量按同 prompt/seed Vanilla 全帧配对。
- seed=0 已有一次，再补两次计时：3 prompt×2×6=36 次；与上项合计90次，不把重复计时当新增质量样本。
- 6 条新 prompt×1 seed×6方法=36次，验证泛化，不调参。
- 243 latent frames（预期约60.56秒）×2个预登记prompt×3方法（Vanilla/Fixed-fast/Front-fast）=6次，检查漂移和内存；不能据此宣称 U-shape 长视频已验收。

正式 VBench 计划的长度默认81 latent frames；如果后续要改为243等长度，须在新长度重新做成本校准和单卡容量检查，不能沿用短视频 matched 标签。

### 6.6 计时与指标

每张卡一次仅一个生成进程，按预先固定的轮换/随机顺序交错运行方法。基线插在配对批次中，记录温度、功耗、占用及主机/cgroup内存。成功运行的慢样本不事后删除；故障与污染记录保留，复跑规则预先说明。

分两种模式：诊断模式开启完整JSONL/采样profiler；计时模式关闭逐层日志、CPU逐事件同步与shadow compute，只保留所有方法相同的轻量summary。决策器自身必要的同步计入方法成本。

统一时间定义：

- `main_dit_sec`：仅主 denoise 模型调用。
- `clean_refresh_sec`：仅完整 clean-context 调用。
- `diffusion_sec`：整个 rolling loop，含主 forward、clean refresh、再加噪、缓存管理。
- `e2e_warm_sec`：模型已加载后，文本编码至视频写盘完成。
- `cold_start_sec`：独立记录安装后/首次启动模型加载、编译等开销。

CUDA阶段前后同步或使用正确event；嵌套inclusive时间不得求和。当前 `model_forward_total` 包含main与clean，需改独立标签。先预热覆盖实际形状，方法warmup策略相同。

计时报告均值、标准差、配对差异和样本数；默认只称“性能候选”的工程门槛为配对平均 diffusion 至少降低5%、e2e均值不倒退且无故障。该5%是预算控制门槛，不是统计显著性证明。2%左右的差异有较大波动时不得强称速度匹配；最多补至5次，否则记录不确定。

质量固定逐帧后逐视频再跨prompt/seed汇总：PSNR(dB)、MSE、固定skimage版本的SSIM、可用时LPIPS。明确色彩范围、分辨率、data_range、LPIPS输入归一化和模型hash；不可用写NA及原因，禁止全局SSIM fallback混成正式SSIM。帧数不一致失败，不能静默截断。bootstrap/误差区间以prompt/seed配对样本为单位，不把数百帧当独立样本。盲化接触图和代表性视频检查主体、运动与漂移；PSNR不是绝对生成质量。

## 7. 随机数、公平性和日志

相同 seed 不足以保证相同噪声。当前再加噪循环每个block生成整个window形状的随机张量，不能顺手改为只给重算block采样。首版保留完整随机调用形状、顺序与数量，step cache新增代码不消耗采样RNG。one-run-one-prompt独立启动；多prompt runner须每sample重置所有缓存和RNG。

在M0审计模式记录初始噪声hash及各window前后的RNG状态hash，重算/复用不应改变噪声流。若未来使用独立Generator或按block随机流，作为所有方法共同的新版本重新验证，不能边迁移边改变。

至少分两种hash：

- `protocol_hash/source_hash` 标识配置和代码。
- `decision_mask_hash` 对排序后的(sample identity/window/global block/stage/branch/decision)编码；排除run_id、method名、路径、时间戳与浮点timestep，否则Fixed和observe无法比较。

JSONL记录active threshold、d、accumulator前后、保护原因、缓存版本/年龄、实际重算层/算子计数。记录实际执行，不能只根据计划mask推算“已省计算”。异常fallback=0作为结果门槛，计划性重算不算错误；无关依赖警告单独分类，不要求stdout一个warning字样也不能出现。

## 8. 与已有KV压缩的关系

主结论始终使用KV compaction off。冻结后以3 prompt、seed=0、81 latent frames做六格消融：

```text
Vanilla / RF-Fixed-fast / Front-fast
各自 × {KV off, ratio025_clean}
```

共18次，验证方法×KV的交互；相同口径已有off结果可引用但注明时间批次。首先做1prompt组合smoke通过再扩展。若要宣称组合速度收益再独立补配对重复计时，不能凭该18次直接下稳健加速结论。压缩clean路径的组合组不再声称“完全未压缩clean attention”，须明示其继承的Round8行为。

## 9. 环境、交付与VBench边界

不能照搬原始requirements的训练、TensorRT、PyCUDA等整套依赖。R0导出实际推理依赖；生成与VBench评测分别建全新环境，锁Python补丁版本、PyTorch/CUDA构建、FlashAttention/解码库与wheel来源hash。若需源码编译FlashAttention，预检nvcc、编译器、Python/PyTorch ABI和目标GPU，不能承诺“只有driver即可安装”；优先提供验证过的匹配wheel。

脚本显式接收源码、Wan基础模型、tokenizer/T5/VAE、RF EMA checkpoint路径。当前源码硬编码 `wan_models/...` 和相对配置路径，需统一参数化或在bundle内工作目录建立明确的只读资产映射，不能依赖原服务器隐藏软链。解包后从任意cwd可运行，启动子进程显式cwd。

`setup_generation_env.sh`和`setup_eval_env.sh`负责全新目标环境、pip check、版本/import与GPU算子自检、文件资产清单。离线模型资产通过已有挂载提供；允许下载锁定的软件依赖。评测环境须逐一加载目标维度的已有权重并做最小前向/解码检查，缺资产就列缺口，不能用plan-only宣告评测环境通过。

干净机器使用最终候选bundle和一张卡，运行21 latent的Vanilla、Fixed-fast、Front-fast、U-shape及81 latent的Vanilla/Front，外加Fixed/observe等价检查；R3完整数值证据随bundle提供。必须生成并解码实际MP4、看到真实skip计数和0异常fallback，不是只import成功。

正式VBench预检固定与交接一致的六维子集、prompt原文/增强文一一映射、方法与seed数、预期视频命名。原入口100字符截断命名需要适配，检查碰撞和VBench元数据匹配。视频数由冻结prompt清单推导，不硬写990；评分job数由方法×维度推导。计划验证分为“无需正式视频的生成前结构检查”和“正式生成后的视频完整性检查”，后者本轮标记pending。

本轮不运行六维评分，也不输出VBench得分。清洁环境权重前向自检不等于真实VBench端到端评分已验收，报告中明确保留该边界。完成预检后准备 `run_generation.sh --plan-only`、`run_eval.sh --plan-only`；未来真实执行必须显式 `--execute`。

交付有两层hash：代码/运行资产内部manifest冻结后不因补充验收报告而改变；外层bundle收录该manifest和对应测试报告，再生成最终SHA256，避免“测试了旧代码、打包了新代码”。失败尝试保留，不覆盖已完成结果；续跑只跳过manifest匹配且校验成功的run。

R0 已实现 `scripts/run_step_cache_r0.sh`、stage/RNG 汇总、视频解码检查和 manifest；真实运行合同见 `configs/step_cache_r0_protocol_v2.json` 与 `docs/round0_step_cache_baseline.md`。其余阶段脚本仍为计划接口。服务器GPU实测由用户执行并回传，我负责本地开发、检查和审阅；没有收到证据的门槛一直标记pending。
