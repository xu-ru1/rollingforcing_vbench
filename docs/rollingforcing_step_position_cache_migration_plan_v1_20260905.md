# Step-position-aware FlowCache → RollingForcing 迁移与 VBench 前验收计划

日期：2026-09-05  
依据：`E:\flowcache\rollingforcing_step_position_cache_handoff.md`、`E:\FlowCache-main` 的 MAGI-1/SkyReels 实现与交付流程，以及当前 `RollingForcing-main` 的 FlowCache Round 0–8.4 代码和实验记录。

## 1. 目标与验收边界

本轮目标是在 RollingForcing 中实现真实节省 DiT 计算的逐视频 block 跨局部去噪阶段缓存，并在基础固定阈值缓存之上加入 step-position-aware 策略。最终主策略为 Front Protect，U-shape 保留为对照：

```text
Fixed          early/middle/late = 1.0 / 1.0 / 1.0
Front Protect  early/middle/late = 0.5 / 1.0 / 1.3
U-shape        early/middle/late = 0.5 / 1.4 / 0.7
```

本轮验收终点是“可以开始正式 VBench”，具体包括：

- 迁移代码、配置、单测和统计工具完成；
- 上游、hook-disabled、Fixed、observe-only 和 Front 的短视频链路通过；
- M1 逐 stage 位置敏感性诊断完成；
- M2 等预算 Fixed/Front/U-shape 验证完成；
- slow/fast 配置经重复计时和非 VBench 质量指标筛选后冻结；
- RollingForcing 现有 `ratio025_clean` 与新 step cache 的兼容性结论明确；
- 干净机器使用一张 GPU，按交付包中的 `.sh` 从环境安装到短视频生成完整跑通；
- 正式 VBench 的 prompt、方法矩阵、生成计划和评分计划能够 dry-run 并通过静态完整性检查；
- 交付包、manifest、SHA256、服务器运行命令和回传清单齐全。

本轮不生成正式 VBench 全量视频，不启动 VBench 打分，也不依据 VBench prompt 或分数重新调阈值。

## 2. 当前基础与缺口

### 2.1 可以直接复用的 RollingForcing 能力

- `utils/flowcache.py` 已有统一配置、JSONL、运行时间、显存、warning/fallback 和 profiler 框架。
- `pipeline/rolling_forcing_inference.py` 已能记录 window/block 身份和实际 timestep，并在每个主 DiT forward 后按 block 计算 output L1rel。
- Round 7/8 已拆分主 forward、clean-cache update 和 attention path 的耗时。
- 现有 KV 压缩候选 `ratio025_clean` 在重复 81-frame、3-prompt 测试中有 6.25% 端到端加速，warning/fallback 为 0，但峰值显存不降。
- 默认配置中所有实验功能均关闭，适合继续保持 disabled 路径等价。

### 2.2 必须补齐的能力

- 当前 `output_reuse_enabled` 没有形成真实复用路径；Round 6.0 只是主 forward 完成后的只读诊断。
- Round 6.0 比较的是相邻访问的最终 `denoised_pred`，阈值不超过 0.10 时没有候选。它不能直接否定 step-position 方案，但说明不能照搬旧 metric 和阈值。
- 当前没有独立的 step policy 模块、逐 stage 强制策略、累计距离状态和 decision-mask hash。
- 当前主 DiT 一次处理整个 rolling window。只在 forward 后替换某个 block 的输出不会加速，必须找到能在重计算前跳过 block token 的接入方式。
- 当前没有干净机器可重建环境、单卡端到端 smoke、冻结配置和 VBench 前交接包。
- 当前目录不是 Git 仓库，无法依赖 `git diff` 给出权威改动范围；交付时必须以源码 manifest 和文件 SHA256 补足可追溯性。

## 3. 迁移中的固定技术决策

### 3.1 step position 按 block 的实际 timestep 定义

决策身份至少包含：

```text
(run_id, prompt_id, seed, window_index,
 global_video_block_id, local_block_index,
 local_stage_index, actual_timestep, branch_id)
```

`local_stage_index` 由 block 的 `actual_timestep` 与 `denoising_step_list` 显式匹配得到。禁止使用 `window_index` 或取模推断阶段。

默认 5 个局部阶段先逐 stage 记录，再聚合为：

```text
stage 0,1 -> early
stage 2   -> middle
stage 3,4 -> late
```

启用任何 Front/U-shape 实验前，必须用轨迹 CSV 人工核对：同一个 global block 是否从高噪声 stage 0 依次走到低噪声 stage 4。

### 3.2 metric 不沿用 Round 6 的最终输出 L1rel

第一候选 metric 是 patch embedding 与 timestep modulation 之后、进入重型 Transformer blocks 之前的 block-pooled feature。它具备以下性质：

- 能在缓存决策前获得；
- 对每个 global video block 独立；
- 相邻局部阶段 shape 稳定；
- 计算成本远小于完整 DiT；
- 包含当前 latent 内容，避免只依赖所有 prompt 近似相同的 timestep embedding。

先记录 raw relative L1，不复制 MAGI/SkyReels 的模型专用多项式或阈值。是否做标定只依据 RollingForcing 自己的轨迹，且标定过程不查看候选视频质量。

### 3.3 五步模型采用短 warmup

MAGI/SkyReels 的 warmup 次数不能直接复制。RollingForcing 每个 block 只有 5 个局部阶段，首版固定：

```text
stage 0：强制重算并建立缓存
stage 4：强制重算并提交最终状态
stage 1/2/3：允许参与缓存决策
```

这等价于一个局部阶段 warmup 和最终阶段 cutoff。若 stage 方向审计得到不同结论，以实际噪声顺序修正，不能只改名称。

### 3.4 clean-context 与原生缓存保持精确

首版始终精确执行 `updating_cache=True` 的 clean-context forward，不改变：

- `kv_cache_clean`；
- `crossattn_cache`；
- attention sink；
- cache eviction；
- scheduler/noisy-cache 更新。

step cache 与当前 KV compaction 是两个正交开关。M0–M2 先关闭 KV compaction，单独证明位置策略；随后再做兼容性矩阵。

### 3.5 必须真实减少计算

缓存命中必须在重型 Transformer 计算前生效，并由 profiler 证明对应 block 的 attention/MLP 工作被跳过。输出后覆盖只能用于分析，不能进入正式 `Fixed/Front/U-shape` 方法。

首选实现是 layer-wise block token cache：

1. 在 patch/time feature 后为每个 global block 做一次决策；
2. 对 reuse block 从每层缓存读取 hidden/K/V 或该层输出；
3. 只对 recompute block 执行该层 Q、attention query、MLP；
4. 将 recompute 结果与 reuse block 状态拼成下一层完整窗口；
5. 保留 active block 对历史 clean KV 和窗口内前序 block 的可见性；
6. 逐层记录实际计算 token 数和节省 token 数。

若 FlashAttention/现有 mask 无法支持稀疏 query，应先实现按连续 block span 分组的局部 forward。若两种方式都无法在不破坏因果依赖的情况下减少计算，则停止“加速迁移”，把该分支明确降级为位置敏感性分析；不得用 whole-window skip 或输出覆盖冒充逐 block 加速。

## 4. 目标代码与目录

计划新增：

```text
RollingForcing-main/
  utils/step_cache_policy.py
  utils/step_cache_state.py
  tests/test_step_cache_policy.py
  tests/test_step_cache_state.py
  tests/test_step_stage_mapping.py
  tests/test_step_cache_decision_hash.py
  configs/rolling_forcing_dmd_step_cache_disabled.yaml
  configs/rolling_forcing_dmd_step_cache_observe.yaml
  configs/rolling_forcing_dmd_step_cache_front_smoke.yaml
  scripts/summarize_step_cache.py
  scripts/replay_step_cache.py
  scripts/compare_videos_light.py
  scripts/run_step_cache_m0.sh
  scripts/run_step_cache_m1.sh
  scripts/run_step_cache_m2.sh
  repro/rollingforcing_step_cache/
    README.md
    capture_env.py
    setup_env.sh
    verify_env.py
    run_single_gpu_smoke.sh
    validate_smoke.py
    freeze_config.py
    build_bundle.py
    run_vbench_generation.sh
    run_vbench_eval.sh
```

计划修改：

```text
inference.py
pipeline/rolling_forcing_inference.py
utils/flowcache.py
utils/wan_wrapper.py
wan/modules/causal_model.py
configs/default_config.yaml
```

优先从 `E:\FlowCache-main\FlowCache4SkyReels-V2\...\step_cache_policy.py` 复用纯策略逻辑；模型状态、partial compute 和缓存生命周期必须按 RollingForcing 重做。

## 5. 配置设计

在现有 `flowcache` 下新增独立命名空间，默认完全关闭：

```yaml
flowcache:
  step_cache:
    enabled: false
    policy: none                 # none | observe_only | force_stage | force_window | dynamic_threshold
    schedule: null               # front_protect | u_shape_protect
    base_threshold: 0.0
    force_stage: null            # 0..4
    force_window: null           # early | middle | late
    warmup_local_stages: 1
    force_last_stage_recompute: true
    metric: patch_time_feature_l1rel
    metric_eps: 1.0e-8
    cache_scope: per_global_video_block
    compute_mode: layer_block_skip
    log_jsonl: true
    log_path: null
    debug_verify: false
```

兼容规则：

- `step_cache.enabled=false` 时不得构造持久状态、改变张量或打印 step-cache 日志。
- `policy=none` 表示统一 base threshold 的 Fixed 行为；`observe_only` 必须与 Fixed 使用同一决策并额外记录日志。
- `dynamic_threshold` 缺少 schedule、`force_stage` 越界、阈值为负、日志路径冲突时启动即失败。
- 阈值、warmup、stage 数和 feature 名均写入每次运行的 config hash。

## 6. 分阶段执行计划

### R0：基线冻结与结构审计

本地工作：

- 建立源码 manifest，记录当前关键文件 SHA256；服务器若有 Git，再同时记录 commit 和 dirty status。
- 固定 T2V、EMA、21-frame smoke、81-frame diagnostic、3 prompt 与 seed。
- 为每个 global block 输出 `window/stage/actual_timestep/noise_rank` 轨迹 CSV。
- 从现有 profiler提取主 DiT、clean-cache forward、VAE 和保存耗时。
- 做 partial-compute 接口 spike，验证 active query/block 的 shape、mask 和 KV 依赖。

服务器运行：单卡串行执行 upstream 21-frame 与 trajectory/profile 21-frame。

通过门槛：

- 原始视频可生成；
- 轨迹身份唯一，stage 方向无歧义；
- 主 DiT 与 clean-cache 的计时边界明确；
- partial compute 方案能在数值和因果依赖上成立，并能减少实际计算 token。

若最后一项失败，先停止后续加速实验，返回接入点诊断，不进入大规模迁移。

### R1：策略、状态与日志骨架

本地工作：

- 实现独立 policy/state 模块；
- 实现 per-block previous feature、accumulator、layer cache 和生命周期；
- 实现 `none/observe_only/force_stage/force_window/dynamic_threshold`；
- 实现逐决策 JSONL、run summary、decision-mask hash；
- 为 5-stage 分箱、warmup/cutoff、累计/重置、非法配置、状态清理和哈希写单测。

通过门槛：纯 CPU 单测全部通过；`py_compile` 通过；disabled path 不新增输出。

### R2：真实 block 级计算跳过

本地工作：

- 将决策点放到 patch/time feature 后、Transformer 重计算前；
- 接入 layer-wise block token cache 或通过 R0 验证的连续 span 方案；
- reuse 只影响主 denoise forward；clean-context forward 始终精确；
- 为实际 Q token、MLP token、缓存读取和重算 token 增加 profiler 计数；
- 每个 block 最终 stage 或离开生命周期后清理状态。

服务器运行：单卡 21-frame all-recompute、轻微 reuse 和高 reuse 三档 smoke。

通过门槛：

- 至少一个非边界 stage 出现真实 reuse；
- profiler 的实际计算 token 与 decision log 对齐；
- reuse 越多时 diffusion 计算量按预期下降；
- 无 stale state、shape drift、KV index drift、NaN、OOM、warning 或 fallback；
- disabled 与 all-recompute 不受 cache state 影响。

### R3 / M0：等价性与接口验收

同一张卡、同 prompt、同 seed、同 checkpoint 串行运行：

```text
A. upstream/default
B. hook-disabled
C. all-recompute（step cache 开启但所有阶段重算）
D. Fixed
E. observe-only
F. Front smoke
```

验收：

- A/B/C 的 latent 或解码帧相等；若底层非确定，预先声明误差容差并同时比较 PSNR。
- D/E 的视频、decision-mask hash、accumulator 更新完全相同。
- F 只因 active threshold 变化而改变 decision；其他配置 hash 字段一致。
- stage 0、stage 4 强制重算；不足完整窗口的头尾样本通过。
- 所有视频可解码，帧数、分辨率、FPS 正确。

### R4 / M1：逐 stage 位置敏感性

使用 3 prompt、1 seed、短视频，串行运行：

```text
Vanilla / no step cache
Fixed / observe-only
Force-stage-0
Force-stage-1
Force-stage-2
Force-stage-3
Force-stage-4
```

阈值可以使用 stress 档以制造足够事件，但必须单独标记，不进入正式候选。输出每 stage 的 reuse 数、加权计算量、diffusion latency、端到端 latency、PSNR、SSIM、LPIPS（不可用写 NA）和峰值显存。

通过门槛：被禁止 stage 的 reuse 为 0；所有实际决策与策略预期一致；能够在相同或可解释的计算预算下判断 early/middle/late 敏感性。只有 RollingForcing 数据支持早期更敏感时，Front Protect 才作为主方法继续。

### R5 / M2：轨迹回放与等预算验证

候选为 Fixed、Front、U-shape：

1. 采集 Fixed 完整 metric/decision 轨迹；
2. 离线回放阈值网格；
3. 只按 diffusion latency、实测/估算 PFLOPs 或 token 加权计算量选匹配点；
4. 不查看候选视频质量选阈值；
5. 生成候选后，验证每个候选的实测 mask 与自己的回放预测一致；
6. 再计算对 Vanilla 的全帧 PSNR/SSIM/LPIPS。

主比较点目标成本差异不超过 2%。受 5-stage 离散性限制无法严格匹配时，保留上下两个点并输出 Pareto 数据，不通过查看质量挑点。

### R6 / M3：slow/fast 筛选、重复计时与 KV 兼容

先在 step-cache 单独路径上完成：

- Fixed 扫描并确定 slow/fast 目标；
- Front/U-shape 扫描并按成本匹配；
- 3 prompt × 3 次重复计时；
- 所有 latency 运行单卡串行，不与其他生成任务并发；
- 冻结候选后再计算非 VBench 质量。

随后只对冻结候选做以下兼容矩阵：

```text
step cache off + KV compaction off
step cache off + ratio025_clean
step cache on  + KV compaction off
step cache on  + ratio025_clean
```

决策规则：

- 若 `ratio025_clean` 与 step cache 组合稳定且增益不被抵消，正式 Fixed/Front/U-shape 共享同一 KV 配置；
- 若存在明显交互、质量异常或收益抵消，正式位置策略比较关闭 KV compaction，并把 `ratio025_clean` 单列为已有基线/消融；
- 禁止 Fixed 与 Front 使用不同 KV 压缩设置。

冻结结果至少包含 Vanilla、Fixed-slow、Front-slow、Fixed-fast、Front-fast；U-shape 和 KV-only 是否进入正式方法表，由 M1/M2 证据和预先写明的规则决定。

### R7：干净机器单卡交付验收

在本地构建 immutable bundle，包含源码、冻结配置、环境锁、prompt 子集、安装/验证/运行脚本、manifest 和 SHA256，不包含 checkpoint、Wan 权重或 VBench 权重。

干净 Linux 机器只需要预装 Conda、CUDA driver，并挂载已有模型。最终用户入口固定为：

```bash
sha256sum -c rollingforcing-step-cache-pre-vbench-YYYYMMDD.tar.gz.sha256
tar -xzf rollingforcing-step-cache-pre-vbench-YYYYMMDD.tar.gz

export HANDOFF_ROOT=/data/rollingforcing-step-cache
export WORK_ROOT=/data/rollingforcing-step-cache-work
export RF_CHECKPOINT=/existing/checkpoints/rolling_forcing_dmd.pt
export WAN_MODEL=/existing/wan_models/Wan2.1-T2V-1.3B

bash "$HANDOFF_ROOT/repro/rollingforcing_step_cache/setup_env.sh" \
  "$HANDOFF_ROOT/environment" \
  "$WORK_ROOT/env"

CUDA_VISIBLE_DEVICES=0 bash \
  "$HANDOFF_ROOT/repro/rollingforcing_step_cache/run_single_gpu_smoke.sh" \
  --python "$WORK_ROOT/env/bin/python" \
  --source "$HANDOFF_ROOT/RollingForcing-main" \
  --checkpoint "$RF_CHECKPOINT" \
  --wan-model "$WAN_MODEL" \
  --output "$WORK_ROOT/smoke" \
  --all
```

`setup_env.sh` 必须新建环境、安装锁定依赖、运行 `pip check`、验证 CUDA/FlashAttention/核心 imports，并拒绝覆盖已有目标环境。模型路径验证只读，不自动下载或替换权重。

`run_single_gpu_smoke.sh --all` 串行执行：

```text
1. py_compile + unit tests
2. upstream/default 21-frame, 1 prompt
3. hook-disabled 21-frame, 同 prompt/seed
4. Fixed/observe-only 21-frame
5. Front 21-frame
6. 视频解码与元信息检查
7. decision identity/hash/状态清理检查
8. latency、显存、warning/fallback 汇总
```

最终 `smoke_report.json` 必须满足：

```text
status=ok
completed_cases=expected_cases
all_videos_decodable=true
upstream_disabled_equivalent=true
fixed_observe_equivalent=true
front_changed_position_decisions=true
real_compute_skip_observed=true
clean_cache_update_exact=true
warning_count=0
fallback_count=0
```

### R8：VBench 前冻结与 dry-run

本阶段只准备，不评分：

- 生成只读 `selected_configs.yaml/json`；
- 固定正式 prompt bundle 及其来源 commit/hash；
- 生成 formal generation plan，校验方法 × prompt × seed 的期望视频数、唯一输出名和命令；
- 生成 VBench scoring plan，校验维度覆盖、输入目录和期望 job 数；
- `run_vbench_generation.sh --plan-only` 与 `run_vbench_eval.sh --plan-only` 均通过；
- 配置冻结后拒绝原地覆盖，任何变化必须产生新版本和新 hash；
- 将单卡 smoke 报告纳入最终 locked bundle，再次校验 SHA256。

R8 通过即达到本项目的“VBench 前准备全部完成”。

## 7. 本地—服务器迭代方式

每轮遵循同一闭环：

1. 本地修改代码、补单测、生成本轮配置和唯一 `.sh` 入口。
2. 本地运行 CPU 单测、`py_compile`、配置/计划 dry-run 和 shell 静态检查。
3. 提供一条服务器启动命令，脚本内部负责串行顺序、日志、退出码和结果检查。
4. 用户在服务器执行，不手工拼接多段 Python 命令，不修改实验阈值。
5. 服务器回传轻量结果包；视频较大时只回传 smoke 视频、缩略图和完整清单，全量视频留服务器。
6. 本地核验 JSONL、hash、日志、指标和失败记录，再决定是否进入下一轮。

每轮回传包至少包含：

```text
command.txt
env.json
source_manifest.json
config.yaml
stdout.log
exit_code.txt
run.jsonl
runtime.json
summary.json / summary.csv
videos_ls_lh.txt
error_grep.txt
代表性 mp4 或抽帧图
```

失败运行不得删除；放入 `attempts/<timestamp>/`，并在 summary 中写明状态和原因。禁止两个会争抢同一 JSONL/输出目录的任务并行。

## 8. VBench 前最终检查表

- [ ] 关键源码版本、checkpoint hash、环境和命令已固定。
- [ ] 5-stage 实际 timestep/noise 方向已通过轨迹 CSV 核对。
- [ ] step cache 的命中发生在重型计算前，并有实际计算减少证据。
- [ ] disabled/all-recompute 与 upstream 等价。
- [ ] Fixed 与 observe-only 的视频及 decision mask 等价。
- [ ] clean-context、原生 KV、cross-attention 和 sink 语义保持精确。
- [ ] M1 逐 stage 诊断完成，位置结论来自 RollingForcing 自身数据。
- [ ] M2 Fixed/Front/U-shape 等预算比较完成，阈值选择未看质量。
- [ ] slow/fast 至少 3 prompt × 3 次单卡串行计时完成。
- [ ] PSNR/SSIM/LPIPS、延迟、FPS、显存、reuse 分布和失败记录齐全。
- [ ] `ratio025_clean` 与 step cache 的组合/分离结论明确。
- [ ] 正式方法、阈值、KV 设置和 prompt/seed 已冻结。
- [ ] 干净机器单卡安装及端到端 smoke 的 `smoke_report.json` 为 `status=ok`。
- [ ] VBench generation/eval 两个 plan-only 检查通过，尚未启动正式任务。
- [ ] locked bundle、manifest、SHA256、README、启动命令和回传清单齐全。

## 9. 推荐立即执行的顺序

1. 先完成 R0 的 stage 轨迹导出和 partial-compute spike；这是整个迁移是否能形成真实加速的首要门槛。
2. R0 通过后，移植纯策略模块并完成 R1 单测，不先写阈值扫描脚本。
3. 完成 R2 的真实 block skip 后，立即做单卡 M0 等价性；未通过前不进入 M1。
4. M1 先逐 stage 建立 RollingForcing 自身的位置敏感性证据，再决定 Front 是否为主策略。
5. 依次完成 M2、M3、KV 兼容矩阵、冻结和干净机器单卡交付验收。
6. 最后只做 VBench plan dry-run 与 bundle 锁定，把正式 VBench 留给下一阶段。
