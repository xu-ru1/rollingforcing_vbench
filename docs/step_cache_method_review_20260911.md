# 方法与实验审核记录

日期：2026-09-11。审核对象：2026-09-05 迁移方案、交接文档和本地实际推理代码。
结论：迁移目标合理，但旧方案有若干会影响实现正确性、实验解释及验收真实性的问题，需采用v2协议。
本次仅修订方案，未改推理代码。未运行服务器实验，所有新方法收益仍未验证。

## 关键问题与修订

| 级别 | 旧方案问题 | 影响 | v2处理 |
|---|---|---|---|
| 高 | 缓存“hidden/K/V或输出”未确定 | 缓存年龄、timestep条件、误差来源与真正跳过的计算均不清楚 | 定义每层整体残差；当前K/V逐层重算，Fixed与动态策略用同一适配器 |
| 高 | 将窗口看成可独立运行的前序block序列 | 当前缓存推理对完整窗口K/V做默认非causal attention；裁剪K/V改变模型 | gather活动Q，保留原完整K/V及可见范围，不增加三角mask |
| 高 | 缩短Q后可能沿用query_length推导history | 实际读取更多历史，方法差异混入上下文差异 | 新接口分开full_window_tokens和active_query_tokens，RoPE按原坐标 |
| 高 | 只保护updating_cache=True | 主denoise也会写window第一个block的KV | 首block强制重算；两类forward的KV写入/eviction都审计 |
| 高 | 要求复用后clean KV与Vanilla“精确相同” | 当前输入轨迹已改变，这一要求不成立 | 保持更新算法/索引/时机，验证同输入快照下clean函数等价 |
| 高 | 内容metric却要求候选复现Fixed轨迹预测mask | 缓存会改变后续latent/特征，跨策略离线轨迹不是反事实真值 | 在线校准；离线只初选，候选实际轨迹自回放检查决策实现 |
| 高 | 保护首尾却要求Force-stage-0/4产生诊断事件 | stage0缺缓存，stage4被保护，两组结果没有剂量 | 主诊断只stage1/2/3；固定同一内部block集合、每block一次复用 |
| 高 | 固定seed但不检查RNG消耗 | scheduler每block生成整window噪声，裁剪采样将改变对照 | 保留随机调用形状/顺序，并验证初始噪声和RNG状态hash |
| 中 | 缓存decision身份与持久state key混用 | 含window/stage的key无法跨阶段复用 | 分开run event identity和跨stage state key；层级统计单列 |
| 中 | hash包含method/run_id等 | Fixed与observe逻辑相同也无法相等 | decision mask hash仅包含可配对身份和决策，协议hash另存 |
| 中 | 用已有本地改版充当官方upstream | 无法证明官方等价 | 标记B0-local并冻结源码；官方等价须独立commit证据 |
| 中 | 只要求全重算等价，未测混合mask | gather/scatter、非连续位置、近似状态处理可能有错 | 增加dense近似oracle与sparse同mask对照，不放入速度表 |
| 中 | 写21/81 frames未区分latent和解码 | 时长、样本成本和帧数验收混乱 | 写明21→81、81→321等预期，并实际解码验证 |
| 中 | 将3次同seed计时当稳健质量证据 | 质量样本仍只有3个，时间和样本方差混淆 | 拆重复计时、新seed、新prompt；质量按配对视频统计 |
| 中 | 2%时延匹配忽略GPU波动 | 噪声可能大于匹配差异 | 首选经验证FLOPs预算；时间需交错重复，报告不确定性 |
| 中 | 先质量筛选再称“未看质量选阈值” | 双重口径和选择偏差 | 校准仅按成本/有效性，配置冻结后揭盲质量，保留失败假设 |
| 中 | 后期可能把KV压缩加入主表 | 会改变已经完成的控制实验 | 主实验固定KV off；六格方法×KV独立消融 |
| 中 | 安装只承诺Conda+driver | FlashAttention源码构建可能缺nvcc/编译器；原路径有硬编码 | wheel或编译依赖清单、显式资产路径、全新环境实际安装 |
| 中 | VBench plan-only作为全部环境验收 | 计划无需实际模型/视频即可生成 | 增加评测依赖与权重最小前向，正式评分仍明确未测 |

## 对应的本地代码证据

以下行号为审核时快照；后续修改应结合函数名定位。

- `pipeline/rolling_forcing_inference.py`，`CausalInferencePipeline.__init__`：warp实际timestep；初始化30层、1560 tokens/frame。`inference_rolling_forcing`：窗口N+4、每window主forward和clean refresh。
- 同文件约410行：`scheduler.add_noise`中每block调用`torch.randn_like(denoised_pred.flatten(0,1))`，随机张量是整个window形状。
- 同文件约446行：clean refresh只处理当前窗口第一个block，启动窗口会重复refresh同一global block。
- `wan/modules/causal_model.py:1489`附近：current_start决定RoPE；`cache_end=cache_start+self.block_length`；主forward也写K/V、更新index和触发eviction。
- 同文件约2148行：`working_cache_max_length = self.max_attention_size - query_length - self.block_length`，改变query_length会影响history截取。
- `_flowcache_profiled_attention`调用`attention(query,key,value,...)`未显式开启causal；`wan/modules/attention.py:244`默认`causal=False`。这里只指缓存推理分支，不概括训练分支的mask。
- `CausalWanAttentionBlock.forward`约2950行：norm/modulation后self-attention，随后cross-attention和FFN；定义总层残差需包含全部gate和更新。
- `CausalWanModel._forward_inference`约3383行：patch/time embedding、整个窗口经过所有blocks、head；没有原生逐video block跨stage缓存。
- `utils/wan_wrapper.py:218`：当前KV分支仅一次条件模型调用，之后flow转换为x0；不能从config guidance字段推导CFG双分支。
- `inference.py:30`参数描述与实际latent用途不符；约243/257行采样latent，约303行以16fps保存视频；约302行prompt[:100]命名需要正式VBench适配。
- `utils/flowcache.py`的`record_output_reuse_dry_run`在完整主forward之后观察最终输出，属于诊断，不能证明实现了计算复用。

## 保留的有效部分

局部stage定义、Front/U-shape倍率、保护首尾、默认关闭、只在推理中改动、Fixed/observe等价、成本配对、日志与回传闭环、先单卡再交付，均保留。KV Round8负结果与小幅收益只作为工程经验，不用作新方法的实验数据。

## 尚需实测回答的问题

1. 第一层池化内容距离是否与层残差误差相关，五步下是否产生可用决策平台？
2. 保留全量当前K/V后，活动Q、cross-attention和MLP减少能否抵消状态读写与gather开销？
3. 残差复用的误差会否通过窗口内attention影响重算block或长期主体一致性？
4. BF16非连续Q数值、真实EMA checkpoint和单卡内存能否通过冻结分辨率验收？
5. Linux环境锁与评测权重是否齐全，能否在独立机器复建？

这些问题分别交由R1/R2、R3、R4–R6、R7处理；任何GPU验证均须以用户回传材料为依据。
