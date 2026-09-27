# RollingForcing step cache 分阶段工作包

日期：2026-09-11。执行协议：`rollingforcing_step_position_cache_migration_plan.md` v2。
状态：R0.2–R0.4 的本地观测、运行器和验收合同已完成；R0.1 源码冻结待随首个服务器 manifest 一并确认，R0.5 的不支持范围显式拒绝待 R1/R2 接口落实。R0–R8 的服务器验收均pending。

## 1. 职责与交付节奏

| 角色 | 工作 |
|---|---|
| 我（本地开发与审阅） | 改代码、写测试/配置/脚本、生成不可变小包与SHA256；检查每轮回传，修复并写结论 |
| 用户（服务器执行） | 提供已有模型/环境/空闲GPU路径；上传本轮包，按唯一命令运行单卡串行测试并回传 |
| 干净机器执行者（用户或交接接收者） | R7在全新目标环境按安装与启动脚本执行，不借用开发机site-packages或隐式软链 |

这里是职责分配，没有另行启动代理，也不假设我已获得服务器连接。每次只交付下一阶段需要的运行包，不要求用户手工改YAML和逐条拼命令。

## 2. 总体排程、依赖与预算

| 阶段 | 本地交付重点 | 服务器实验/视频数上限 | 依赖与通过后下一步 |
|---|---|---:|---|
| R0 基线与协议冻结 | 路径/源码/环境/阶段/RNG盘点 | 4 | 可运行B0、阶段轨迹明确后进R1 |
| R1 策略状态与特征观测 | policy/state/日志及CPU单测 | 0新增，优先复用R0观测；需新feature观测时另列3 | metric与状态合同锁定后进R2 |
| R2 部分计算原型 | 全K/V、部分Q、层残差、dense oracle | 3 | 算子确有减少、混合mask正确后进R3 |
| R3 M0完整验收 | 数值/边界/状态/视频检查器 | 19 | 兼容与稀疏实现通过后进R4 |
| R4 M1位置诊断 | 固定剂量stage注入与统计 | 15 | 诊断可信后进R5，不强求Front获胜 |
| R5 M2成本校准 | 有限在线扫描、预算匹配、冻结v1 | 102；需时延匹配时另加至多36 | 有实际不同预算点与可解释成本后进R6 |
| R6 冻结配置确认 | 重复计时/泛化/长视频/KV消融 | 150 | 能形成实用运行点、材料完整后进R7 |
| R7 清洁环境单卡验收 | 安装/验证/交付候选包 | 8，另有模型/算子自检 | 新环境实际推理通过后进R8 |
| R8 VBench前收尾 | 资产预检、计划、命名、locked bundle | 0正式VBench视频/评分 | 输出PRE_VBENCH_READY |

这是按六方法、两速度档的条件上限：主路线总计约301个非VBench测试视频；R5使用重复时延匹配时最高337，如R1须新观测则最高340。不是一次提交全部任务。若已有同协议hash的结果可复用，会扣除；失败、数值自检、预热和有证据必要的复跑另记，不伪装免费。方法数改变时自动重新生成数量。每轮启动前给出该轮预计GPU小时：用已测同长度单视频耗时乘计划数量，另列加载/预热，当前不臆造耗时承诺。

执行依赖为R0→R1→R2→R3→R4→R5→R6→R7→R8。环境锁的草稿可从R0准备；R7最终锁只能来自实际已验证版本。R0不再要求先完成R2的真实skip，避免阶段循环依赖。

## 3. R0：建立可追溯基线

本地任务：

- R0.1 保存迁移前只读B0-local源码快照及file hashes；若有官方commit另列来源。
- R0.2 编写只读preflight，收集Python/PyTorch/CUDA/backend、checkpoint EMA key/hash、Wan/T5/VAE/tokenizer文件、GPU与cgroup限额。
- R0.3 编写stage轨迹导出：复用真实scheduler warp，输出每block实际timestep和唯一匹配结果；增加RNG观测，不改变采样。
- R0.4 冻结prompt/seed/长度、数值门槛生成规则、日志与计时边界，写`protocol_v2.json`。
- R0.5 验证基础模型路径参数化方案；将I2V、多分支、非默认分辨率列为不支持范围并显式报错。

服务器：同一张空闲卡，21 latent单prompt B0重复2次；另21 latent轨迹观测1次、81 latent profile1次，共4视频。

回传：`reports/preflight_manifest.json`、`trace_21_trace.jsonl`、`trace_81_trace.jsonl`、两份`*_trace_summary.json`、`r0_assessment.json`、分项profile、三个视频检查JSON、4个视频和退出码。R0 的重复基线只做解码帧/摘要诊断；latent-level `numerics_gate.json` 留到 R3 的同输入 oracle。

验收：checkpoint实际加载；21/81解码长度正确；同block从stage0至4且无漏/重复；81的135次主决策、31次clean refresh核对；数值/资源基线清楚。缺模型或原版跑不通先修基础链路，不开始新缓存算法。

## 4. R1：策略、状态、metric合同

本地任务：

- R1.1 `utils/step_cache_policy.py`：fixed、observe、动态倍率、保护优先级、force_stage、诊断注入的独立入口。
- R1.2 `utils/step_cache_state.py`：per-sample/global-block状态、每层残差、生命周期、缓存年龄和内存统计。
- R1.3 metric固定池化网格、FP32相对L1；zero/NaN/shape异常明确定义；观测与重计算互不污染。
- R1.4 schema区分决策事件和层执行；协议hash与可配对mask hash分开。
- R1.5 CPU测试覆盖严格小于阈值、累计保留/归零、复用时更新metric但不刷新残差、首尾/首block保护、退出清理、多sample隔离、非法配置。

服务器：如R0材料未包含目标feature，只补3prompt、seed0、81 latent的只读观测，不开启真实复用。

交付：上述模块、`tests/test_step_cache_*.py`、schema、`metric_distribution.json`、测试记录。

验收：策略和state能脱离CUDA测试；同一距离轨迹Fixed与observe一致；disabled无缓存分配；没有复制其他模型专用阈值。若metric没有有效区分信息，只改metric并重新定版，尚不扫描大批阈值。

## 5. R2：真实部分计算原型

本地任务：

- R2.1 修改`wan/modules/causal_model.py`，拆分全量K/V与active Q；history按完整窗口长度，RoPE按原始坐标。
- R2.2 对active行执行attention输出、cross-attention和MLP，reuse行加上一stage最近真实重算层残差；head完整计算。
- R2.3 首block原生KV写入和clean bypass明确；wrapper/pipeline只传stage和决策上下文，不改变scheduler。
- R2.4 建dense参考实现，只用于同mask oracle；小张量覆盖current_only、anchor/history/current、头尾窗口、非连续active blocks及跨eviction。
- R2.5 统计实际K/Q/MLP tokens、缓存bytes、gather/scatter开销；FLOPs公式分别计入矩阵投影、QK/AV、MLP，不包含未建模模块时标明范围。

服务器：先GPU小张量/attention kernel对照，再21 latent单prompt全重算、少量注入、较多注入共3视频。注入是测试路径，不作为部署阈值。

交付：`partial_compute_validation.json`、`operator_counts.json`、`memory_report.json`、3视频及配置。

验收：same-mask sparse/dense在冻结容差内；确实只执行active行的目标算子；K/V可见集合不变；所有首block写操作仍执行。不得用自报reuse_count代替算子执行证据。

失败处理：索引/数值问题修复后只复测相关case；严重显存或kernel开销问题可形成缩小layer集合的独立适配器版本，所有方法一致并重过R2/M0。没有可用部分计算路径则记录技术未完成，不进入阈值扫描。

## 6. R3 / M0：完整正确性与单卡跑通

服务器视频矩阵：

| 子集 | 配置 | 数量 |
|---|---|---:|
| 21 latent，1prompt | B0两次、disabled、all-recompute、Fixed quiet、observe、Front、U-shape | 8 |
| 边界 | 3/6/12 latent各1次，检查不足窗口；无合法reuse也可正确通过 | 3 |
| 81 latent | Vanilla/Fixed/Front ×2prompt | 6 |
| sample reset | 同一进程连续2sample，核对空状态开始 | 2 |

本地检查器：分别核对B0-disabled、all-recompute、Fixed-observe、same-mask oracle；记录latent误差与解码帧结果，保护/清理/初始噪声/RNG和分支身份；观察full模式与summary模式开销区别。

验收：19个视频/预期case全部有效，真正命中出现在合法stage；无异常fallback或索引错误；81长于KV容量的过程实际覆盖eviction；metric、残差不被clean refresh覆盖。Front在某个短样本与Fixed mask相同本身不判失败；策略倍率单测和后续在线阈值证明其作用。

## 7. R4 / M1：位置敏感性

本地：生成`m1_injection_plan.json`，锁定block IDs 4..11和K=8，不按误差挑事件。注入stage1/2/3，每个video block只一次，其他stage重算。输出质量后再按stage聚合。

服务器：81 latent、3prompt、seed0；Vanilla、Fixed、Inject-stage1、2、3，共15视频。若质量工具缺LPIPS写NA；SSIM必须固定后端。

回传：所有decision trace、三组实际8次注入清单、算子成本、逐帧指标、每视频指标和匿名接触图/代表性MP4。

验收：同一block集合、事件数、缓存年龄；额外计算差异完整记录；报告paired PSNR/MSE/SSIM而非只挑最好样本。结论可以支持/反对/无法判断早期更敏感。结果不明确时继续R5只作为验证既定假设，不增加“挑赢”的stage规则。

## 8. R5 / M2：有限在线扫描与匹配

本地：编写`run_calibration.py`、`audit_decisions.py`、`select_matched.py`，所有阈值来源为R1观测；每方法5个初值、最多加至9个；Fine refinement共至多18次生成。主数据目录和quality目录分开，选择器不读取quality。

服务器：3prompt、seed0、81 latent，单卡串行交错。生成每个候选的真实轨迹和成本。候选自己的在线轨迹自回放必须精确复现其决策；不得拿Fixed离线轨迹强行验收Front。若FLOPs模型未验证，则对最多6方法×3prompt各追加2次入围计时（最多36次），用重复时延匹配；不得将单次测量写成稳定时延匹配。协议与预热口径相同的这些计时可在R6引用并扣除重复任务。

输出：`candidate_runs.csv`、`cost_model_validation.json`、`decision_audit.json`、`matched_selection.json`、`selected_configs_v1.json`及哈希。

门槛：Fixed/Front/U-shape使用同backend和保护机制；成本模型可解释；两个目标档实际不同且配对误差≤2%，或如实提供相邻点/仅一个档。选择文件记录候选总数、排除理由、预算偏差、尚未读取质量的选择依据。没有匹配点不伪造matched标签，按上限收束并报告。

## 9. R6 / M3：冻结后的确认与扩展

将R5冻结清单作为唯一输入。标准六方法的任务量：

| 工作 | 任务数 | 目的 |
|---|---:|---|
| 原3prompt×seed0/1/2×6方法 | 54 | 配对质量、新seed |
| seed0再重复2次×3prompt×6方法 | 36 | 每配置3次计时；不增加独立质量样本 |
| 6新prompt×seed1×6方法 | 36 | 未参与校准prompt的检查 |
| 243 latent×2prompt×Vanilla/Fixed-fast/Front-fast | 6 | 约60秒长序列容量、漂移 |
| 3prompt×{Vanilla/Fixed-fast/Front-fast}×{KV off/on} | 18 | 组合交互，单次不宣称稳健组合加速 |

统计任务：独立重算逐帧到逐视频均值；同时输出per-prompt结果、paired差值、计时均值/std、显存、cache bytes和复用分布。主表保留U-shape、Fixed和负结果，不改阈值；质量遇到明显失败先标记不适用，若改算法则新版本重新校准，旧结果仍保留。

验收：具备实用运行点（默认至少一个候选diffusion平均降低5%、e2e不倒退、容量通过）；该门槛不等于统计显著性。新策略未优于Fixed可验收工程并如实报告科学结果；完全没有加速点则迁移加速目标未完成，需要评审后续路线。

计时方差过大时最多补到每配置5次，并记录额外预算；仍不清楚则标记不确定，不不断重跑到赢。候选数/方法数变化由清单自动推导工作量。

## 10. R7：干净机器单卡验收

本地任务：

- `repro/rollingforcing_step_cache/setup_generation_env.sh`：独立新环境、锁定依赖与可用wheel/编译前置检查。
- `setup_eval_env.sh`：独立评测环境，不与生成环境混装。
- `verify_generation_env.py`：真实CUDA/attention算子、基础模型资产、EMA加载、解码器验证。
- `verify_eval_env.py`：六维所需已有权重、import、解码和最小前向，不评分。
- `run_single_gpu_smoke.sh`：统一case与退出码；`collect_results.py`收集回传。
- `build_bundle.py`：源码/配置/环境/prompt/license清单、文件hash和包hash，排除大权重与临时输出。

干净机器：不得激活旧开发环境；先bash安装，再同一GPU运行8个smoke视频：21 latent的Vanilla、Fixed-fast、Front-fast、U-shape、observe、all-recompute，以及81 latent的Vanilla、Front-fast。真正解码并检查decision及算子计数，至少重复启动一次验证不依赖上一进程缓存。

回传：`install_generation.log`、`install_eval.log`、两环境的lock/pip-check/版本与资产报告、`smoke_report.json`、8视频、完整命令和退出码。

验收：全新环境可按脚本复建；模型从指定本地资产加载；GPU可运行；8/8有效；Fixed/observe对应结果等价；真实skip与原生KV操作均符合协议。若只有新环境但不是独立干净机器，只记录`fresh_env_validated`，最终clean-machine一项仍pending。Windows CPU测试不能替代此门槛。

## 11. R8：只收尾，不开正式VBench

- 固定prompt bundle来源/hash和原文—增强文映射，名冲突/非法路径即失败。
- 生成完整方法×prompt×seed计划，video/job数自动推导，不直接复制990/36常数。
- `run_generation.sh --plan-only`验证命令/配置/模型资产；`run_eval.sh --plan-only`验证维度/元数据/预期目录，允许正式视频尚未存在并显式写`awaiting_generation`。
- `verify_eval_env.py`实际依赖/权重自检必须成功；正式评分端到端状态仍写`not_run`。
- 将R0–R7结论、内部manifest与smoke报告装入最终locked bundle，单独生成SHA256；不把未测过的新代码加入已验收包。
- 输出`pre_vbench_readiness.json`逐项状态和证据路径；所有必需项通过才置`PRE_VBENCH_READY`。

## 12. 统一启动与回传合同（待实现接口）

以下为未来脚本应实现的接口，不是当前可执行命令。每轮交付时会给出真实bundle路径、脚本及参数；当前不要求用户运行不存在的脚本。

```bash
bash repro/rollingforcing_step_cache/run_stage.sh \
  --stage R0 \
  --python /absolute/env/bin/python \
  --checkpoint /existing/rolling_forcing_dmd.pt \
  --wan-model /existing/Wan2.1-T2V-1.3B \
  --gpu 0 \
  --output /absolute/work/R0 \
  --execute
```

接口要求：默认只计划，执行需`--execute`；仅选择指定GPU，子进程内部统一逻辑device0，拒绝继承多卡distributed变量；所有生成串行。`set -euo pipefail`并正确传播子进程/tee退出码，失败停止后续阶段。每轮生成`stage_report.json`记录expected/completed/failed，而非根据文件存在就成功。

每run目录：`command.json`、`effective_config.json`、`runtime.json`、`decision.jsonl`（诊断）、`stdout.log`、`exit_code.txt`、视频与哈希。每stage附环境、源码manifest、统计、失败attempt和结果清单。续跑按sample/config/source哈希确认完成状态，不覆盖旧结果。

轻量回传包必须含所有文本和smoke代表视频；质量审核提供逐帧表及预登记抽帧图，完整大视频可留服务器。我审核后写`Rk_assessment.md`：通过项、失败项、允许进入哪一步、实际下一条运行命令。未回传的服务器结果保持pending，不自行打勾。

## 13. 第一批工作边界

先实施R0.1–R0.5和R1的纯策略/状态单测；服务器第一包只跑R0四个视频。取得GPU/模型路径、实际stage、重复数值和耗时后，再交R2原型。这样首批结果直接决定后续adapter可行性、内存上限和每轮运行预算。
