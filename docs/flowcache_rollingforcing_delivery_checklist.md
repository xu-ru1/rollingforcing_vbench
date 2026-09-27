# FlowCache / RollingForcing 阶段交付清单

## 主要代码文件

建议打包并核对这些代码文件：

- `inference.py`：入口参数、EvalMetrics、FlowCache/profiler 开关接入。
- `pipeline/rolling_forcing_inference.py`：RollingForcing inference loop、clean_cache_update、output reuse dry-run 与 profiling 调用点。
- `utils/flowcache.py`：FlowCache manager、JSONL telemetry、summary aggregation、warning/error 统计、Round 7/8 profiler 支持。
- `utils/wan_wrapper.py`：模型包装与调用路径适配。
- `wan/modules/causal_model.py`：CausalWan self-attention/KV cache path instrumentation 与诊断入口。
- `wan/modules/attention.py`：Round 8 attention path 相关局部支持，如当前提交包含该改动需打包。

注意：当前本地目录不是 git repository，无法用 `git diff` 自动给出权威改动清单，最终交付请以服务器代码包或提交记录再核对一次。

## 新增/更新文档

- `docs/flowcache_rollingforcing_stage_report.md`：主报告。
- `docs/flowcache_rollingforcing_experiment_summary.md`：组会汇报版摘要。
- `docs/flowcache_rollingforcing_stage_conclusion_round8_4.md`：截至 Round 8.4 的阶段结论。
- `docs/flowcache_rollingforcing_delivery_checklist.md`：本交付清单。
- `docs/round0_baseline_plan.md` 到 `docs/round8_0_attention_path_diagnosis.md`：各 Round 设计与实验说明。
- `docs/round8_fast_three_round_decision.md`：Round 8.1-8.3 三轮快速决策协议。

## 新增/关键配置

- `configs/default_config.yaml`：FlowCache 默认关闭配置。
- `configs/rolling_forcing_dmd_flowcache_round4*.yaml`
- `configs/rolling_forcing_dmd_flowcache_round5*.yaml`
- `configs/rolling_forcing_dmd_flowcache_round6_0_dryrun.yaml`
- `configs/rolling_forcing_dmd_flowcache_round7_0_profile.yaml`
- `configs/rolling_forcing_dmd_flowcache_round7_1_fine_profile.yaml`
- `configs/rolling_forcing_dmd_flowcache_round8_0_attention_profile.yaml`
- `configs/rolling_forcing_dmd_flowcache_round8_2_fast_compacted.yaml`

## 新增/关键脚本

- `scripts/summarize_flowcache_eval.py`
- `scripts/summarize_round5c.py`
- `scripts/summarize_round7_profile.py`
- `scripts/summarize_round7_1_profile.py`
- `scripts/summarize_round8_0_attention_profile.py`
- `scripts/run_round8_0_attention_profile.sh`：Round 8.0 服务器安全运行脚本，避免多行命令断开导致 `--output_folder: command not found`。
- `scripts/run_round8_fast_three_rounds.sh`：Round 8.1-8.3 三轮快速决策脚本。
- `scripts/summarize_round8_fast_three_rounds.py`：三轮快速决策结果汇总器。

## 关键 logs 与 JSONL

建议打包以下轻量文本结果：

- `logs/flowcache_stage_metrics_summary.csv`
- `logs/flowcache_stage_metrics_summary.json`
- `logs/flowcache_stage_report_notes.txt`
- `logs/round4.2-logs/round4_2_summary.csv`
- `logs/round5c-logs/round5c_summary.json`
- `logs/round5c-logs/round5c_summary.txt`
- `logs/round6-logs/round6_0_dryrun_21.jsonl`
- `logs/round6-logs/round6_0_81_dryrun.jsonl`
- `logs/round7-logs/round7_profile_report.json`
- `logs/round7-logs/round7_profile_report.txt`
- `logs/round7.1-logs/round7_1_81_3prompt_profile_summary.json`
- `logs/round7.1-logs/round7_1_profile_report.txt`
- `logs/round8-logs/round8_0_attention_report.txt`
- `logs/round8-logs/round8_0_evaluation.md`
- `scripts/run_round8_0_attention_profile.sh`

## 不建议直接打包的大文件

- 视频目录整体可能较大，建议只打包代表性视频或 `ls -lh` 输出。
- 全量 JSONL 若过大，可优先打包 summary JSON/TXT；Round 6/7/5C 的关键 JSONL 建议保留。
- checkpoint、模型权重、环境缓存不建议作为阶段报告附件打包。

## 仍缺失或需服务器补齐

- Round 8.1-8.3 三轮快速决策结果尚待服务器运行补齐。
- 部分早期 Round 0/1/2/3 smoke 缺少 EvalMetrics runtime/peak memory。
- 视频目录 `ls -lh` 输出未统一收集，尤其 Round 5/6/7/8。
- 若要严格比较 memory，需要服务器端一致环境下的 peak allocated/reserved 与显卡型号记录。
- 当前不是 git repository，建议服务器回传 `git status --short`、`git diff --stat` 或提交 hash。

## 建议交付包结构

- `docs/*.md`：所有 Round 文档加三份阶段交付文档。
- `configs/rolling_forcing_dmd_flowcache_*.yaml` 与 `configs/default_config.yaml`。
- `scripts/summarize_round*.py` 和 `scripts/summarize_flowcache_eval.py`。
- `logs/*summary*.json`、`logs/*summary*.txt`、关键 JSONL 和报告 CSV/JSON。
- 一份 `videos_ls_lh.txt`，记录代表性视频目录内容和大小。
