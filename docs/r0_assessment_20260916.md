# R0 服务器验收记录（2026-09-16）

结论：**R0 通过，可进入 R1。** 本记录审核服务器目录 `outputs/step_cache_r0/r0_20260916T091948Z`。它只验证基线可观测性、环境和随机数合同；尚未实现或验证任何真实 step-cache 复用。

## 已验证的事实

- Python 3.10.20、PyTorch 2.5.1+cu124、CUDA 12.4；单卡为 NVIDIA A800 80GB PCIe。
- 17,028,919,541-byte checkpoint `rolling_forcing_dmd.pt` 已加载 `generator_ema`；checkpoint SHA256 为 `08448992460d85ef1b992dd30585d5724d098d805a48399730c8e717027a6d9d`。
- Wan 模型目录、T5 和 VAE 必需资产均存在。
- 21 latent 生成解码为 81 帧，81 latent 生成解码为 321 帧；分辨率 832×480、16 fps。
- 21 latent trace：7 个 video block，各 stage 0–4 各 7 次，共 35 个 stage event、11 个 window、33 个 RNG event；无漏项、重复项或主 forward 内 RNG 改变。
- 81 latent trace：27 个 video block，各 stage 0–4 各 27 次，共 135 个 stage event、31 个 window、93 个 RNG event；无漏项、重复项或主 forward 内 RNG 改变。
- 服务器的关键 R0 源码 SHA256 与本地工作区一致，包括 `pipeline/rolling_forcing_inference.py`、`utils/flowcache.py`、`utils/step_cache_r0.py`、R0 配置和运行器。

## 基线资源记录

81 latent trace 的 coarse profile：总端到端 75.01 s，rolling denoise loop 50.95 s，主 model forward（主 forward 与 clean refresh 的 inclusive 合计）43.82 s，clean refresh 9.35 s，VAE decode 13.90 s；峰值 allocated/reserved 分别为 24.73/31.53 GiB。该 profile 含 R0 CPU 同步与日志，不能作为后续 step-cache 性能结论或精确速度基线。

## 重复基线与限制

两次 21 latent B0 都有 81 个解码帧，但 decoded-frame SHA256 不同。进一步的逐像素检查显示 B0-A/B0-B 的平均绝对差为 2.549（uint8，99% 分位为 14）；B0-A/trace-21 为 2.546（99% 分位同为 14）。R0 trace 与自然重复波动相当，因此未发现插桩额外改变输出的证据。

这不是 latent-level 数值等价证明。R3 将采用相同输入/KV/state snapshot 下的 dense/sparse oracle，并在冻结容差后再作数值验收。R0 的 Git 字段失败是因为服务器源码目录不是 Git checkout，已由文件 SHA256 manifest 替代。

## R1 准入

允许开始纯 CPU 的策略、状态和 metric 合同实现。R1 的首批服务器任务仍保持 read-only：若需要实际 first-layer feature 分布，只新增观测，不启用 residual cache 或部分计算。
