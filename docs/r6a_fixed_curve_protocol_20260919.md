# R6A：126-latent RF+Fixed 成本曲线

R6A 是正式 VBench 前的质量盲成本校准第一步，只回答 RollingForcing 在官方发布
的 126-latent 生成规格下能达到哪些真实成本点。本轮不选择 slow/fast，不运行
Front/U-shape，不读取 VBench、latent MSE 或视频质量。

## 固定条件

- 单卡 GPU 7，串行运行；EMA checkpoint 与 Wan2.1-T2V-1.3B 资产不变。
- 126 latent、3 latent/block、42 个完整 block、seed 0。
- prompt 0 是独立 warmup；prompt 1--3 依次为既有 cat、rainy-car、robot 校准集。
- 每个候选独立进程，均先 warmup；统计后 3 条 prompt 的 diffusion CUDA event
  时间。diffusion 边界包括 rolling loop、main DiT、clean refresh、再加噪和
  必要缓存管理，排除文本编码、VAE 和写盘。
- 运行顺序为 Vanilla-start、五个 Fixed 阈值、Vanilla-end。前后 Vanilla 用于
  检测整轮机器漂移，不将二者伪装成更多独立 prompt。
- 每个视频必须解码为 501 帧、16 fps、832x480；解码帧 hash 写入报告后删除
  MP4，减少回传体积。
- 初始噪声及推理前后 CPU/CUDA RNG 状态只保存 SHA256。所有方法同一 prompt
  的这些 hash 必须完全一致。

## 扫描点

Fixed base threshold：`0.20, 0.26, 0.32, 0.40, 0.50`。

这些点只构成第一轮粗曲线：0.26/0.32 包围 R5B 的 0.2785 开发点，0.20 提供
保守端，0.40/0.50 探索更高复用端。它们不是正式 slow/fast 配置。收到服务器
曲线后，只依据 latency、实际算子成本、运行稳定性和曲线分离度提出 slow/fast
候选，再由用户确认。

## 验收

- CPU step-cache tests 全部通过；7 个进程全部正常结束。
- 每个进程恰有 4 个 diffusion 时间，首条只作 warmup。
- 所有 28 个视频通过501帧/16fps/832x480解码检查。
- 同 prompt 的 initial-noise 和 RNG hash 跨全部方法一致。
- Fixed 的 K/V token 恒等于完整窗口；`Q + reuse = full-window`。
- 每个进程 184 次 main forward、5520 次 layer call，结束后 residual/state 为0。
- 报告前后 Vanilla 的相对漂移及每点 CV；本轮不因 CV 较大自动重跑或自动选点。
