# R7：正式 VBench 前的冻结与 smoke

六组配置已按 R6D/R6E 的 2% diffusion-latency 协议冻结：Vanilla disabled，
Fixed-slow 0.26，Front-slow 0.24，Fixed-fast 0.40，Front-fast 0.54，
U-shape-fast 0.58。R7 不启动 946x6 VBench。

R7 对六组各生成 cat、rainy-car、robot 三个 126-latent、seed-0 样本，并验证：

1. immutable manifest 的有效配置、配置哈希、checkpoint 哈希、Git HEAD（若目录保存
   `.git`）或确定性 source-tree hash、生成规格及裁剪规则；
2. 全部原始视频为 501 decoded RGB frames、832x480、16 fps；
3. 每条 VBench 输入均是原始视频解码后前 `[0, 480)` 帧的无损 H.264 重封装，解码
   pixel SHA256 必须与原视频前 480 帧完全相同；
4. 主 denoise 与 `updating_cache=True` clean-cache forward 都由 forward hooks 和
   attention-call instrumentation 统计。FLOPs 使用每个 MAC 计两 FLOPs，包含实际
   Linear、Conv3d、QK/AV；不混入 VAE、视频编码或 scheduler，也不把 token 比例称作
   FLOPs；
5. 缓存组继续验证完整 K/V、Q+reuse 守恒和 terminal state 清空。

R7 只在 smoke 完成后返回 manifest、hash、FLOPs、crop report 和检查结果；视频文件
在检查后删除，防止调试包膨胀。通过后才准备正式 946x6 VBench 生成。
