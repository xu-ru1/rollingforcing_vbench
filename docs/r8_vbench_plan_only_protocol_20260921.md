# R8：正式 VBench 的 plan-only 预检

R8 不生成视频、不运行 VBench 指标、不读取任何质量分数，也不下载或加载 VBench
指标权重。它只冻结并验证正式执行所需的输入、命名和目录契约。

## 受控比较口径

- 使用 VBench 1.0 官方 `VBench_full_info.json` 的 946 条 metadata record 与全部 16 个维度；
- 六个方法均使用一个样本：seed `0`，且每条 prompt 前重置 RNG，保证 initial noise
  与完整随机数轨迹在方法间一一对应；
- 这是一项 cache-policy 的质量—效率受控比较，**不采用** VBench 推荐的多 seed 聚合
  或 temporal-flickering 的 25 个采样视频协议。报告必须明确这一边界；
- 每个样本为 126 latent、501 decoded RGB frames、16 fps、832x480；评测输入固定为
  原视频解码后前 `[0, 480)` 帧，不重采样、不缩放；
- latency、PFLOPs 与 token 统计均按完整 126-latent 推理，PFLOPs 包含
  `updating_cache=True` clean-cache forward。

## R8 输出

`pre_vbench_immutable_manifest.json` 记录 R7 源码树 SHA、关键源码 SHA、checkpoint
SHA、六个有效配置 SHA、VBench metadata SHA、prompt SHA、生成与裁剪规则。
R8 会将官方 metadata、prompt 文本和 index 映射复制到本次输出目录。当前冻结的 metadata
有 946 条 record、944 个唯一 prompt：两条 prompt 各被两个维度复用。R8 对每个唯一 prompt
只生成一个 seed-0 视频，VBench standard mode 按官方 metadata 将该同名视频用于相应的两个
record。因此实际输出量是 `944 x 6 = 5664`，不会生成文件名冲突且内容必然相同的两对视频。

生成阶段继续使用已经 R7 验证的 indexed 视频命名：`<index>-0_ema.mp4`。裁剪后，
`materialize_step_cache_vbench_standard.py` 会以硬链接或逐字节验证的复制方式创建
VBench 标准文件名 `<exact prompt>-0.mp4`。因此 VBench standard mode 使用官方 metadata
和辅助标签，不会退化为 custom-input 模式。

评测计划将 temporal-flickering 单独经过官方 `static_filter.py`，并记录保留下来的
视频数；其他 15 维使用同一批 480 帧标准命名视频。R8 的 `PLAN_READY` 只表示输入与
命令计划已冻结，VBench 指标权重与正式视频/评分仍为 `not_started`。离线 plan-only 可直接
使用随包附带的官方 metadata；若提供本地 VBench checkout，R8 还会导入该 checkout 并调用其
标准文件名解析器。
