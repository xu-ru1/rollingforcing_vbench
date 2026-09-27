# Round 0 Baseline Plan：RollingForcing -> FlowCache 适配前准备

本文档只做 baseline 与代码地图准备，不实现 FlowCache，不改变原始 inference 逻辑。

## 1. 代码地图

| 文件路径 | 类名/函数名 | 作用 | 与 FlowCache 适配的关系 | 是否建议下一轮修改 |
|---|---|---|---|---|
| `README.md` | Quick Start | 官方安装、checkpoint 下载、CLI inference、Gradio demo 命令 | baseline 跑法的主要来源；确认官方 config、checkpoint、prompt 路径 | 否 |
| `requirements.txt` | 无 | Python 依赖列表，包含 `torch==2.5.1`、`torchvision==0.20.1`、`diffusers==0.31.0`、`gradio` 等 | 服务器环境检查依据；FlashAttention 需按 README 另装 | 否 |
| `inference.py` | top-level script | CLI inference 入口；解析参数、加载 config、加载 checkpoint、构造 dataset、调用 pipeline、保存视频 | FlowCache 默认关闭时必须保持该入口行为不变；未来可只加可选配置读取，不改默认路径 | 可能，下一轮只做可选 flag 接入 |
| `inference.py` | `argparse` 参数区 | `--config_path`、`--checkpoint_path`、`--data_path`、`--extended_prompt_path`、`--output_folder`、`--num_output_frames`、`--i2v`、`--use_ema`、`--num_samples`、`--save_with_index` | `--num_output_frames` 是 smoke test 的最小长度控制；未来 FlowCache flag 可从 config 或 CLI 进入 | 可能，保持默认关闭 |
| `inference.py` | config 加载逻辑 | `OmegaConf.load(args.config_path)` 与 `configs/default_config.yaml` merge | 新 FlowCache 配置可进入 YAML，但默认值必须不改变当前行为 | 可能，下一轮小改 |
| `inference.py` | pipeline 选择 | 有 `denoising_step_list` 时进入 `CausalInferencePipeline`，否则进入 `CausalDiffusionInferencePipeline` | 当前官方 `rolling_forcing_dmd.yaml` 走 few-step rolling forcing 主路径 | 否 |
| `inference.py` | checkpoint 加载 | `torch.load(args.checkpoint_path)`；`--use_ema` 时读 `generator_ema`，否则读 `generator` | baseline 需准备 `checkpoints/rolling_forcing_dmd.pt`；FlowCache 不应改 checkpoint 格式 | 否 |
| `inference.py` | 输出保存 | `torchvision.io.write_video(output_path, video[seed_idx], fps=16)` | baseline 产物路径；FlowCache 适配时可对比同 prompt 输出和日志 | 否 |
| `app.py` | `_load_pipeline` / `build_predict` | Gradio 入口；固定使用 `CausalInferencePipeline`，默认 config 和 checkpoint | 可用于人工 demo，不建议作为 Round 0 smoke test 主路径 | 否 |
| `configs/default_config.yaml` | 默认配置 | 默认 `independent_first_frame=false`、`context_noise=0`、`num_frames=81` 等 | FlowCache 新配置应在默认关闭状态下与这些默认值兼容 | 可能，下一轮只加默认 false |
| `configs/rolling_forcing_dmd.yaml` | 官方 DMD inference/training config | 包含 `denoising_step_list: [1000,800,600,400,200]`、`warp_denoising_step: true`、`num_frame_per_block: 3`、`model_kwargs.timestep_shift: 5.0` | 当前 rolling window 长度由 denoising step 数决定；FlowCache chunk 可先对齐 `num_frame_per_block=3` | 可能，下一轮只加可选 FlowCache 配置 |
| `prompts/example_prompts.txt` | 文本 prompt 文件 | README 默认 CLI 示例使用的 prompt 列表 | baseline 可直接用；最小 smoke 建议另准备一行 prompt 文件避免跑多条 | 否 |
| `utils/dataset.py` | `TextDataset` | 按行读取 prompt；可选读取 extended prompt | baseline 输入格式依据；FlowCache 不需要改 dataset | 否 |
| `utils/dataset.py` | `TextImagePairDataset` | I2V 数据集读取 | I2V 路径暂不作为 Round 0 主线；FlowCache 先聚焦 T2V | 否 |
| `pipeline/__init__.py` | exports | 导出 `CausalInferencePipeline`、`CausalDiffusionInferencePipeline` 等 | 明确 inference 主类来源 | 否 |
| `pipeline/rolling_forcing_inference.py` | `CausalInferencePipeline.__init__` | 初始化 Wan diffusion wrapper、text encoder、VAE、scheduler、denoising step list、KV cache 句柄 | FlowCache inference-only 主接入层；可加 no-op manager，但默认必须关闭 | 是，下一轮小范围改 |
| `pipeline/rolling_forcing_inference.py` | `CausalInferencePipeline.inference_rolling_forcing` | 主 rolling forcing inference；构造 rolling windows、`noisy_cache`、`shared_timestep`，循环调用 generator | rolling window / chunk-wise reuse 的首选接入点；L1rel 判断应放在 generator 前后 | 是 |
| `pipeline/rolling_forcing_inference.py` | rolling window 构造 | `rolling_window_length_blocks = len(self.denoising_step_list)`；`window_start_blocks` / `window_end_blocks` | FlowCache chunk id、active window、clean chunk 边界都应从这里定义 | 是 |
| `pipeline/rolling_forcing_inference.py` | `noisy_cache` 更新 | 每个 window 之后对 block 调 `scheduler.add_noise(...)` 更新下一 timestep 的 noisy latent | output reuse residual 的保存/应用点；应在 latent update 后记录 residual | 是 |
| `pipeline/rolling_forcing_inference.py` | clean cache rerun | 对 `denoised_pred[:, :num_frame_per_block]` 以 `updating_cache=True` 回跑 generator | clean-chunk KV 生成点；第一版 FlowCache 应保持它精确执行 | 是 |
| `pipeline/rolling_forcing_inference.py` | `_initialize_kv_cache` | 为每层分配 `k/v/global_end_index/local_end_index`；当前 size 为 `1560 * 24` | history KV compression 需要在此基础上增加 tracker；默认不改 allocation | 是 |
| `pipeline/rolling_forcing_inference.py` | `_initialize_crossattn_cache` | 为每层分配文本 cross-attn KV cache | 不建议压缩；FlowCache 应保持 untouched | 否 |
| `pipeline/causal_diffusion_inference.py` | `CausalDiffusionInferencePipeline.inference` | 非 rolling few-step 的 causal diffusion 路径；有 temporal loop 和 scheduler timesteps loop | 不是官方 DMD config 的主路径；后续可作为兼容分支 | 暂不建议 |
| `pipeline/rolling_forcing_training.py` | `RollingForcingTrainingPipeline` | 训练/反向模拟中也有 rolling forcing 与 self forcing | 本任务明确 inference-only，不改训练逻辑 | 否 |
| `utils/wan_wrapper.py` | `WanTextEncoder` | 加载 T5 encoder 和 tokenizer，路径固定在 `wan_models/Wan2.1-T2V-1.3B` | baseline 必须准备 base Wan 模型目录；FlowCache 不应改 | 否 |
| `utils/wan_wrapper.py` | `WanVAEWrapper` | 加载 VAE，提供 `encode_to_latent` / `decode_to_pixel` | 输出 decode 路径；FlowCache 不应改 VAE | 否 |
| `utils/wan_wrapper.py` | `WanDiffusionWrapper` | 包装 `CausalWanModel`，建立 scheduler，将 flow prediction 转为 x0 | FlowCache 可在 wrapper 层做统一开关或 manager 传递 | 可能 |
| `utils/wan_wrapper.py` | `WanDiffusionWrapper.forward` | 将 `[B,F,C,H,W]` 转为 Wan 输入，传入 `kv_cache`、`crossattn_cache`、`current_start`、`updating_cache` | KV compression manager 可从这里透传到模型，但不建议 Round 0 实现 | 是，下一轮 |
| `wan/modules/causal_model.py` | `CausalWanModel.forward` | 有 `kv_cache` 时走 `_forward_inference`，否则走 `_forward_train` | inference KV 路径入口；默认路径必须保持不变 | 可能 |
| `wan/modules/causal_model.py` | `CausalWanModel._forward_inference` | patch embedding、time/text embedding、遍历 transformer blocks，将每层 cache 传入 block | KV tracker/manager 可随 kwargs 传入每层；下一轮小心加默认 None | 是 |
| `wan/modules/causal_model.py` | `CausalWanAttentionBlock.forward` | 一层 transformer block：self-attn、cross-attn、ffn | history KV compression 需要进入 self-attn 前后；cross-attn 不动 | 是 |
| `wan/modules/causal_model.py` | `CausalWanSelfAttention.forward` | 生成 Q/K/V，写入 KV cache，处理 eviction，抽取 anchor/working/current KV 并拼接 attention 输入 | attention sink、history KV、current window KV 的核心位置；KV compression 的核心接入点 | 是 |
| `wan/modules/causal_model.py` | `sink_tokens = 1 * self.block_length` | attention sink 目前硬编码为第一个 block | FlowCache 必须保护该区域不压缩；后续可配置但默认保持 1 block | 是 |
| `wan/modules/causal_model.py` | `working_cache_key` / `anchor_cache_key` / `input_key` | 从 cache 读取 history 与 sink，再与 current `roped_key` 拼接 | clean history compression 应只作用在 working clean 区域，不能压缩 sink/current KV | 是 |
| `wan/modules/model.py` | `WanT2VCrossAttention.forward` | 文本 cross-attn K/V cache 保存与复用 | 不建议做 FlowCache KV compression；保持默认 untouched | 否 |
| `wan/modules/attention.py` | `attention` / `flash_attention` | attention backend，优先 flash-attn，否则 SDPA | 不建议 Round 0 修改 kernel；FlowCache 在其上游改 KV 组织 | 否 |
| `wan/text2video.py` / `wan/image2video.py` | `WanT2V.generate` / `WanI2V.generate` | 原 Wan 采样实现，含传统 diffusion scheduler loop | 不是 RollingForcing CLI 主路径，但可作为参考 | 否 |
| `model/*.py` / `trainer/*.py` | DMD / SiD / GAN / CausVid / Trainer | 训练与 loss 逻辑 | 本任务不改训练；避免触碰 | 否 |

## 2. 最小 baseline inference 运行方案

### 2.1 重要限制

- 官方 CLI 的 T2V latent shape 在 `inference.py` 中固定为 `[num_samples, num_output_frames, 16, 60, 104]`，因此 Round 0 不建议尝试低分辨率。
- `num_frame_per_block` 在官方 config 中为 `3`，所以 `--num_output_frames` 应为 3 的倍数。
- `app.py` 中 UI 限制 `num_frames` 在 `21..252` 且为 3 的倍数；CLI 默认也是 `21`。最小 smoke test 建议使用 `--num_output_frames 21`。
- 官方 `rolling_forcing_dmd.yaml` 已经是 few-step：`denoising_step_list` 只有 5 个值。Round 0 不建议改 config 减 step。

### 2.2 需要提前准备的路径

不要假设服务器能联网下载大模型。请先检查是否已有：

```text
wan_models/Wan2.1-T2V-1.3B/
checkpoints/rolling_forcing_dmd.pt
configs/rolling_forcing_dmd.yaml
configs/default_config.yaml
```

`utils/wan_wrapper.py` 会读取以下 Wan base model 文件或目录：

```text
wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/
wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
wan_models/Wan2.1-T2V-1.3B/  # CausalWanModel.from_pretrained 的目录
```

如果 checkpoint 或 base model 不存在，需要用户确认已有共享路径，或手动下载/拷贝到上述位置。不要在 smoke 脚本里自动下载大文件。

### 2.3 环境检查命令

推荐先创建日志目录，再运行：

```bash
mkdir -p logs videos/baseline_smoke
bash scripts/print_env_check.sh | tee logs/env_check.log
```

也可以手动检查：

```bash
pwd
python --version
which python
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda)"
nvidia-smi
python -c "import flash_attn; print('flash_attn ok')"
ls configs
ls checkpoints
ls wan_models
```

### 2.4 最小 prompt 准备

为了只跑一条 prompt，建议在服务器工作目录外或 `logs/` 下准备一个一行 prompt 文件：

```bash
printf '%s\n' 'A calm lake at sunrise, cinematic, gentle camera movement.' > logs/smoke_prompt.txt
```

如果不想创建临时 prompt 文件，可以直接用 `prompts/example_prompts.txt`，但它会按文件中的多行 prompt 逐条生成，不是最小 smoke。

### 2.5 最小 baseline CLI 命令

推荐从仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path logs/smoke_prompt.txt \
  --output_folder videos/baseline_smoke \
  --num_output_frames 21 \
  --num_samples 1 \
  --use_ema \
  --save_with_index 2>&1 | tee logs/baseline_smoke.log
```

说明：

- `CUDA_VISIBLE_DEVICES=0` 只是示例。服务器上应先用 `nvidia-smi` 选空闲且合适的卡；小 smoke 不要占用 80G 大卡。
- `--num_output_frames 21` 是最小官方风格 smoke；不要在 Round 0 改源码降低分辨率。
- `--save_with_index` 避免 prompt 过长导致文件名不方便。
- 如果 checkpoint 里没有 `generator_ema`，去掉 `--use_ema` 后重试。

### 2.6 Gradio baseline

Gradio 不是 Round 0 首选，因为会启动服务并需要浏览器访问。但如需人工确认：

```bash
python app.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --output_dir videos/gradio
```

## 3. 服务器运行前检查清单

- 确认代码放在 `/data/<用户名>/` 下，避免在 `/mnt` 上写大量输出。
- 确认当前目录是 RollingForcing 仓库根目录。
- 确认 Python 环境已激活，且 `python --version` 约为 3.10。
- 确认 `torch`、`torchvision` 版本接近 `requirements.txt`。
- 确认 `torch.cuda.is_available()` 为 `True`。
- 确认 `flash_attn` 可以 import。
- 确认 `wan_models/Wan2.1-T2V-1.3B` 存在。
- 确认 `checkpoints/rolling_forcing_dmd.pt` 存在。
- 确认 `configs/rolling_forcing_dmd.yaml` 与 `configs/default_config.yaml` 存在。
- 确认有单行 smoke prompt 文件。
- 确认 `logs/` 和 `videos/baseline_smoke/` 已创建。
- 用 `nvidia-smi` 选择合适 GPU，不要占用别人正在使用的卡。
- 不要使用 sudo，不要改系统配置，不要删除或 kill 别人的任务。

## 4. 服务器运行后请回传的日志清单

请把以下内容复制回来用于下一轮分析：

```text
logs/env_check.log
logs/baseline_smoke.log
运行命令的完整文本
使用的 git commit hash 或源码包版本说明
nvidia-smi 运行前后的截图或文本
checkpoint 实际路径
Wan base model 实际路径
输出视频文件名和大小
如失败，完整 traceback
```

如果生成成功，也请记录：

```text
总耗时
峰值显存
使用 GPU 型号
是否使用 --use_ema
num_output_frames
prompt 文件内容
```

## 5. 下一轮 FlowCache 适配建议

Round 1 建议只做默认关闭的配置和 no-op manager：

- 在 config 中加入 `flowcache_enable: false`、`flowcache_output_reuse_enable: false`、`flowcache_kv_compress_enable: false`。
- 在 `CausalInferencePipeline.__init__` 中读取配置，但不改变默认路径。
- 增加一个轻量 `FlowCacheState` / `FlowCacheManager`，默认不启用。
- 加 disabled-mode smoke 对比，确保原始 inference 行为不变。

Round 2 再做 KV metadata tracker：

- 记录 `block_id -> KV local range`。
- 区分 `sink`、`active/current window`、`clean history`。
- 先只观察并打印，不压缩。

Round 3 再接 clean-chunk KV compression：

- 保护 attention sink 和 current KV。
- 只压缩已退出 rolling window 的 clean history KV。
- 默认关闭，打开后先跑短 smoke。

Round 4 再接 chunk-wise output reuse：

- 在 `inference_rolling_forcing` 中，`noisy_input/current_timestep` 构造后、generator 调用前接 L1rel 判断。
- residual 在 `scheduler.add_noise` 更新 `noisy_cache` 后记录。
- clean-cache refresh 初版保持精确执行，不跳过。
