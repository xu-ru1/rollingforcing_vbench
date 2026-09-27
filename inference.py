import argparse
import gc
import hashlib
import json
import torch
import os
import time
from contextlib import nullcontext
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
import imageio
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--reset_seed_per_prompt",
    action="store_true",
    help=(
        "Reset CPU and CUDA RNG to --seed before every prompt. Intended for "
        "controlled repeated timing; disabled by default to preserve normal generation semantics."
    ),
)
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--profile", action="store_true",
                    help="Print lightweight inference timing and CUDA memory metrics")
parser.add_argument("--eval_metrics", action="store_true",
                    help="Print one-line end-to-end evaluation runtime and CUDA memory metrics")
parser.add_argument("--allow_random_init", action="store_true",
                    help="Run without a generator checkpoint. Output will be random/noisy; use only for debugging.")
parser.add_argument("--save_step_cache_tensors_folder", type=str,
                    help="Optional folder for CPU initial-noise and final-latent tensors used by step-cache numerical checks.")
parser.add_argument(
    "--audit_hash_log",
    type=str,
    help=(
        "Optional JSONL output containing scalar metadata and SHA256 hashes for "
        "initial noise, final latent, and RNG states. No tensors are retained."
    ),
)
args = parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def require_existing_file(arg_name, path):
    if not path:
        parser.error(f"--{arg_name} is required")
    if not os.path.isfile(path):
        parser.error(f"--{arg_name} does not exist: {path}")


require_existing_file("config_path", args.config_path)
require_existing_file("data_path", args.data_path)
if args.extended_prompt_path:
    require_existing_file("extended_prompt_path", args.extended_prompt_path)
if args.checkpoint_path:
    require_existing_file("checkpoint_path", args.checkpoint_path)
elif not args.allow_random_init:
    parser.error(
        "--checkpoint_path is required for normal inference. "
        "Use --allow_random_init only when random/noisy debug output is intended."
    )

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    print(f"[Inference] Loading checkpoint: {args.checkpoint_path} (use_ema={args.use_ema})", flush=True)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict_key = 'generator_ema' if args.use_ema else 'generator'
    if state_dict_key not in state_dict:
        available_keys = ", ".join(map(str, state_dict.keys())) if hasattr(state_dict, "keys") else type(state_dict).__name__
        raise KeyError(
            f"Checkpoint {args.checkpoint_path} does not contain '{state_dict_key}'. "
            f"Available keys: {available_keys}"
        )
    if args.use_ema:
        state_dict_to_load = state_dict[state_dict_key]
        def remove_fsdp_prefix(state_dict):
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if "_fsdp_wrapped_module." in key:
                    new_key = key.replace("_fsdp_wrapped_module.", "")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        state_dict_to_load = remove_fsdp_prefix(state_dict_to_load)
    else:
        state_dict_to_load = state_dict[state_dict_key]
    pipeline.generator.load_state_dict(state_dict_to_load)
    # The checkpoint can contain both generator variants and remains referenced by
    # these module-level names for the lifetime of formal multi-prompt inference.
    # Release the CPU tensors as soon as the selected weights have been copied into
    # the generator.  This changes no model values or RNG state, but avoids keeping
    # one full checkpoint copy resident beside the GPU pipeline.
    del state_dict_to_load
    del state_dict
    gc.collect()
    print(f"[Inference] Loaded checkpoint weights: {state_dict_key}", flush=True)
else:
    print(
        "[Inference][warning] Running without --checkpoint_path because --allow_random_init was set; "
        "output will be random/noisy.",
        flush=True,
    )

pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
    if args.save_step_cache_tensors_folder:
        os.makedirs(args.save_step_cache_tensors_folder, exist_ok=True)
    if args.audit_hash_log:
        audit_parent = os.path.dirname(os.path.abspath(args.audit_hash_log))
        os.makedirs(audit_parent, exist_ok=True)
        with open(args.audit_hash_log, "w", encoding="utf-8"):
            pass

if dist.is_initialized():
    dist.barrier()

if args.eval_metrics:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    eval_metrics_start_time = time.time()
    eval_metrics_saved_videos = 0


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


flowcache_manager = getattr(pipeline, "flowcache_manager", None)
if flowcache_manager is not None:
    flowcache_manager.start_profiler_run(
        prompt_count=num_prompts,
        sample_count=args.num_samples,
        device=device,
    )
    flowcache_manager.start_attention_profiler_run(
        prompt_count=num_prompts,
        sample_count=args.num_samples,
        device=device,
    )
    round7_total_profile_token = flowcache_manager.start_profiler_phase(
        "total_inference",
        prompt_idx=None,
        sample_idx=None,
        count=num_prompts * args.num_samples,
        device=device,
    )
else:
    round7_total_profile_token = None
round7_profiler_saved_videos = 0

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()
    if args.reset_seed_per_prompt:
        # Repeated timing inputs must start from identical CPU/CUDA RNG states.
        # Keep the rank offset used by distributed initialization for consistency.
        set_seed(args.seed + local_rank)
    prompt_profile_token = (
        flowcache_manager.start_profiler_phase(
            "prompt_total",
            prompt_idx=idx,
            count=args.num_samples,
            device=device,
        )
        if flowcache_manager is not None else None
    )
    pipeline_overhead_token = (
        flowcache_manager.start_profiler_phase(
            "python_pipeline_overhead",
            prompt_idx=idx,
            count=1,
            device=device,
        )
        if flowcache_manager is not None else None
    )

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    audit_before = None
    if args.audit_hash_log:
        audit_before = {
            "event": "inference_audit_hash",
            "prompt_idx": int(idx),
            "seed": int(args.seed),
            "num_output_frames": int(args.num_output_frames),
            "initial_noise_shape": list(sampled_noise.shape),
            "initial_noise_dtype": str(sampled_noise.dtype),
            "initial_noise_sha256": tensor_sha256(sampled_noise),
            "cpu_rng_after_noise_sha256": tensor_sha256(torch.get_rng_state()),
            "cuda_rng_after_noise_sha256": (
                tensor_sha256(torch.cuda.get_rng_state(device))
                if torch.cuda.is_available() else None
            ),
        }

    if flowcache_manager is not None:
        flowcache_manager.end_profiler_phase(pipeline_overhead_token)

    if args.save_step_cache_tensors_folder:
        torch.save(
            sampled_noise.detach().to(device="cpu").contiguous(),
            os.path.join(args.save_step_cache_tensors_folder, f"{idx}_initial_noise.pt"),
        )

    # Generate 81 frames
    video, latents = pipeline.inference_rolling_forcing(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        profile=args.profile,
        prompt_idx=idx,
    )
    if audit_before is not None:
        audit_before.update({
            "final_latent_shape": list(latents.shape),
            "final_latent_dtype": str(latents.dtype),
            "final_latent_sha256": tensor_sha256(latents),
            "cpu_rng_after_inference_sha256": tensor_sha256(torch.get_rng_state()),
            "cuda_rng_after_inference_sha256": (
                tensor_sha256(torch.cuda.get_rng_state(device))
                if torch.cuda.is_available() else None
            ),
        })
        with open(args.audit_hash_log, "a", encoding="utf-8") as audit_handle:
            audit_handle.write(json.dumps(audit_before, sort_keys=True) + "\n")
    if args.save_step_cache_tensors_folder:
        torch.save(
            latents.detach().to(device="cpu").contiguous(),
            os.path.join(args.save_step_cache_tensors_folder, f"{idx}_latents.pt"),
        )
    with flowcache_manager.profile_phase(
        "python_pipeline_overhead",
        prompt_idx=idx,
        count=1,
        device=device,
    ) if flowcache_manager is not None else nullcontext():
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        num_generated_frames += latents.shape[1]

        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    with flowcache_manager.profile_phase(
        "video_save_total",
        prompt_idx=idx,
        count=args.num_samples,
        device=device,
    ) if flowcache_manager is not None else nullcontext():
        if idx < num_prompts:
            model = "regular" if not args.use_ema else "ema"
            for seed_idx in range(args.num_samples):
                # All processes save their videos
                if args.save_with_index:
                    output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
                else:
                    output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
                write_video(output_path, video[seed_idx], fps=16)
                round7_profiler_saved_videos += 1
                if flowcache_manager is not None:
                    flowcache_manager.record_saved_videos(1)
                if args.eval_metrics:
                    eval_metrics_saved_videos += 1
                # imageio.mimwrite(output_path, video[seed_idx], fps=16, quality=8, output_params=["-loglevel", "error"])

    if flowcache_manager is not None:
        flowcache_manager.end_profiler_phase(prompt_profile_token)

if args.eval_metrics:
    eval_metrics_profile_token = (
        flowcache_manager.start_profiler_phase(
            "eval_metrics_total",
            count=1,
            device=device,
        )
        if flowcache_manager is not None else None
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_sec = time.time() - eval_metrics_start_time
    peak_allocated_gb = None
    peak_reserved_gb = None
    if torch.cuda.is_available():
        scale = 1024 ** 3
        peak_allocated_gb = torch.cuda.max_memory_allocated() / scale
        peak_reserved_gb = torch.cuda.max_memory_reserved() / scale
    print(
        "[EvalMetrics] "
        f"runtime_sec={runtime_sec:.6f} "
        f"peak_cuda_allocated_gb={peak_allocated_gb if peak_allocated_gb is not None else 'none'} "
        f"peak_cuda_reserved_gb={peak_reserved_gb if peak_reserved_gb is not None else 'none'} "
        f"saved_videos={eval_metrics_saved_videos} "
        f"output_folder={args.output_folder}"
    )
    if flowcache_manager is not None:
        flowcache_manager.end_profiler_phase(eval_metrics_profile_token)

if flowcache_manager is not None:
    flowcache_manager.end_profiler_phase(
        round7_total_profile_token,
        count=num_prompts * args.num_samples,
    )
    flowcache_manager.summarize_profiler(
        prompt_count=num_prompts,
        sample_count=args.num_samples,
        saved_videos=round7_profiler_saved_videos,
    )
    flowcache_manager.summarize_attention_profiler(
        prompt_count=num_prompts,
        sample_count=args.num_samples,
    )
    
