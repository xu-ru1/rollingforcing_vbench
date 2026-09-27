from typing import List, Optional
import torch

from utils.flowcache import FlowCacheManager
from utils.step_cache_r0 import (
    resolve_stage_index,
    validate_block_timestep_values,
)
from utils.step_cache_metric import MetricBlockContext, StepCacheMetricRecorder
from utils.step_cache_flops import DiTFLOPRecorder
from utils.step_cache_runtime import StepCacheRuntime
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def _flowcache_timestep_summary(value) -> Optional[str]:
    try:
        if value is None:
            return None
        if hasattr(value, "detach"):
            flat = value.detach().flatten()
            unique_values = torch.unique(flat).float().cpu().tolist()
        elif isinstance(value, (list, tuple)):
            unique_values = list(dict.fromkeys(value))
        else:
            unique_values = [value]
        parts = []
        for item in unique_values[:8]:
            if isinstance(item, float):
                parts.append(f"{item:.6g}")
            else:
                parts.append(str(item))
        if len(unique_values) > 8:
            parts.append("...")
        return ",".join(parts)
    except Exception:
        return None


def _record_r0_rng_state(manager, *, prompt_idx, window_index, phase, device):
    if not getattr(manager, "should_record_r0_trace", False):
        return
    if not getattr(manager.config, "r0_trace_rng_state", False):
        return
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    manager.record_r0_rng_state(
        prompt_idx=prompt_idx,
        window_index=window_index,
        phase=phase,
        cpu_rng_state=cpu_state,
        cuda_rng_state=cuda_state,
        cuda_device=device if torch.cuda.is_available() else None,
    )


def _record_r0_stage_trace(
    manager,
    *,
    prompt_idx,
    window_index,
    start_block,
    end_block,
    current_timestep,
    denoising_step_list,
    num_frame_per_block,
    current_num_frames,
):
    if not getattr(manager, "should_record_r0_trace", False):
        return
    tolerance = float(manager.config.r0_stage_match_tolerance)
    stage_values = [
        float(item) for item in denoising_step_list.detach().float().cpu().tolist()
    ]
    for global_block_id in range(start_block, end_block + 1):
        local_block_index = global_block_id - start_block
        local_start = local_block_index * num_frame_per_block
        local_end = (local_block_index + 1) * num_frame_per_block
        timestep_values = (
            current_timestep[:, local_start:local_end]
            .detach()
            .float()
            .cpu()
            .flatten()
            .tolist()
        )
        actual_timestep = validate_block_timestep_values(timestep_values, tolerance)
        local_stage_index = resolve_stage_index(
            actual_timestep, stage_values, tolerance)
        manager.record_r0_stage_trace(
            prompt_idx=prompt_idx,
            window_index=window_index,
            start_block=start_block,
            end_block=end_block,
            global_video_block_id=global_block_id,
            local_block_index=local_block_index,
            actual_timestep=actual_timestep,
            local_stage_index=local_stage_index,
            total_local_stages=len(stage_values),
            num_frame_per_block=num_frame_per_block,
            current_num_frames=current_num_frames,
            stage_values=stage_values,
        )


def _build_step_cache_metric_contexts(
    *, prompt_idx, window_index, start_block, end_block, current_timestep,
    denoising_step_list, num_frame_per_block,
):
    tolerance = 1.0e-4
    stage_values = [float(item) for item in denoising_step_list.detach().float().cpu().tolist()]
    contexts = []
    for global_block_id in range(start_block, end_block + 1):
        local_block_index = global_block_id - start_block
        local_start = local_block_index * num_frame_per_block
        local_end = (local_block_index + 1) * num_frame_per_block
        actual_timestep = validate_block_timestep_values(
            current_timestep[:, local_start:local_end].detach().float().cpu().flatten().tolist(),
            tolerance,
        )
        contexts.append(MetricBlockContext(
            sample_id=0 if prompt_idx is None else int(prompt_idx),
            window_index=window_index,
            global_video_block_id=global_block_id,
            local_block_index=local_block_index,
            local_stage_index=resolve_stage_index(actual_timestep, stage_values, tolerance),
            actual_timestep=actual_timestep,
            branch="conditional",
        ))
    return contexts


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.max_attention_tokens = getattr(self.generator, "seq_len", 21 * self.frame_seq_length)

        self.kv_cache_clean = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")
        self.flowcache_manager = FlowCacheManager.from_args(args)
        self.step_cache_metric_recorder = StepCacheMetricRecorder.from_args(args)
        self.step_cache_runtime = StepCacheRuntime(args)
        self.step_cache_flop_recorder = DiTFLOPRecorder.from_args(args)
        self.step_cache_flop_recorder.attach(self.generator.model)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference_rolling_forcing(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        prompt_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        step_cache_sample_id = 0 if prompt_idx is None else int(prompt_idx)
        self.step_cache_flop_recorder.begin_sample(step_cache_sample_id)
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        with self.flowcache_manager.profile_phase(
            "text_encode_total",
            prompt_idx=prompt_idx,
            count=batch_size,
            device=noise.device,
        ):
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        with self.flowcache_manager.profile_phase(
            "kv_cache_init_total",
            prompt_idx=prompt_idx,
            count=batch_size,
            device=noise.device,
        ):
            if self.kv_cache_clean is None:
                self._initialize_kv_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device
                )
                self._initialize_crossattn_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device
                )
            else:
                # reset cross attn cache
                for block_index in range(self.num_transformer_blocks):
                    self.crossattn_cache[block_index]["is_init"] = False
                # reset kv cache
                for block_index in range(len(self.kv_cache_clean)):
                    self.kv_cache_clean[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_clean[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # implementing rolling forcing 
        # construct the rolling forcing windows
        num_denoising_steps = len(self.denoising_step_list)
        rolling_window_length_blocks = num_denoising_steps
        window_start_blocks = []
        window_end_blocks = []
        window_num = num_blocks + rolling_window_length_blocks - 1

        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)

        # init noisy cache
        noisy_cache = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        output_reuse_last_step_idx = [None] * num_blocks
        output_reuse_last_timestep = [None] * num_blocks

        # init denosing timestep, same accross windows
        shared_timestep = torch.ones(
            [batch_size, rolling_window_length_blocks * self.num_frame_per_block],
            device=noise.device,
            dtype=torch.float32)
        
        for index, current_timestep in enumerate(reversed(self.denoising_step_list)): # from clean to noisy 
            shared_timestep[:, index * self.num_frame_per_block:(index + 1) * self.num_frame_per_block] *= current_timestep


        # Denoising loop with rolling forcing
        denoise_loop_profile_token = self.flowcache_manager.start_profiler_phase(
            "denoise_loop_total",
            prompt_idx=prompt_idx,
            count=window_num,
            device=noise.device,
        )
        for window_index in range(window_num):

            if profile:
                block_start.record()

            print('window_index:', window_index)
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index] # include
            print(f"start_block: {start_block}, end_block: {end_block}")

            current_start_frame = start_block * self.num_frame_per_block
            current_end_frame = (end_block + 1) * self.num_frame_per_block # not include
            current_num_frames = current_end_frame - current_start_frame

            # noisy_input: new noise and previous denoised noisy frames, only last block is pure noise
            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block or current_start_frame == 0:
                noisy_input = torch.cat([
                    noisy_cache[:, current_start_frame : current_end_frame - self.num_frame_per_block],
                    noise[:, current_end_frame - self.num_frame_per_block : current_end_frame ]
                ], dim=1)
            else: # at the end of the video
                noisy_input = noisy_cache[:, current_start_frame:current_end_frame]

            # init denosing timestep
            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block:
                current_timestep = shared_timestep
            elif current_start_frame == 0:
                current_timestep = shared_timestep[:,-current_num_frames:]
            elif current_end_frame == num_frames:
                current_timestep = shared_timestep[:,:current_num_frames]
            else:
                raise ValueError("current_num_frames should be equal to rolling_window_length_blocks * self.num_frame_per_block, or the first or last window.")

            self.flowcache_manager.begin_window(
                window_index=window_index,
                start_block=start_block,
                end_block=end_block,
                rolling_window_length_blocks=rolling_window_length_blocks,
                num_frame_per_block=self.num_frame_per_block,
                denoising_step_list=self.denoising_step_list,
                noisy_cache_shape=noisy_cache.shape,
                noisy_input_shape=noisy_input.shape,
                current_timestep_shape=current_timestep.shape,
                current_start_frame=current_start_frame,
                current_end_frame=current_end_frame,
                current_num_frames=current_num_frames,
            )
            _record_r0_stage_trace(
                self.flowcache_manager,
                prompt_idx=prompt_idx,
                window_index=window_index,
                start_block=start_block,
                end_block=end_block,
                current_timestep=current_timestep,
                denoising_step_list=self.denoising_step_list,
                num_frame_per_block=self.num_frame_per_block,
                current_num_frames=current_num_frames,
            )
            _record_r0_rng_state(
                self.flowcache_manager,
                prompt_idx=prompt_idx,
                window_index=window_index,
                phase="before_main_forward",
                device=noise.device,
            )
            self.flowcache_manager.log_kv_range(
                self.kv_cache_clean,
                event="before_denoise",
                window_index=window_index,
                start_block=start_block,
                end_block=end_block,
                rolling_window_length_blocks=rolling_window_length_blocks,
                num_frame_per_block=self.num_frame_per_block,
                frame_seq_length=self.frame_seq_length,
                current_num_frames=current_num_frames,
                max_attention_tokens=self.max_attention_tokens,
            )

            step_cache_contexts = None
            if self.step_cache_metric_recorder.enabled or self.step_cache_runtime.enabled:
                step_cache_contexts = _build_step_cache_metric_contexts(
                    prompt_idx=prompt_idx,
                    window_index=window_index,
                    start_block=start_block,
                    end_block=end_block,
                    current_timestep=current_timestep,
                    denoising_step_list=self.denoising_step_list,
                    num_frame_per_block=self.num_frame_per_block,
                )
            if self.step_cache_metric_recorder.enabled:
                self.step_cache_metric_recorder.begin_main_forward(
                    step_cache_contexts,
                    num_frame_per_block=self.num_frame_per_block,
                )
            if self.step_cache_runtime.enabled:
                self.step_cache_runtime.begin_main_forward(
                    step_cache_contexts,
                    num_layers=len(self.generator.model.blocks),
                    tokens_per_block=self.num_frame_per_block * self.frame_seq_length,
                    num_frame_per_block=self.num_frame_per_block,
                )

            # calling DiT
            with self.flowcache_manager.profile_phase(
                "model_forward_total",
                prompt_idx=prompt_idx,
                window_index=window_index,
                device=noise.device,
            ):
                attention_context = None
                if getattr(
                    self.flowcache_manager,
                    "attention_profiler_enabled",
                    False,
                ):
                    attention_context = (
                        self.flowcache_manager.set_attention_profiler_context(
                            prompt_idx=prompt_idx,
                            window_idx=window_index,
                            timestep=_flowcache_timestep_summary(
                                current_timestep),
                            branch="denoise_attention",
                        )
                    )
                try:
                    self.step_cache_flop_recorder.begin_forward("main_denoise")
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=current_timestep,
                        kv_cache=self.kv_cache_clean,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        flowcache_manager=self.flowcache_manager,
                        flowcache_window_index=window_index,
                        flowcache_event="denoise_attention",
                        step_cache_metric_recorder=self.step_cache_metric_recorder,
                        step_cache_runtime=self.step_cache_runtime,
                    )
                finally:
                    self.step_cache_flop_recorder.end_forward()
                    if self.step_cache_metric_recorder.enabled:
                        self.step_cache_metric_recorder.end_main_forward()
                    if self.step_cache_runtime.enabled:
                        self.step_cache_runtime.end_main_forward()
                    if attention_context is not None:
                        self.flowcache_manager.restore_attention_profiler_context(
                            attention_context)
            _record_r0_rng_state(
                self.flowcache_manager,
                prompt_idx=prompt_idx,
                window_index=window_index,
                phase="after_main_forward",
                device=noise.device,
            )
            self.flowcache_manager.log_kv_range(
                self.kv_cache_clean,
                event="after_denoise",
                window_index=window_index,
                start_block=start_block,
                end_block=end_block,
                rolling_window_length_blocks=rolling_window_length_blocks,
                num_frame_per_block=self.num_frame_per_block,
                frame_seq_length=self.frame_seq_length,
                current_num_frames=current_num_frames,
                max_attention_tokens=self.max_attention_tokens,
            )

            if self.flowcache_manager.should_run_output_reuse_dry_run:
                with torch.no_grad():
                    for block_idx in range(start_block, end_block + 1):
                        chunk_idx = block_idx - start_block
                        local_start = chunk_idx * self.num_frame_per_block
                        local_end = (chunk_idx + 1) * self.num_frame_per_block
                        global_start = block_idx * self.num_frame_per_block
                        global_end = (block_idx + 1) * self.num_frame_per_block
                        block_time_step = current_timestep[
                            :,
                            local_start:local_end,
                        ].mean().item()
                        matches = torch.abs(self.denoising_step_list - block_time_step) < 1e-4
                        match_indices = torch.nonzero(matches, as_tuple=True)[0]
                        block_step_idx = (
                            int(match_indices[0].item())
                            if len(match_indices) > 0 else None
                        )

                        self.flowcache_manager.record_output_reuse_dry_run(
                            previous_tensor=output[:, global_start:global_end],
                            current_tensor=denoised_pred[:, local_start:local_end],
                            prompt_idx=prompt_idx,
                            window_index=window_index,
                            step_idx=block_step_idx,
                            timestep=block_time_step,
                            previous_step_idx=output_reuse_last_step_idx[block_idx],
                            previous_timestep=output_reuse_last_timestep[block_idx],
                            block_idx=block_idx,
                            chunk_idx=chunk_idx,
                            start_block=start_block,
                            end_block=end_block,
                        )
                        output_reuse_last_step_idx[block_idx] = block_step_idx
                        output_reuse_last_timestep[block_idx] = block_time_step

            output[:, current_start_frame:current_end_frame] = denoised_pred
                

            # update noisy_cache, which is detached from the computation graph
            with self.flowcache_manager.profile_phase(
                "noisy_cache_update_total",
                prompt_idx=prompt_idx,
                window_index=window_index,
                device=noise.device,
            ):
                with torch.no_grad():
                    for block_idx in range(start_block, end_block + 1):
                        
                        block_time_step = current_timestep[:, 
                                        (block_idx - start_block)*self.num_frame_per_block : 
                                        (block_idx - start_block+1)*self.num_frame_per_block].mean().item()
                        matches = torch.abs(self.denoising_step_list - block_time_step) < 1e-4
                        block_timestep_index = torch.nonzero(matches, as_tuple=True)[0]

                        if block_timestep_index == len(self.denoising_step_list) - 1:
                            continue

                        next_timestep = self.denoising_step_list[block_timestep_index + 1].to(noise.device)

                        noisy_cache[:, block_idx * self.num_frame_per_block:
                                        (block_idx+1) * self.num_frame_per_block] = \
                            self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])[:, (block_idx - start_block)*self.num_frame_per_block:
                                                                        (block_idx - start_block+1)*self.num_frame_per_block]

            _record_r0_rng_state(
                self.flowcache_manager,
                prompt_idx=prompt_idx,
                window_index=window_index,
                phase="after_noisy_cache_update",
                device=noise.device,
            )


            # rerun with timestep zero to update the clean cache, which is also detached from the computation graph
            with self.flowcache_manager.profile_phase(
                "clean_cache_update_total",
                prompt_idx=prompt_idx,
                window_index=window_index,
                device=noise.device,
            ):
                with torch.no_grad():
                    self.flowcache_manager.log_event(
                        event="before_clean_cache_update",
                        window_index=window_index,
                        details={
                            "start_block": start_block,
                            "end_block": end_block,
                        },
                    )
                    context_timestep = torch.ones_like(current_timestep) * self.args.context_noise
                    # # add context noise
                    # denoised_pred = self.scheduler.add_noise(
                    #     denoised_pred.flatten(0, 1),
                    #     torch.randn_like(denoised_pred.flatten(0, 1)),
                    #     context_timestep * torch.ones(
                    #         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    # ).unflatten(0, denoised_pred.shape[:2])

                    # only cache the first block
                    denoised_pred = denoised_pred[:,:self.num_frame_per_block]
                    context_timestep = context_timestep[:,:self.num_frame_per_block]
                    with self.flowcache_manager.profile_phase(
                        "model_forward_total",
                        prompt_idx=prompt_idx,
                        window_index=window_index,
                        device=noise.device,
                    ):
                        attention_context = None
                        if getattr(
                            self.flowcache_manager,
                            "attention_profiler_enabled",
                            False,
                        ):
                            attention_context = (
                                self.flowcache_manager
                                .set_attention_profiler_context(
                                    prompt_idx=prompt_idx,
                                    window_idx=window_index,
                                    timestep=_flowcache_timestep_summary(
                                        context_timestep),
                                    branch="clean_cache_update",
                                )
                            )
                        try:
                            self.step_cache_flop_recorder.begin_forward("clean_cache_update")
                            self.generator(
                                noisy_image_or_video=denoised_pred,
                                conditional_dict=conditional_dict,
                                timestep=context_timestep,
                                kv_cache=self.kv_cache_clean,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                updating_cache=True,
                                flowcache_manager=self.flowcache_manager,
                                flowcache_window_index=window_index,
                                flowcache_event="clean_cache_update_attention",
                            )
                        finally:
                            self.step_cache_flop_recorder.end_forward()
                            if attention_context is not None:
                                self.flowcache_manager.restore_attention_profiler_context(
                                    attention_context)
                    self.flowcache_manager.log_kv_range(
                        self.kv_cache_clean,
                        event="after_clean_cache_update",
                        window_index=window_index,
                        start_block=start_block,
                        end_block=end_block,
                        rolling_window_length_blocks=rolling_window_length_blocks,
                        num_frame_per_block=self.num_frame_per_block,
                        frame_seq_length=self.frame_seq_length,
                        current_num_frames=self.num_frame_per_block,
                        max_attention_tokens=self.max_attention_tokens,
                    )
                    self.flowcache_manager.log_event(
                        event="after_clean_cache_update",
                        window_index=window_index,
                        details={
                            "start_block": start_block,
                            "end_block": end_block,
                        },
                    )

            if self.step_cache_runtime.enabled:
                self.step_cache_runtime.finish_window_after_clean()
            self.flowcache_manager.end_window(window_index)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)


        self.flowcache_manager.end_profiler_phase(
            denoise_loop_profile_token,
            count=window_num,
        )

        with self.flowcache_manager.profile_phase(
            "flowcache_logging_total",
            prompt_idx=prompt_idx,
            device=noise.device,
        ):
            self.flowcache_manager.summarize()
            self.flowcache_manager.summarize_candidates()
            self.flowcache_manager.summarize_compression_dry_run()
            self.flowcache_manager.summarize_real_compression()
            self.flowcache_manager.summarize_cache_body_compression()
            self.flowcache_manager.summarize_persistent_cache_compression()
            self.flowcache_manager.summarize_compacted_kv()
            self.flowcache_manager.summarize_output_reuse_dry_run()

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        with self.flowcache_manager.profile_phase(
            "vae_decode_total",
            prompt_idx=prompt_idx,
            count=batch_size,
            device=noise.device,
        ):
            video = self.vae.decode_to_pixel(output, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")
            if torch.cuda.is_available():
                scale = 1024 ** 3
                peak_allocated = torch.cuda.max_memory_allocated() / scale
                peak_reserved = torch.cuda.max_memory_reserved() / scale
                print(f"  - Peak CUDA allocated: {peak_allocated:.6f} GB")
                print(f"  - Peak CUDA reserved: {peak_reserved:.6f} GB")

        # The observer is read-only, but its previous-feature table has the
        # same sample lifetime as runtime metric/residual state.  Clear both
        # here so consecutive prompts served by one pipeline instance cannot
        # inherit a previous sample's observer feature.
        if self.step_cache_metric_recorder.enabled:
            self.step_cache_metric_recorder.clear_sample(step_cache_sample_id)
        if self.step_cache_runtime.enabled:
            self.step_cache_runtime.write_summary()
            self.step_cache_runtime.clear_sample(step_cache_sample_id)
        self.step_cache_flop_recorder.end_sample(step_cache_sample_id)
        self.step_cache_flop_recorder.write_summary()

        if return_latents:
            return video, output
        else:
            return video



    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_clean = []
        # if self.local_attn_size != -1:
        #     # Use the local attention size to compute the KV cache size
        #     kv_cache_size = self.local_attn_size * self.frame_seq_length
        # else:
        #     # Use the default KV cache size
        kv_cache_size = 1560 * 24

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_clean = kv_cache_clean  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
