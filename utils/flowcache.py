import json
import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FlowCacheConfig:
    enabled: bool = False
    debug: bool = False
    output_reuse_enabled: bool = False
    output_reuse_dry_run_enabled: bool = False
    output_reuse_metric: str = "l1rel"
    output_reuse_thresholds: tuple = (0.03, 0.05, 0.08, 0.10)
    output_reuse_log_jsonl: bool = True
    output_reuse_debug_verify: bool = False
    kv_compress_enabled: bool = False
    kv_compress_dry_run: bool = False
    kv_compress_real_enabled: bool = False
    kv_compress_target_ratio: float = 0.5
    kv_compress_min_candidate_tokens: int = 4680
    kv_compress_strategy: str = "uniform"
    kv_compress_protect_sink: bool = True
    kv_compress_protect_current: bool = True
    kv_compress_apply_to_denoise: bool = True
    kv_compress_apply_to_clean_cache_update: bool = False
    kv_compress_apply_to_current_only: bool = False
    kv_compress_shadow_compare: bool = False
    kv_compress_shadow_max_events: int = 5
    kv_compress_max_windows: Optional[Any] = None
    kv_compress_max_layers: Optional[Any] = None
    cache_body_compress_enabled: bool = False
    cache_body_compress_real_enabled: bool = False
    cache_body_compress_target_ratio: float = 0.5
    cache_body_compress_strategy: str = "uniform"
    cache_body_compress_min_candidate_tokens: int = 4680
    cache_body_compress_apply_to_denoise: bool = True
    cache_body_compress_apply_to_clean_cache_update: bool = False
    cache_body_compress_apply_to_current_only: bool = False
    cache_body_compress_protect_sink: bool = True
    cache_body_compress_protect_current: bool = True
    cache_body_compress_max_windows: Optional[Any] = None
    cache_body_compress_max_layers: Optional[Any] = None
    cache_body_compress_debug_verify: bool = True
    persistent_cache_compress_enabled: bool = False
    persistent_cache_compress_real_enabled: bool = False
    persistent_cache_compress_target_ratio: float = 0.5
    persistent_cache_compress_strategy: str = "uniform"
    persistent_cache_compress_min_candidate_tokens: int = 4680
    persistent_cache_compress_apply_to_denoise: bool = True
    persistent_cache_compress_apply_to_clean_cache_update: bool = False
    persistent_cache_compress_apply_to_current_only: bool = False
    persistent_cache_compress_protect_sink: bool = True
    persistent_cache_compress_protect_current: bool = True
    persistent_cache_compress_sidecar_enabled: bool = True
    persistent_cache_compress_debug_verify: bool = True
    persistent_cache_compress_max_windows: Optional[Any] = None
    persistent_cache_compress_max_layers: Optional[Any] = None
    compacted_kv_enabled: bool = False
    compacted_kv_real_enabled: bool = False
    compacted_kv_target_ratio: float = 0.5
    compacted_kv_strategy: str = "uniform"
    compacted_kv_min_history_tokens: int = 4680
    compacted_kv_apply_to_denoise: bool = True
    compacted_kv_apply_to_current_only: bool = False
    compacted_kv_apply_to_clean_cache_update: bool = False
    compacted_kv_protect_sink: bool = True
    compacted_kv_protect_current: bool = True
    compacted_kv_debug_verify: bool = True
    compacted_kv_log_events: bool = True
    compacted_kv_max_events: Optional[int] = None
    eval_metrics_enabled: bool = False
    eval_runtime_enabled: bool = False
    eval_memory_enabled: bool = False
    metadata_enabled: bool = False
    metadata_output_path: Optional[str] = None
    r0_trace_enabled: bool = False
    r0_trace_rng_state: bool = False
    r0_trace_output_path: Optional[str] = None
    r0_stage_match_tolerance: float = 1.0e-4
    compression_candidate_enabled: bool = False
    log_attention_parts: bool = False
    log_compression_candidates: bool = False
    l1rel_threshold: float = 0.0
    protect_sink_tokens: bool = True
    log_every_window: bool = True
    log_kv_ranges: bool = False
    log_summary: bool = True
    profiler_enabled: bool = False
    profiler_mode: str = "coarse"
    profiler_output_path: Optional[str] = "logs/round7_profile.jsonl"
    profiler_summary_path: Optional[str] = "logs/round7_profile_summary.json"
    profiler_log_events: bool = True
    profiler_cuda_synchronize: bool = False
    profiler_fine_enabled: bool = False
    profiler_sample_layers: tuple = (0, 10, 20, 29)
    profiler_sample_every_n_steps: int = 1
    profiler_record_block_forward: bool = True
    profiler_record_attention: bool = True
    profiler_record_mlp: bool = True
    profiler_record_kv_ops: bool = True
    profiler_record_clean_cache_update: bool = True
    profiler_record_decode_save: bool = True
    attention_profiler_enabled: bool = False
    attention_profiler_output_path: Optional[str] = (
        "logs/round8_0_attention_profile.jsonl")
    attention_profiler_summary_path: Optional[str] = (
        "logs/round8_0_attention_profile_summary.json")
    attention_profiler_mode: str = "sampled"
    attention_profiler_cuda_event: bool = True
    attention_profiler_cuda_synchronize: bool = False
    attention_profiler_sample_layers: tuple = (0, 10, 20, 29)
    attention_profiler_sample_every_n_steps: int = 8
    attention_profiler_max_events: int = 240
    attention_profiler_record_rope: bool = True
    attention_profiler_record_qkv: bool = True
    attention_profiler_record_cache_assembly: bool = True
    attention_profiler_record_padding: bool = True
    attention_profiler_record_kernel: bool = True
    attention_profiler_record_output_proj: bool = True
    attention_profiler_record_cache_write: bool = True
    attention_profiler_record_clean_cache_update: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "FlowCacheConfig":
        defaults = cls()
        raw_config = _get_value(args, "flowcache", None)
        if raw_config is None:
            return defaults

        return cls(
            enabled=_get_bool(raw_config, "enabled", defaults.enabled),
            debug=_get_bool(raw_config, "debug", defaults.debug),
            output_reuse_enabled=_get_bool(
                raw_config, "output_reuse_enabled", defaults.output_reuse_enabled),
            output_reuse_dry_run_enabled=_get_bool(
                raw_config,
                "output_reuse_dry_run_enabled",
                defaults.output_reuse_dry_run_enabled),
            output_reuse_metric=_get_str(
                raw_config, "output_reuse_metric", defaults.output_reuse_metric),
            output_reuse_thresholds=_get_float_tuple(
                raw_config,
                "output_reuse_thresholds",
                defaults.output_reuse_thresholds),
            output_reuse_log_jsonl=_get_bool(
                raw_config,
                "output_reuse_log_jsonl",
                defaults.output_reuse_log_jsonl),
            output_reuse_debug_verify=_get_bool(
                raw_config,
                "output_reuse_debug_verify",
                defaults.output_reuse_debug_verify),
            kv_compress_enabled=_get_bool(
                raw_config, "kv_compress_enabled", defaults.kv_compress_enabled),
            kv_compress_dry_run=_get_bool(
                raw_config, "kv_compress_dry_run", defaults.kv_compress_dry_run),
            kv_compress_real_enabled=_get_bool(
                raw_config, "kv_compress_real_enabled",
                defaults.kv_compress_real_enabled),
            kv_compress_target_ratio=_get_float(
                raw_config, "kv_compress_target_ratio",
                defaults.kv_compress_target_ratio),
            kv_compress_min_candidate_tokens=_get_int(
                raw_config, "kv_compress_min_candidate_tokens",
                defaults.kv_compress_min_candidate_tokens),
            kv_compress_strategy=_get_str(
                raw_config, "kv_compress_strategy",
                defaults.kv_compress_strategy),
            kv_compress_protect_sink=_get_bool(
                raw_config, "kv_compress_protect_sink",
                defaults.kv_compress_protect_sink),
            kv_compress_protect_current=_get_bool(
                raw_config, "kv_compress_protect_current",
                defaults.kv_compress_protect_current),
            kv_compress_apply_to_denoise=_get_bool(
                raw_config, "kv_compress_apply_to_denoise",
                defaults.kv_compress_apply_to_denoise),
            kv_compress_apply_to_clean_cache_update=_get_bool(
                raw_config, "kv_compress_apply_to_clean_cache_update",
                defaults.kv_compress_apply_to_clean_cache_update),
            kv_compress_apply_to_current_only=_get_bool(
                raw_config, "kv_compress_apply_to_current_only",
                defaults.kv_compress_apply_to_current_only),
            kv_compress_shadow_compare=_get_bool(
                raw_config, "kv_compress_shadow_compare",
                defaults.kv_compress_shadow_compare),
            kv_compress_shadow_max_events=_get_int(
                raw_config, "kv_compress_shadow_max_events",
                defaults.kv_compress_shadow_max_events),
            kv_compress_max_windows=_get_optional_int_filter(
                raw_config, "kv_compress_max_windows",
                defaults.kv_compress_max_windows),
            kv_compress_max_layers=_get_optional_int_filter(
                raw_config, "kv_compress_max_layers",
                defaults.kv_compress_max_layers),
            cache_body_compress_enabled=_get_bool(
                raw_config, "cache_body_compress_enabled",
                defaults.cache_body_compress_enabled),
            cache_body_compress_real_enabled=_get_bool(
                raw_config, "cache_body_compress_real_enabled",
                defaults.cache_body_compress_real_enabled),
            cache_body_compress_target_ratio=_get_float(
                raw_config, "cache_body_compress_target_ratio",
                defaults.cache_body_compress_target_ratio),
            cache_body_compress_strategy=_get_str(
                raw_config, "cache_body_compress_strategy",
                defaults.cache_body_compress_strategy),
            cache_body_compress_min_candidate_tokens=_get_int(
                raw_config, "cache_body_compress_min_candidate_tokens",
                defaults.cache_body_compress_min_candidate_tokens),
            cache_body_compress_apply_to_denoise=_get_bool(
                raw_config, "cache_body_compress_apply_to_denoise",
                defaults.cache_body_compress_apply_to_denoise),
            cache_body_compress_apply_to_clean_cache_update=_get_bool(
                raw_config, "cache_body_compress_apply_to_clean_cache_update",
                defaults.cache_body_compress_apply_to_clean_cache_update),
            cache_body_compress_apply_to_current_only=_get_bool(
                raw_config, "cache_body_compress_apply_to_current_only",
                defaults.cache_body_compress_apply_to_current_only),
            cache_body_compress_protect_sink=_get_bool(
                raw_config, "cache_body_compress_protect_sink",
                defaults.cache_body_compress_protect_sink),
            cache_body_compress_protect_current=_get_bool(
                raw_config, "cache_body_compress_protect_current",
                defaults.cache_body_compress_protect_current),
            cache_body_compress_max_windows=_get_optional_int_filter(
                raw_config, "cache_body_compress_max_windows",
                defaults.cache_body_compress_max_windows),
            cache_body_compress_max_layers=_get_optional_int_filter(
                raw_config, "cache_body_compress_max_layers",
                defaults.cache_body_compress_max_layers),
            cache_body_compress_debug_verify=_get_bool(
                raw_config, "cache_body_compress_debug_verify",
                defaults.cache_body_compress_debug_verify),
            persistent_cache_compress_enabled=_get_bool(
                raw_config, "persistent_cache_compress_enabled",
                defaults.persistent_cache_compress_enabled),
            persistent_cache_compress_real_enabled=_get_bool(
                raw_config, "persistent_cache_compress_real_enabled",
                defaults.persistent_cache_compress_real_enabled),
            persistent_cache_compress_target_ratio=_get_float(
                raw_config, "persistent_cache_compress_target_ratio",
                defaults.persistent_cache_compress_target_ratio),
            persistent_cache_compress_strategy=_get_str(
                raw_config, "persistent_cache_compress_strategy",
                defaults.persistent_cache_compress_strategy),
            persistent_cache_compress_min_candidate_tokens=_get_int(
                raw_config, "persistent_cache_compress_min_candidate_tokens",
                defaults.persistent_cache_compress_min_candidate_tokens),
            persistent_cache_compress_apply_to_denoise=_get_bool(
                raw_config, "persistent_cache_compress_apply_to_denoise",
                defaults.persistent_cache_compress_apply_to_denoise),
            persistent_cache_compress_apply_to_clean_cache_update=_get_bool(
                raw_config, "persistent_cache_compress_apply_to_clean_cache_update",
                defaults.persistent_cache_compress_apply_to_clean_cache_update),
            persistent_cache_compress_apply_to_current_only=_get_bool(
                raw_config, "persistent_cache_compress_apply_to_current_only",
                defaults.persistent_cache_compress_apply_to_current_only),
            persistent_cache_compress_protect_sink=_get_bool(
                raw_config, "persistent_cache_compress_protect_sink",
                defaults.persistent_cache_compress_protect_sink),
            persistent_cache_compress_protect_current=_get_bool(
                raw_config, "persistent_cache_compress_protect_current",
                defaults.persistent_cache_compress_protect_current),
            persistent_cache_compress_sidecar_enabled=_get_bool(
                raw_config, "persistent_cache_compress_sidecar_enabled",
                defaults.persistent_cache_compress_sidecar_enabled),
            persistent_cache_compress_debug_verify=_get_bool(
                raw_config, "persistent_cache_compress_debug_verify",
                defaults.persistent_cache_compress_debug_verify),
            persistent_cache_compress_max_windows=_get_optional_int_filter(
                raw_config, "persistent_cache_compress_max_windows",
                defaults.persistent_cache_compress_max_windows),
            persistent_cache_compress_max_layers=_get_optional_int_filter(
                raw_config, "persistent_cache_compress_max_layers",
                defaults.persistent_cache_compress_max_layers),
            compacted_kv_enabled=_get_bool(
                raw_config, "compacted_kv_enabled",
                defaults.compacted_kv_enabled),
            compacted_kv_real_enabled=_get_bool(
                raw_config, "compacted_kv_real_enabled",
                defaults.compacted_kv_real_enabled),
            compacted_kv_target_ratio=_get_float(
                raw_config, "compacted_kv_target_ratio",
                defaults.compacted_kv_target_ratio),
            compacted_kv_strategy=_get_str(
                raw_config, "compacted_kv_strategy",
                defaults.compacted_kv_strategy),
            compacted_kv_min_history_tokens=_get_int(
                raw_config, "compacted_kv_min_history_tokens",
                defaults.compacted_kv_min_history_tokens),
            compacted_kv_apply_to_denoise=_get_bool(
                raw_config, "compacted_kv_apply_to_denoise",
                defaults.compacted_kv_apply_to_denoise),
            compacted_kv_apply_to_current_only=_get_bool(
                raw_config, "compacted_kv_apply_to_current_only",
                defaults.compacted_kv_apply_to_current_only),
            compacted_kv_apply_to_clean_cache_update=_get_bool(
                raw_config, "compacted_kv_apply_to_clean_cache_update",
                defaults.compacted_kv_apply_to_clean_cache_update),
            compacted_kv_protect_sink=_get_bool(
                raw_config, "compacted_kv_protect_sink",
                defaults.compacted_kv_protect_sink),
            compacted_kv_protect_current=_get_bool(
                raw_config, "compacted_kv_protect_current",
                defaults.compacted_kv_protect_current),
            compacted_kv_debug_verify=_get_bool(
                raw_config, "compacted_kv_debug_verify",
                defaults.compacted_kv_debug_verify),
            compacted_kv_log_events=_get_bool(
                raw_config, "compacted_kv_log_events",
                defaults.compacted_kv_log_events),
            compacted_kv_max_events=_get_optional_int_value(
                raw_config, "compacted_kv_max_events",
                defaults.compacted_kv_max_events),
            eval_metrics_enabled=_get_bool(
                raw_config, "eval_metrics_enabled",
                defaults.eval_metrics_enabled),
            eval_runtime_enabled=_get_bool(
                raw_config, "eval_runtime_enabled",
                defaults.eval_runtime_enabled),
            eval_memory_enabled=_get_bool(
                raw_config, "eval_memory_enabled",
                defaults.eval_memory_enabled),
            metadata_enabled=_get_bool(
                raw_config, "metadata_enabled", defaults.metadata_enabled),
            metadata_output_path=_get_optional_str(
                raw_config, "metadata_output_path", defaults.metadata_output_path),
            r0_trace_enabled=_get_bool(
                raw_config, "r0_trace_enabled", defaults.r0_trace_enabled),
            r0_trace_rng_state=_get_bool(
                raw_config, "r0_trace_rng_state", defaults.r0_trace_rng_state),
            r0_trace_output_path=_get_optional_str(
                raw_config, "r0_trace_output_path", defaults.r0_trace_output_path),
            r0_stage_match_tolerance=_get_float(
                raw_config,
                "r0_stage_match_tolerance",
                defaults.r0_stage_match_tolerance),
            compression_candidate_enabled=_get_bool(
                raw_config, "compression_candidate_enabled",
                defaults.compression_candidate_enabled),
            log_attention_parts=_get_bool(
                raw_config, "log_attention_parts", defaults.log_attention_parts),
            log_compression_candidates=_get_bool(
                raw_config, "log_compression_candidates",
                defaults.log_compression_candidates),
            l1rel_threshold=_get_float(
                raw_config, "l1rel_threshold", defaults.l1rel_threshold),
            protect_sink_tokens=_get_bool(
                raw_config, "protect_sink_tokens", defaults.protect_sink_tokens),
            log_every_window=_get_bool(
                raw_config, "log_every_window", defaults.log_every_window),
            log_kv_ranges=_get_bool(
                raw_config, "log_kv_ranges", defaults.log_kv_ranges),
            log_summary=_get_bool(
                raw_config, "log_summary", defaults.log_summary),
            profiler_enabled=_get_bool(
                raw_config, "profiler_enabled", defaults.profiler_enabled),
            profiler_mode=_get_str(
                raw_config, "profiler_mode", defaults.profiler_mode),
            profiler_output_path=_get_optional_str(
                raw_config, "profiler_output_path",
                defaults.profiler_output_path),
            profiler_summary_path=_get_optional_str(
                raw_config, "profiler_summary_path",
                defaults.profiler_summary_path),
            profiler_log_events=_get_bool(
                raw_config, "profiler_log_events",
                defaults.profiler_log_events),
            profiler_cuda_synchronize=_get_bool(
                raw_config, "profiler_cuda_synchronize",
                defaults.profiler_cuda_synchronize),
            profiler_fine_enabled=_get_bool(
                raw_config, "profiler_fine_enabled",
                defaults.profiler_fine_enabled),
            profiler_sample_layers=_get_optional_int_filter(
                raw_config, "profiler_sample_layers",
                defaults.profiler_sample_layers) or (),
            profiler_sample_every_n_steps=max(
                1,
                _get_int(
                    raw_config,
                    "profiler_sample_every_n_steps",
                    defaults.profiler_sample_every_n_steps)),
            profiler_record_block_forward=_get_bool(
                raw_config, "profiler_record_block_forward",
                defaults.profiler_record_block_forward),
            profiler_record_attention=_get_bool(
                raw_config, "profiler_record_attention",
                defaults.profiler_record_attention),
            profiler_record_mlp=_get_bool(
                raw_config, "profiler_record_mlp",
                defaults.profiler_record_mlp),
            profiler_record_kv_ops=_get_bool(
                raw_config, "profiler_record_kv_ops",
                defaults.profiler_record_kv_ops),
            profiler_record_clean_cache_update=_get_bool(
                raw_config, "profiler_record_clean_cache_update",
                defaults.profiler_record_clean_cache_update),
            profiler_record_decode_save=_get_bool(
                raw_config, "profiler_record_decode_save",
                defaults.profiler_record_decode_save),
            attention_profiler_enabled=_get_bool(
                raw_config,
                "attention_profiler_enabled",
                defaults.attention_profiler_enabled),
            attention_profiler_output_path=_get_optional_str(
                raw_config,
                "attention_profiler_output_path",
                defaults.attention_profiler_output_path),
            attention_profiler_summary_path=_get_optional_str(
                raw_config,
                "attention_profiler_summary_path",
                defaults.attention_profiler_summary_path),
            attention_profiler_mode=_get_str(
                raw_config,
                "attention_profiler_mode",
                defaults.attention_profiler_mode),
            attention_profiler_cuda_event=_get_bool(
                raw_config,
                "attention_profiler_cuda_event",
                defaults.attention_profiler_cuda_event),
            attention_profiler_cuda_synchronize=_get_bool(
                raw_config,
                "attention_profiler_cuda_synchronize",
                defaults.attention_profiler_cuda_synchronize),
            attention_profiler_sample_layers=_get_optional_int_filter(
                raw_config,
                "attention_profiler_sample_layers",
                defaults.attention_profiler_sample_layers) or (),
            attention_profiler_sample_every_n_steps=max(
                1,
                _get_int(
                    raw_config,
                    "attention_profiler_sample_every_n_steps",
                    defaults.attention_profiler_sample_every_n_steps)),
            attention_profiler_max_events=max(
                0,
                _get_int(
                    raw_config,
                    "attention_profiler_max_events",
                    defaults.attention_profiler_max_events)),
            attention_profiler_record_rope=_get_bool(
                raw_config,
                "attention_profiler_record_rope",
                defaults.attention_profiler_record_rope),
            attention_profiler_record_qkv=_get_bool(
                raw_config,
                "attention_profiler_record_qkv",
                defaults.attention_profiler_record_qkv),
            attention_profiler_record_cache_assembly=_get_bool(
                raw_config,
                "attention_profiler_record_cache_assembly",
                defaults.attention_profiler_record_cache_assembly),
            attention_profiler_record_padding=_get_bool(
                raw_config,
                "attention_profiler_record_padding",
                defaults.attention_profiler_record_padding),
            attention_profiler_record_kernel=_get_bool(
                raw_config,
                "attention_profiler_record_kernel",
                defaults.attention_profiler_record_kernel),
            attention_profiler_record_output_proj=_get_bool(
                raw_config,
                "attention_profiler_record_output_proj",
                defaults.attention_profiler_record_output_proj),
            attention_profiler_record_cache_write=_get_bool(
                raw_config,
                "attention_profiler_record_cache_write",
                defaults.attention_profiler_record_cache_write),
            attention_profiler_record_clean_cache_update=_get_bool(
                raw_config,
                "attention_profiler_record_clean_cache_update",
                defaults.attention_profiler_record_clean_cache_update),
        )


@dataclass
class FlowCacheWindowRecord:
    window_index: int
    start_block: int
    end_block: int
    num_frame_per_block: int
    rolling_window_length_blocks: int
    current_start_frame: Optional[int]
    current_end_frame: Optional[int]
    current_num_frames: Optional[int]
    denoising_step_list: List[Any]
    noisy_cache_shape: Optional[tuple]
    noisy_input_shape: Optional[tuple]
    current_timestep_shape: Optional[tuple]
    begin_timestamp: float
    end_timestamp: Optional[float] = None
    elapsed_sec: Optional[float] = None


@dataclass
class FlowCacheKVRange:
    event: str
    timestamp: float
    window_index: int
    start_block: int
    end_block: int
    num_frame_per_block: int
    rolling_window_length_blocks: int
    layer_idx: int
    sink_start: int
    sink_end: int
    history_start: int
    history_end: int
    current_start: int
    current_end: int
    global_end_index: Optional[int]
    local_end_index: Optional[int]
    cache_tokens: int
    working_history_tokens: int
    current_tokens: int
    total_visible_kv_tokens: int
    total_kv_tokens: int
    block_length: int
    frame_seq_length: int
    current_num_frames: Optional[int]
    max_attention_tokens: Optional[int]


@dataclass
class FlowCacheAttentionParts:
    event: str
    timestamp: float
    source_event: str
    window_index: Optional[int]
    layer_idx: Optional[int]
    attention_branch: str
    updating_cache: bool
    block_length: int
    sink_tokens: int
    anchor_tokens: int
    anchor_value_tokens: int
    working_tokens: int
    working_value_tokens: int
    current_tokens: int
    current_value_tokens: int
    input_tokens: int
    input_value_tokens: int
    total_visible_kv_tokens: int
    total_kv_tokens: int
    current_start: Optional[int]
    cache_start: Optional[int]
    cache_end: Optional[int]
    global_end_index: Optional[int]
    local_end_index: Optional[int]


@dataclass
class FlowCacheCompressionCandidate:
    event: str
    timestamp: float
    source_event: str
    window_index: Optional[int]
    layer_idx: Optional[int]
    attention_branch: str
    block_length: int
    sink_tokens: int
    anchor_tokens: int
    working_tokens: int
    current_tokens: int
    input_tokens: int
    total_visible_kv_tokens: int
    total_kv_tokens: int
    protected_sink_tokens: int
    protected_current_tokens: int
    compressible_history_tokens: int
    candidate_ratio: float
    candidate_region_start: int
    candidate_region_end: int
    global_end_index: Optional[int]
    local_end_index: Optional[int]


@dataclass
class FlowCacheCompressionDryRun:
    event: str
    timestamp: float
    window_index: Optional[int]
    layer_idx: Optional[int]
    branch: str
    attention_event: str
    source_event: str
    protected_sink_tokens: int
    protected_current_tokens: int
    compressible_history_tokens: int
    original_total_kv_tokens: int
    projected_history_tokens: int
    projected_total_kv_tokens: int
    saved_tokens: int
    saving_ratio: float
    candidate_keep_ratio: float
    kv_compress_target_ratio: float
    apply_to_branch: bool
    skipped_reason: Optional[str]
    candidate_region_start: int
    candidate_region_end: int


@dataclass
class FlowCacheRealCompression:
    event: str
    timestamp: float
    window_index: Optional[int]
    layer_idx: Optional[int]
    branch: str
    attention_event: str
    source_event: str
    applied: bool
    skipped_reason: Optional[str]
    strategy: str
    kv_compress_target_ratio: float
    original_history_tokens: int
    compressed_history_tokens: int
    protected_sink_tokens: int
    protected_current_tokens: int
    original_total_kv_tokens: int
    compressed_total_kv_tokens: int
    saved_tokens: int
    saving_ratio: float
    keep_ratio: float
    candidate_region_start: int
    candidate_region_end: int
    fallback: bool


@dataclass
class FlowCacheCacheBodyCompression:
    event: str
    timestamp: float
    window_index: Optional[int]
    layer_idx: Optional[int]
    branch: str
    attention_event: str
    source_event: str
    applied: bool
    skipped_reason: Optional[str]
    original_history_tokens: int
    compressed_history_tokens: int
    protected_sink_tokens: int
    protected_current_tokens: int
    original_visible_tokens: int
    compressed_visible_tokens: int
    saved_visible_tokens: int
    visible_saving_ratio: float
    strategy: str
    cache_body_compress_target_ratio: float
    keep_ratio: float
    mode: str
    fallback: bool


@dataclass
class FlowCachePersistentCacheCompression:
    event: str
    timestamp: float
    window_index: Optional[int]
    layer_idx: Optional[int]
    branch: str
    attention_event: str
    source_event: str
    applied: bool
    skipped_reason: Optional[str]
    mode: str
    original_history_tokens: int
    compressed_history_tokens: int
    protected_sink_tokens: int
    protected_current_tokens: int
    original_visible_tokens: int
    compressed_visible_tokens: int
    saved_visible_tokens: int
    visible_saving_ratio: float
    strategy: str
    target_ratio: float
    keep_ratio: float
    sidecar_created: bool
    sidecar_reused: bool
    sidecar_invalidated: bool
    sidecar_fallback: bool
    sidecar_valid: bool
    source_start: Optional[int]
    source_end: Optional[int]
    fallback: bool


@dataclass
class FlowCachePersistentCacheSidecar:
    event: str
    timestamp: float
    action: str
    window_index: Optional[int]
    layer_idx: Optional[int]
    source_start: Optional[int]
    source_end: Optional[int]
    compressed_tokens: int
    reason: Optional[str]


@dataclass
class FlowCacheCompactedKV:
    event: str
    timestamp: float
    prompt_idx: Optional[int]
    sample_idx: Optional[int]
    window_index: Optional[int]
    layer_idx: Optional[int]
    branch: str
    attention_event: str
    source_event: str
    applied: bool
    skipped_reason: Optional[str]
    fallback: bool
    fallback_reason: Optional[str]
    original_history_tokens: int
    compacted_history_tokens: int
    protected_sink_tokens: int
    protected_current_tokens: int
    original_visible_tokens: int
    compacted_visible_tokens: int
    saved_visible_tokens: int
    visible_saving_ratio: float
    target_ratio: float
    strategy: str
    keep_ratio: float
    dtype: Optional[str]
    device: Optional[str]
    elapsed_ms: Optional[float]


@dataclass
class FlowCacheAttentionOutputDiff:
    event: str
    timestamp: float
    window_index: Optional[int]
    layer_idx: Optional[int]
    branch: str
    attention_event: str
    source_event: str
    mean_abs_diff: float
    max_abs_diff: float
    relative_l1: float
    cosine_similarity: Optional[float]


@dataclass
class FlowCacheOutputReuseDryRun:
    event: str
    timestamp: float
    prompt_idx: Optional[int]
    sample_idx: Optional[int]
    window_idx: Optional[int]
    window_index: Optional[int]
    step_idx: Optional[int]
    previous_step_idx: Optional[int]
    timestep: Optional[float]
    previous_timestep: Optional[float]
    block_idx: Optional[int]
    chunk_idx: Optional[int]
    block_id: str
    chunk_id: str
    start_block: Optional[int]
    end_block: Optional[int]
    metric: str
    l1rel: Optional[float]
    threshold: float
    reusable_by_threshold: bool
    tensor_shape: str
    tensor_like: int
    fallback: bool
    fallback_reason: Optional[str]
    warning_reason: Optional[str]


class FlowCacheManager:
    """
    Default-off FlowCache metadata tracker.

    Round 2 only records lightweight metadata. It never stores tensor contents,
    mutates caches, changes attention inputs, or skips model execution.
    """

    def __init__(self, config: Optional[FlowCacheConfig] = None):
        self.config = config or FlowCacheConfig()
        self.window_records: List[FlowCacheWindowRecord] = []
        self.kv_range_records: List[FlowCacheKVRange] = []
        self.attention_part_records: List[FlowCacheAttentionParts] = []
        self.compression_candidate_records: List[FlowCacheCompressionCandidate] = []
        self.compression_dry_run_records: List[FlowCacheCompressionDryRun] = []
        self.real_compression_records: List[FlowCacheRealCompression] = []
        self.cache_body_compression_records: List[FlowCacheCacheBodyCompression] = []
        self.persistent_cache_compression_records: List[FlowCachePersistentCacheCompression] = []
        self.persistent_cache_sidecar_records: List[FlowCachePersistentCacheSidecar] = []
        self.compacted_kv_records: List[FlowCacheCompactedKV] = []
        self.attention_output_diff_records: List[FlowCacheAttentionOutputDiff] = []
        self.output_reuse_dry_run_records: List[FlowCacheOutputReuseDryRun] = []
        self.r0_trace_records: List[Dict[str, Any]] = []
        self.event_records: List[Dict[str, Any]] = []
        self._active_windows: Dict[int, FlowCacheWindowRecord] = {}
        self._warned_jsonl = False
        self._warning_count = 0
        self._output_reuse_total_candidates = 0
        self._output_reuse_valid_metric_count = 0
        self._output_reuse_l1rel_values: List[float] = []
        self._output_reuse_reusable_count_by_threshold: Dict[str, int] = {}
        self._output_reuse_parse_errors = 0
        self._created_timestamp = time.time()
        self._flowcache_run_id = (
            f"{os.getpid()}-{int(self._created_timestamp * 1000000)}"
        )
        self._persistent_sidecar_action_counts: Dict[str, int] = {}
        self._persistent_summary_index = 0
        self._profiler_started = False
        self._profiler_run_context: Dict[str, Any] = {}
        self._profiler_phase_stats: Dict[str, Dict[str, Any]] = {}
        self._profiler_events: List[Dict[str, Any]] = []
        self._profiler_sampled_layer_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._profiler_sampled_step_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._profiler_overhead_sec = 0.0
        self._profiler_saved_videos = 0
        self._profiler_summary_written = False
        self._profiler_warned_jsonl = False
        self._profiler_warned_summary = False
        self._attention_profiler_started = False
        self._attention_profiler_summary_written = False
        self._attention_profiler_warned_jsonl = False
        self._attention_profiler_warned_summary = False
        self._attention_profiler_events: List[Dict[str, Any]] = []
        self._attention_profiler_phase_stats: Dict[str, Dict[str, Any]] = {}
        self._attention_profiler_layer_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._attention_profiler_window_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._attention_profiler_branch_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._attention_profiler_timer_type_counts: Dict[str, int] = {}
        self._attention_profiler_flash_counts: Dict[str, int] = {}
        self._attention_profiler_fallback_count = 0
        self._attention_profiler_overhead_sec = 0.0
        self._attention_profiler_max_reached = False
        self._attention_profiler_context: Dict[str, Any] = {}
        self._attention_profiler_run_context: Dict[str, Any] = {}

    @classmethod
    def from_args(cls, args: Any) -> "FlowCacheManager":
        return cls(FlowCacheConfig.from_args(args))

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def debug(self) -> bool:
        return self.config.debug

    @property
    def profiler_enabled(self) -> bool:
        return self.config.profiler_enabled

    @property
    def attention_profiler_enabled(self) -> bool:
        return self.config.attention_profiler_enabled

    @property
    def should_profile_attention(self) -> bool:
        return (
            self.config.profiler_enabled and
            self.config.profiler_fine_enabled and
            self.config.profiler_record_attention and
            self.config.profiler_mode.lower() in {
                "fine", "fine_sampled", "sampled_fine", "detailed", "full", "all"
            }
        )

    @property
    def should_profile_block_forward(self) -> bool:
        return (
            self.config.profiler_enabled and
            self.config.profiler_fine_enabled and
            self.config.profiler_record_block_forward and
            self.config.profiler_mode.lower() in {
                "fine", "fine_sampled", "sampled_fine", "detailed", "full", "all"
            }
        )

    @property
    def should_profile_mlp(self) -> bool:
        return (
            self.config.profiler_enabled and
            self.config.profiler_fine_enabled and
            self.config.profiler_record_mlp and
            self.config.profiler_mode.lower() in {
                "fine", "fine_sampled", "sampled_fine", "detailed", "full", "all"
            }
        )

    @property
    def should_profile_kv_ops(self) -> bool:
        return (
            self.config.profiler_enabled and
            self.config.profiler_fine_enabled and
            self.config.profiler_record_kv_ops and
            self.config.profiler_mode.lower() in {
                "fine", "fine_sampled", "sampled_fine", "detailed", "full", "all"
            }
        )

    def should_profile_phase(
        self,
        phase: str,
        *,
        window_index: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ) -> bool:
        if not self.config.profiler_enabled:
            return False

        fine_phases = {
            "block_forward_sampled",
            "attention_forward_sampled",
            "attention_qkv_sampled",
            "attention_compute_sampled",
            "attention_output_sampled",
            "cross_attention_sampled",
            "mlp_forward_sampled",
            "kv_cache_read_sampled",
            "kv_cache_write_sampled",
            "kv_eviction_sampled",
        }
        if phase in fine_phases:
            if phase in {
                "attention_forward_sampled",
                "attention_qkv_sampled",
                "attention_compute_sampled",
                "attention_output_sampled",
                "cross_attention_sampled",
            } and not self.should_profile_attention:
                return False
            if phase == "block_forward_sampled" and not self.should_profile_block_forward:
                return False
            if phase == "mlp_forward_sampled" and not self.should_profile_mlp:
                return False
            if phase in {
                "kv_cache_read_sampled",
                "kv_cache_write_sampled",
                "kv_eviction_sampled",
            } and not self.should_profile_kv_ops:
                return False
            if not self._profiler_layer_selected(layer_idx):
                return False
            if not self._profiler_step_selected(window_index):
                return False

        if phase == "clean_cache_update_total":
            return self.config.profiler_record_clean_cache_update

        if phase in {"vae_decode_total", "video_save_total"}:
            return self.config.profiler_record_decode_save

        return True

    def start_profiler_run(
        self,
        *,
        prompt_count: Optional[int] = None,
        sample_count: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> None:
        if not self.config.profiler_enabled or self._profiler_started:
            return

        overhead_start = time.perf_counter()
        try:
            self._profiler_started = True
            self._profiler_run_context = {
                "prompt_count": prompt_count,
                "sample_count": sample_count,
                "device": str(device) if device is not None else None,
                "started_timestamp": time.time(),
                "started_perf_counter": time.perf_counter(),
                "mode": self.config.profiler_mode,
                "cuda_synchronize": self.config.profiler_cuda_synchronize,
                "fine_enabled": self.config.profiler_fine_enabled,
            }
            self._profiler_reset_peak_memory()
            self._write_profiler_jsonl({
                "event": "profiler_start",
                "phase": "profiler_start",
                "prompt_idx": None,
                "sample_idx": None,
                "elapsed_sec": 0.0,
                "elapsed_ms": 0.0,
                "count": 1,
                "mode": self.config.profiler_mode,
                "cuda_synchronize": self.config.profiler_cuda_synchronize,
                "fine_enabled": self.config.profiler_fine_enabled,
                "device": str(device) if device is not None else None,
                "warning_count": self._warning_count,
                "fallback_reason": None,
                "prompt_count": prompt_count,
                "sample_count": sample_count,
            })
            print(
                "[FlowCache][profiler] "
                f"enabled=true mode={self.config.profiler_mode} "
                f"fine_enabled={self.config.profiler_fine_enabled} "
                f"cuda_synchronize={self.config.profiler_cuda_synchronize} "
                f"output_path={self.config.profiler_output_path} "
                f"summary_path={self.config.profiler_summary_path}"
            )
        except Exception as exc:
            self._warning(f"profiler start failed: {exc}")
        finally:
            self._profiler_overhead_sec += time.perf_counter() - overhead_start

    def start_profiler_phase(
        self,
        phase: str,
        *,
        prompt_idx: Optional[int] = None,
        sample_idx: Optional[int] = None,
        window_index: Optional[int] = None,
        layer_idx: Optional[int] = None,
        count: int = 1,
        device: Optional[Any] = None,
        fallback_reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.should_profile_phase(
            phase, window_index=window_index, layer_idx=layer_idx):
            return None

        overhead_start = time.perf_counter()
        try:
            if not self._profiler_started:
                self.start_profiler_run(device=device)
            self._profiler_sync(device)
            return {
                "phase": phase,
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "window_index": window_index,
                "layer_idx": layer_idx,
                "count": count,
                "device": str(device) if device is not None else None,
                "_sync_device": device,
                "fallback_reason": fallback_reason,
                "start_perf_counter": time.perf_counter(),
            }
        except Exception as exc:
            self._warning(f"profiler phase start failed for phase={phase}: {exc}")
            return None
        finally:
            self._profiler_overhead_sec += time.perf_counter() - overhead_start

    def end_profiler_phase(
        self,
        token: Optional[Dict[str, Any]],
        *,
        count: Optional[int] = None,
        fallback_reason: Optional[str] = None,
    ) -> None:
        if token is None or not self.config.profiler_enabled:
            return

        overhead_start = time.perf_counter()
        try:
            self._profiler_sync(token.get("_sync_device"))
            elapsed_sec = max(
                0.0,
                time.perf_counter() - float(token["start_perf_counter"]))
            phase = str(token["phase"])
            event_count = int(count if count is not None else token.get("count", 1))
            reason = fallback_reason or token.get("fallback_reason")
            payload = {
                "event": "profile_phase",
                "phase": phase,
                "prompt_idx": token.get("prompt_idx"),
                "sample_idx": token.get("sample_idx"),
                "window_index": token.get("window_index"),
                "layer_idx": token.get("layer_idx"),
                "elapsed_sec": elapsed_sec,
                "elapsed_ms": elapsed_sec * 1000.0,
                "count": event_count,
                "mode": self.config.profiler_mode,
                "cuda_synchronize": self.config.profiler_cuda_synchronize,
                "fine_enabled": self.config.profiler_fine_enabled,
                "device": token.get("device"),
                "warning_count": self._warning_count,
                "fallback_reason": reason,
            }
            self._record_profiler_payload(payload)
        except Exception as exc:
            self._warning(
                f"profiler phase end failed for phase={token.get('phase')}: {exc}")
        finally:
            self._profiler_overhead_sec += time.perf_counter() - overhead_start

    def profile_phase(
        self,
        phase: str,
        *,
        prompt_idx: Optional[int] = None,
        sample_idx: Optional[int] = None,
        window_index: Optional[int] = None,
        layer_idx: Optional[int] = None,
        count: int = 1,
        device: Optional[Any] = None,
        fallback_reason: Optional[str] = None,
    ) -> "_FlowCacheProfilerScope":
        return _FlowCacheProfilerScope(
            self,
            phase,
            prompt_idx=prompt_idx,
            sample_idx=sample_idx,
            window_index=window_index,
            layer_idx=layer_idx,
            count=count,
            device=device,
            fallback_reason=fallback_reason,
        )

    def start_attention_profiler_run(
        self,
        *,
        prompt_count: Optional[int] = None,
        sample_count: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> None:
        if (
            not self.config.attention_profiler_enabled or
            self._attention_profiler_started
        ):
            return

        overhead_start = time.perf_counter()
        try:
            self._attention_profiler_started = True
            self._attention_profiler_run_context = {
                "prompt_count": prompt_count,
                "sample_count": sample_count,
                "device": str(device) if device is not None else None,
                "started_timestamp": time.time(),
                "started_perf_counter": time.perf_counter(),
                "mode": self.config.attention_profiler_mode,
                "cuda_event": self.config.attention_profiler_cuda_event,
                "cuda_synchronize": (
                    self.config.attention_profiler_cuda_synchronize),
            }
            self._profiler_reset_peak_memory()
        except Exception as exc:
            self._warning(f"attention profiler start failed: {exc}")
        finally:
            self._attention_profiler_overhead_sec += (
                time.perf_counter() - overhead_start)

    def set_attention_profiler_context(
        self,
        *,
        prompt_idx: Optional[int] = None,
        sample_idx: Optional[int] = None,
        window_idx: Optional[int] = None,
        step_idx: Optional[Any] = None,
        timestep: Optional[Any] = None,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        previous = dict(self._attention_profiler_context)
        if self.config.attention_profiler_enabled:
            self._attention_profiler_context = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "window_idx": window_idx,
                "step_idx": step_idx,
                "timestep": timestep,
                "branch": branch,
            }
        return previous

    def restore_attention_profiler_context(
        self,
        context: Optional[Dict[str, Any]],
    ) -> None:
        if self.config.attention_profiler_enabled:
            self._attention_profiler_context = dict(context or {})

    def should_profile_attention_path(
        self,
        *,
        phase: Optional[str] = None,
        window_idx: Optional[int] = None,
        window_index: Optional[int] = None,
        layer_idx: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> bool:
        if not self.config.attention_profiler_enabled:
            return False
        if phase is not None and not self._attention_profiler_phase_enabled(
            phase, branch=branch):
            return False

        selected_window = window_idx if window_idx is not None else window_index
        if selected_window is None:
            selected_window = self._attention_profiler_context.get("window_idx")
        mode = str(self.config.attention_profiler_mode).lower()
        if mode not in {"all", "full", "detailed"}:
            if not self._attention_profiler_layer_selected(layer_idx):
                return False
            if not self._attention_profiler_window_selected(selected_window):
                return False

        max_events = int(self.config.attention_profiler_max_events)
        if max_events > 0 and len(self._attention_profiler_events) >= max_events:
            self._attention_profiler_max_reached = True
            return False
        return True

    def start_attention_profiler_phase(
        self,
        phase: str,
        *,
        prompt_idx: Optional[int] = None,
        sample_idx: Optional[int] = None,
        window_idx: Optional[int] = None,
        window_index: Optional[int] = None,
        step_idx: Optional[Any] = None,
        timestep: Optional[Any] = None,
        layer_idx: Optional[int] = None,
        branch: Optional[str] = None,
        device: Optional[Any] = None,
        visible_tokens: Optional[int] = None,
        history_tokens: Optional[int] = None,
        current_tokens: Optional[int] = None,
        sink_tokens: Optional[int] = None,
        q_shape: Optional[Any] = None,
        k_shape: Optional[Any] = None,
        v_shape: Optional[Any] = None,
        dtype: Optional[Any] = None,
        used_flash_attention: Optional[Any] = None,
        fallback_reason: Optional[str] = None,
        warning: Optional[str] = None,
        timing_scope: str = "subphase",
    ) -> Optional[Dict[str, Any]]:
        context = self._attention_profiler_context
        selected_window = (
            window_idx if window_idx is not None else
            window_index if window_index is not None else
            context.get("window_idx")
        )
        selected_branch = branch if branch is not None else context.get("branch")
        if not self.should_profile_attention_path(
            phase=phase,
            window_idx=selected_window,
            layer_idx=layer_idx,
            branch=selected_branch,
        ):
            return None

        overhead_start = time.perf_counter()
        try:
            if not self._attention_profiler_started:
                self.start_attention_profiler_run(device=device)
            timer_type = "cpu_wall"
            timer_fallback_reason = fallback_reason
            start_event = None
            end_event = None
            if self.config.attention_profiler_cuda_event:
                try:
                    import torch
                    has_cuda_device = (
                        device is not None and
                        (
                            getattr(device, "type", None) == "cuda" or
                            str(device).startswith("cuda")
                        )
                    )
                    if torch.cuda.is_available() and has_cuda_device:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        timer_type = "cuda_event"
                    else:
                        timer_fallback_reason = (
                            timer_fallback_reason or "cuda_event_not_applicable")
                except Exception as exc:
                    timer_fallback_reason = (
                        timer_fallback_reason or
                        f"cuda_event_start_failed:{exc}")
                    start_event = None
                    end_event = None
            else:
                timer_fallback_reason = (
                    timer_fallback_reason or "cuda_event_disabled")

            return {
                "phase": phase,
                "prompt_idx": (
                    prompt_idx if prompt_idx is not None else
                    context.get("prompt_idx")),
                "sample_idx": (
                    sample_idx if sample_idx is not None else
                    context.get("sample_idx")),
                "window_idx": selected_window,
                "step_idx": (
                    step_idx if step_idx is not None else
                    context.get("step_idx")),
                "timestep": (
                    timestep if timestep is not None else
                    context.get("timestep")),
                "layer_idx": layer_idx,
                "branch": selected_branch,
                "device": str(device) if device is not None else None,
                "visible_tokens": visible_tokens,
                "history_tokens": history_tokens,
                "current_tokens": current_tokens,
                "sink_tokens": sink_tokens,
                "q_shape": self._attention_shape_text(q_shape),
                "k_shape": self._attention_shape_text(k_shape),
                "v_shape": self._attention_shape_text(v_shape),
                "dtype": str(dtype) if dtype is not None else None,
                "used_flash_attention": used_flash_attention,
                "fallback_reason": timer_fallback_reason,
                "warning": warning,
                "timing_scope": timing_scope,
                "timer_type": timer_type,
                "start_event": start_event,
                "end_event": end_event,
                "start_perf_counter": time.perf_counter(),
            }
        except Exception as exc:
            self._warning(
                f"attention profiler phase start failed for phase={phase}: {exc}")
            return None
        finally:
            self._attention_profiler_overhead_sec += (
                time.perf_counter() - overhead_start)

    def end_attention_profiler_phase(
        self,
        token: Optional[Dict[str, Any]],
        *,
        branch: Optional[str] = None,
        device: Optional[Any] = None,
        visible_tokens: Optional[int] = None,
        history_tokens: Optional[int] = None,
        current_tokens: Optional[int] = None,
        sink_tokens: Optional[int] = None,
        q_shape: Optional[Any] = None,
        k_shape: Optional[Any] = None,
        v_shape: Optional[Any] = None,
        dtype: Optional[Any] = None,
        used_flash_attention: Optional[Any] = None,
        fallback_reason: Optional[str] = None,
        warning: Optional[str] = None,
    ) -> None:
        if token is None or not self.config.attention_profiler_enabled:
            return

        overhead_start = time.perf_counter()
        try:
            timer_type = str(token.get("timer_type") or "cpu_wall")
            reason = fallback_reason or token.get("fallback_reason")
            elapsed_ms = None
            if timer_type == "cuda_event":
                try:
                    end_event = token.get("end_event")
                    start_event = token.get("start_event")
                    if end_event is None or start_event is None:
                        raise RuntimeError("missing_cuda_event")
                    end_event.record()
                    if self.config.attention_profiler_cuda_synchronize:
                        import torch
                        torch.cuda.synchronize(device=device if device else None)
                    else:
                        end_event.synchronize()
                    elapsed_ms = float(start_event.elapsed_time(end_event))
                except Exception as exc:
                    timer_type = "cpu_wall"
                    reason = reason or f"cuda_event_end_failed:{exc}"
            if elapsed_ms is None:
                elapsed_ms = max(
                    0.0,
                    (time.perf_counter() -
                     float(token["start_perf_counter"])) * 1000.0)

            final_branch = branch if branch is not None else token.get("branch")
            final_device = str(device) if device is not None else token.get("device")
            final_visible_tokens = self._attention_first_not_none(
                visible_tokens, token.get("visible_tokens"))
            final_history_tokens = self._attention_first_not_none(
                history_tokens, token.get("history_tokens"))
            final_current_tokens = self._attention_first_not_none(
                current_tokens, token.get("current_tokens"))
            final_sink_tokens = self._attention_first_not_none(
                sink_tokens, token.get("sink_tokens"))
            final_q_shape = self._attention_shape_text(
                q_shape if q_shape is not None else token.get("q_shape"))
            final_k_shape = self._attention_shape_text(
                k_shape if k_shape is not None else token.get("k_shape"))
            final_v_shape = self._attention_shape_text(
                v_shape if v_shape is not None else token.get("v_shape"))
            final_dtype = str(dtype) if dtype is not None else token.get("dtype")
            final_flash = self._attention_first_not_none(
                used_flash_attention, token.get("used_flash_attention"))
            final_warning = warning if warning is not None else token.get("warning")

            payload = {
                "event": "attention_profile",
                "flowcache_run_id": self._flowcache_run_id,
                "prompt_idx": token.get("prompt_idx"),
                "sample_idx": token.get("sample_idx"),
                "window_idx": token.get("window_idx"),
                "window_index": token.get("window_idx"),
                "step_idx": token.get("step_idx"),
                "timestep": token.get("timestep"),
                "layer_idx": token.get("layer_idx"),
                "branch": final_branch,
                "phase": token.get("phase"),
                "elapsed_ms": elapsed_ms,
                "elapsed_sec": elapsed_ms / 1000.0,
                "timer_type": timer_type,
                "visible_tokens": final_visible_tokens,
                "history_tokens": final_history_tokens,
                "current_tokens": final_current_tokens,
                "sink_tokens": final_sink_tokens,
                "q_shape": final_q_shape,
                "k_shape": final_k_shape,
                "v_shape": final_v_shape,
                "dtype": final_dtype,
                "device": final_device,
                "used_flash_attention": final_flash,
                "fallback_reason": reason,
                "warning": final_warning,
                "timing_scope": token.get("timing_scope"),
                "mode": self.config.attention_profiler_mode,
                "cuda_event": self.config.attention_profiler_cuda_event,
                "cuda_synchronize": (
                    self.config.attention_profiler_cuda_synchronize),
            }
            self._record_attention_profiler_payload(payload)
        except Exception as exc:
            self._warning(
                "attention profiler phase end failed for "
                f"phase={token.get('phase')}: {exc}")
        finally:
            self._attention_profiler_overhead_sec += (
                time.perf_counter() - overhead_start)

    def summarize_attention_profiler(
        self,
        *,
        prompt_count: Optional[int] = None,
        sample_count: Optional[int] = None,
    ) -> None:
        if (
            not self.config.attention_profiler_enabled or
            self._attention_profiler_summary_written
        ):
            return

        overhead_start = time.perf_counter()
        try:
            if not self._attention_profiler_started:
                self.start_attention_profiler_run(
                    prompt_count=prompt_count,
                    sample_count=sample_count,
                )
            if self.config.attention_profiler_cuda_synchronize:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception as exc:
                    self._warning(
                        f"attention profiler cuda synchronize failed: {exc}")
            started = self._attention_profiler_run_context.get(
                "started_perf_counter")
            total_runtime_sec = (
                max(0.0, time.perf_counter() - float(started))
                if started is not None else 0.0
            )
            phase_total_ms = {
                phase: float(stats.get("elapsed_ms", 0.0) or 0.0)
                for phase, stats in sorted(
                    self._attention_profiler_phase_stats.items())
            }
            avg_ms_by_phase = {
                phase: self._attention_avg_ms(stats)
                for phase, stats in sorted(
                    self._attention_profiler_phase_stats.items())
            }
            p50_ms_by_phase = {
                phase: self._attention_percentile(
                    stats.get("values_ms", []), 0.50)
                for phase, stats in sorted(
                    self._attention_profiler_phase_stats.items())
            }
            p90_ms_by_phase = {
                phase: self._attention_percentile(
                    stats.get("values_ms", []), 0.90)
                for phase, stats in sorted(
                    self._attention_profiler_phase_stats.items())
            }
            non_inclusive_total_ms = sum(
                elapsed
                for phase, elapsed in phase_total_ms.items()
                if phase not in {
                    "attention_forward_total",
                    "clean_cache_update_attention",
                }
            )
            phase_time_ratio = {
                phase: (
                    elapsed / non_inclusive_total_ms
                    if non_inclusive_total_ms > 0 else 0.0)
                for phase, elapsed in phase_total_ms.items()
            }
            top_attention_bottleneck_phases = (
                self._attention_top_bottleneck_phases(phase_total_ms))
            peak_allocated_gb, peak_reserved_gb = self._profiler_peak_memory()
            prompt_count = (
                prompt_count if prompt_count is not None else
                self._attention_profiler_run_context.get("prompt_count")
            )
            sample_count = (
                sample_count if sample_count is not None else
                self._attention_profiler_run_context.get("sample_count")
            )
            model_forward_total_sec = float(
                self._profiler_phase_stats.get(
                    "model_forward_total", {}).get("elapsed_sec", 0.0) or 0.0)
            payload = {
                "event": "attention_profiler_summary",
                "flowcache_run_id": self._flowcache_run_id,
                "total_runtime_sec": total_runtime_sec,
                "model_forward_total_sec": model_forward_total_sec,
                "attention_sample_count": len(self._attention_profiler_events),
                "prompt_count": prompt_count,
                "sample_count": sample_count,
                "phase_total_ms": phase_total_ms,
                "phase_time_ratio": phase_time_ratio,
                "avg_ms_by_phase": avg_ms_by_phase,
                "p50_ms_by_phase": p50_ms_by_phase,
                "p90_ms_by_phase": p90_ms_by_phase,
                "layer_breakdown": self._attention_format_breakdown(
                    self._attention_profiler_layer_stats),
                "window_breakdown": self._attention_format_breakdown(
                    self._attention_profiler_window_stats),
                "branch_breakdown": self._attention_format_breakdown(
                    self._attention_profiler_branch_stats),
                "timer_type_counts": dict(sorted(
                    self._attention_profiler_timer_type_counts.items())),
                "used_flash_attention_counts": dict(sorted(
                    self._attention_profiler_flash_counts.items())),
                "warning_count": self._warning_count,
                "fallback_count": self._attention_profiler_fallback_count,
                "max_events": self.config.attention_profiler_max_events,
                "max_events_reached": self._attention_profiler_max_reached,
                "peak_cuda_allocated_gb": peak_allocated_gb,
                "peak_cuda_reserved_gb": peak_reserved_gb,
                "top_attention_bottleneck_phases": (
                    top_attention_bottleneck_phases),
                "next_step_recommendation": (
                    self._attention_next_step_recommendation(
                        phase_total_ms, phase_time_ratio)),
                "timing_note": (
                    "attention_forward_total and clean_cache_update_attention "
                    "are inclusive spans; subphase rows are diagnostic and "
                    "must not be summed with inclusive spans."),
                "config": {
                    "mode": self.config.attention_profiler_mode,
                    "cuda_event": self.config.attention_profiler_cuda_event,
                    "cuda_synchronize": (
                        self.config.attention_profiler_cuda_synchronize),
                    "sample_layers": list(
                        self.config.attention_profiler_sample_layers),
                    "sample_every_n_steps": (
                        self.config.attention_profiler_sample_every_n_steps),
                    "max_events": self.config.attention_profiler_max_events,
                    "record_rope": self.config.attention_profiler_record_rope,
                    "record_qkv": self.config.attention_profiler_record_qkv,
                    "record_cache_assembly": (
                        self.config.attention_profiler_record_cache_assembly),
                    "record_padding": (
                        self.config.attention_profiler_record_padding),
                    "record_kernel": (
                        self.config.attention_profiler_record_kernel),
                    "record_output_proj": (
                        self.config.attention_profiler_record_output_proj),
                    "record_cache_write": (
                        self.config.attention_profiler_record_cache_write),
                    "record_clean_cache_update": (
                        self.config
                        .attention_profiler_record_clean_cache_update),
                },
            }
            self._write_attention_profiler_summary(payload)
            top_text = "|".join(
                f"{item['phase']}:{item['total_ms']:.3f}ms"
                for item in top_attention_bottleneck_phases[:5]
            ) or "none"
            print(
                "[FlowCache][attention_profiler_summary] "
                f"total_runtime_sec={total_runtime_sec:.6f} "
                f"attention_sample_count={len(self._attention_profiler_events)} "
                f"timer_type_counts={payload['timer_type_counts']} "
                f"fallback_count={self._attention_profiler_fallback_count} "
                f"warning_count={self._warning_count} "
                f"top_phases={top_text}"
            )
            self._attention_profiler_summary_written = True
        except Exception as exc:
            self._warning(f"attention profiler summary failed: {exc}")
        finally:
            self._attention_profiler_overhead_sec += (
                time.perf_counter() - overhead_start)

    def record_saved_videos(self, count: int = 1) -> None:
        if not self.config.profiler_enabled:
            return
        self._profiler_saved_videos += int(count)

    def summarize_profiler(
        self,
        *,
        prompt_count: Optional[int] = None,
        sample_count: Optional[int] = None,
        saved_videos: Optional[int] = None,
    ) -> None:
        if not self.config.profiler_enabled or self._profiler_summary_written:
            return

        overhead_start = time.perf_counter()
        try:
            self._profiler_sync()
            phase_time_sec = {
                phase: stats["elapsed_sec"]
                for phase, stats in sorted(self._profiler_phase_stats.items())
            }
            event_count_by_phase = {
                phase: stats["count"]
                for phase, stats in sorted(self._profiler_phase_stats.items())
            }
            total_runtime_sec = phase_time_sec.get("total_inference")
            if total_runtime_sec is None:
                started = self._profiler_run_context.get("started_perf_counter")
                total_runtime_sec = (
                    max(0.0, time.perf_counter() - float(started))
                    if started is not None else
                    sum(phase_time_sec.values())
                )
            divisor = total_runtime_sec if total_runtime_sec and total_runtime_sec > 0 else 1.0
            phase_time_ratio = {
                phase: elapsed / divisor
                for phase, elapsed in phase_time_sec.items()
            }
            sampled_phase_time_sec = {
                phase: elapsed
                for phase, elapsed in phase_time_sec.items()
                if self._profiler_is_sampled_phase(phase)
            }
            sampled_phase_count = {
                phase: event_count_by_phase.get(phase, 0)
                for phase in sampled_phase_time_sec
            }
            sampled_avg_ms_by_phase = {
                phase: (
                    sampled_phase_time_sec[phase] /
                    sampled_phase_count[phase] * 1000.0
                    if sampled_phase_count[phase] else 0.0
                )
                for phase in sampled_phase_time_sec
            }
            top_bottleneck_phases = self._profiler_top_phases(phase_time_sec)
            peak_allocated_gb, peak_reserved_gb = self._profiler_peak_memory()
            prompt_count = (
                prompt_count if prompt_count is not None else
                self._profiler_run_context.get("prompt_count")
            )
            sample_count = (
                sample_count if sample_count is not None else
                self._profiler_run_context.get("sample_count")
            )
            saved_videos = (
                saved_videos if saved_videos is not None else
                self._profiler_saved_videos
            )
            payload = {
                "event": "profiler_summary",
                "flowcache_run_id": self._flowcache_run_id,
                "total_runtime_sec": total_runtime_sec,
                "prompt_count": prompt_count,
                "sample_count": sample_count,
                "saved_videos": saved_videos,
                "phase_time_sec": phase_time_sec,
                "phase_time_ratio": phase_time_ratio,
                "event_count_by_phase": event_count_by_phase,
                "model_forward_total_sec": phase_time_sec.get("model_forward_total", 0.0),
                "denoise_loop_total_sec": phase_time_sec.get("denoise_loop_total", 0.0),
                "vae_decode_total_sec": phase_time_sec.get("vae_decode_total", 0.0),
                "video_save_total_sec": phase_time_sec.get("video_save_total", 0.0),
                "clean_cache_update_total_sec": phase_time_sec.get(
                    "clean_cache_update_total", 0.0),
                "sampled_phase_time_sec": sampled_phase_time_sec,
                "sampled_phase_count": sampled_phase_count,
                "sampled_avg_ms_by_phase": sampled_avg_ms_by_phase,
                "sampled_layer_breakdown": self._profiler_format_breakdown(
                    self._profiler_sampled_layer_stats),
                "sampled_step_breakdown": self._profiler_format_breakdown(
                    self._profiler_sampled_step_stats),
                "warning_count": self._warning_count,
                "profiler_overhead_estimate": self._profiler_overhead_sec,
                "peak_cuda_allocated_gb": peak_allocated_gb,
                "peak_cuda_reserved_gb": peak_reserved_gb,
                "top_bottleneck_phases": top_bottleneck_phases,
                "next_step_recommendation": self._profiler_next_step_recommendation(
                    phase_time_ratio,
                    sampled_phase_time_sec,
                    sampled_avg_ms_by_phase),
                "config": {
                    "mode": self.config.profiler_mode,
                    "cuda_synchronize": self.config.profiler_cuda_synchronize,
                    "fine_enabled": self.config.profiler_fine_enabled,
                    "sample_layers": list(self.config.profiler_sample_layers),
                    "sample_every_n_steps": self.config.profiler_sample_every_n_steps,
                    "record_block_forward": self.config.profiler_record_block_forward,
                    "record_attention": self.config.profiler_record_attention,
                    "record_mlp": self.config.profiler_record_mlp,
                    "record_kv_ops": self.config.profiler_record_kv_ops,
                    "record_clean_cache_update": (
                        self.config.profiler_record_clean_cache_update),
                    "record_decode_save": self.config.profiler_record_decode_save,
                },
                "mode": self.config.profiler_mode,
                "cuda_synchronize": self.config.profiler_cuda_synchronize,
                "device": self._profiler_run_context.get("device"),
                "fallback_reason": None,
            }
            self._write_profiler_summary(payload)
            self._write_profiler_jsonl(payload)

            top_text = "|".join(
                f"{item['phase']}:{item['elapsed_sec']:.6f}"
                for item in top_bottleneck_phases[:5]
            ) or "none"
            print(
                "[FlowCache][profiler_summary] "
                f"total_runtime_sec={total_runtime_sec:.6f} "
                f"prompt_count={prompt_count} "
                f"sample_count={sample_count} "
                f"saved_videos={saved_videos} "
                f"warning_count={self._warning_count} "
                f"profiler_overhead_estimate={self._profiler_overhead_sec:.6f} "
                f"peak_cuda_allocated_gb={peak_allocated_gb if peak_allocated_gb is not None else 'none'} "
                f"peak_cuda_reserved_gb={peak_reserved_gb if peak_reserved_gb is not None else 'none'} "
                f"top_phases={top_text}"
            )
            self._profiler_summary_written = True
        except Exception as exc:
            self._warning(f"profiler summary failed: {exc}")
        finally:
            self._profiler_overhead_sec += time.perf_counter() - overhead_start

    @property
    def metadata_enabled(self) -> bool:
        return self.config.metadata_enabled

    @property
    def should_log(self) -> bool:
        return self.config.enabled and self.config.debug

    @property
    def should_run_output_reuse_dry_run(self) -> bool:
        return (
            self.config.enabled and
            self.config.output_reuse_dry_run_enabled
        )

    @property
    def should_track_metadata(self) -> bool:
        return self.config.enabled and self.config.metadata_enabled

    @property
    def should_track_candidates(self) -> bool:
        return self.config.enabled and self.config.compression_candidate_enabled

    @property
    def should_run_compression_dry_run(self) -> bool:
        return (
            self.config.enabled and
            self.config.kv_compress_enabled and
            self.config.kv_compress_dry_run
        )

    @property
    def should_run_real_compression(self) -> bool:
        return (
            self.config.enabled and
            self.config.kv_compress_enabled and
            self.config.kv_compress_real_enabled and
            not self.config.kv_compress_dry_run
        )

    @property
    def should_track_cache_body_compression(self) -> bool:
        return (
            self.config.enabled and
            self.config.cache_body_compress_enabled
        )

    @property
    def should_run_cache_body_compression(self) -> bool:
        return (
            self.should_track_cache_body_compression and
            self.config.cache_body_compress_real_enabled
        )

    @property
    def should_track_persistent_cache_compression(self) -> bool:
        return (
            self.config.enabled and
            self.config.persistent_cache_compress_enabled
        )

    @property
    def should_track_compacted_kv(self) -> bool:
        return (
            self.config.enabled and
            self.config.compacted_kv_enabled
        )

    @property
    def should_run_compacted_kv(self) -> bool:
        return (
            self.should_track_compacted_kv and
            self.config.compacted_kv_real_enabled
        )

    @property
    def should_run_persistent_cache_compression(self) -> bool:
        return (
            self.should_track_persistent_cache_compression and
            self.config.persistent_cache_compress_real_enabled and
            self.config.persistent_cache_compress_sidecar_enabled
        )

    @property
    def should_run_shadow_compare(self) -> bool:
        return (
            self.should_run_real_compression and
            self.config.kv_compress_shadow_compare and
            len(self.attention_output_diff_records) <
            max(0, self.config.kv_compress_shadow_max_events)
        )

    @property
    def should_log_metadata(self) -> bool:
        return self.should_track_metadata and self.config.debug

    @property
    def should_record_r0_trace(self) -> bool:
        return self.config.enabled and self.config.r0_trace_enabled

    @property
    def should_log_candidates(self) -> bool:
        return self.should_track_candidates and self.config.debug

    def begin_window(
        self,
        *,
        window_index: int,
        start_block: int,
        end_block: int,
        rolling_window_length_blocks: int,
        num_frame_per_block: int,
        denoising_step_list: Any,
        noisy_cache_shape: Optional[Iterable[int]] = None,
        noisy_input_shape: Optional[Iterable[int]] = None,
        current_timestep_shape: Optional[Iterable[int]] = None,
        current_start_frame: Optional[int] = None,
        current_end_frame: Optional[int] = None,
        current_num_frames: Optional[int] = None,
    ) -> None:
        if not self.config.enabled:
            return
        try:
            record = FlowCacheWindowRecord(
                window_index=window_index,
                start_block=start_block,
                end_block=end_block,
                num_frame_per_block=num_frame_per_block,
                rolling_window_length_blocks=rolling_window_length_blocks,
                current_start_frame=current_start_frame,
                current_end_frame=current_end_frame,
                current_num_frames=current_num_frames,
                denoising_step_list=_to_list(denoising_step_list),
                noisy_cache_shape=_shape_to_tuple(noisy_cache_shape),
                noisy_input_shape=_shape_to_tuple(noisy_input_shape),
                current_timestep_shape=_shape_to_tuple(current_timestep_shape),
                begin_timestamp=time.time(),
            )

            if self.should_track_metadata:
                self._active_windows[window_index] = record

            if self.should_log and self.config.log_every_window:
                print(
                    "[FlowCache][window] "
                    f"index={window_index} "
                    f"start_block={start_block} "
                    f"end_block={end_block} "
                    f"num_frame_per_block={num_frame_per_block} "
                    f"rolling_window_length_blocks={rolling_window_length_blocks} "
                    f"current_start_frame={current_start_frame} "
                    f"current_end_frame={current_end_frame} "
                    f"current_num_frames={current_num_frames} "
                    f"denoising_step_list={record.denoising_step_list} "
                    f"noisy_cache_shape={record.noisy_cache_shape} "
                    f"noisy_input_shape={record.noisy_input_shape} "
                    f"current_timestep_shape={record.current_timestep_shape}"
                )
        except Exception as exc:
            self._warning(f"begin_window failed for window_index={window_index}: {exc}")

    def end_window(self, window_index: int) -> None:
        if not self.should_track_metadata:
            return

        try:
            record = self._active_windows.pop(window_index, None)
            if record is None:
                self._warning(f"missing begin_window record for window_index={window_index}")
                return

            record.end_timestamp = time.time()
            record.elapsed_sec = record.end_timestamp - record.begin_timestamp
            self.window_records.append(record)

            if self.should_log_metadata:
                print(
                    "[FlowCache][event] "
                    f"event=window_end "
                    f"window_index={record.window_index} "
                    f"elapsed_sec={record.elapsed_sec:.6f}"
                )
        except Exception as exc:
            self._warning(f"end_window failed for window_index={window_index}: {exc}")

    def log_window(self, **kwargs: Any) -> None:
        self.begin_window(**kwargs)

    def log_event(
        self,
        *,
        event: str,
        window_index: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.should_track_metadata:
            return

        try:
            payload = {
                "event": event,
                "window_index": window_index,
                "timestamp": time.time(),
            }
            if details:
                payload.update(details)
            self.event_records.append(payload)

            if self.should_log_metadata:
                detail_text = " ".join(
                    f"{key}={value}" for key, value in payload.items()
                    if key not in {"event", "timestamp"})
                print(f"[FlowCache][event] event={event} {detail_text}".rstrip())
        except Exception as exc:
            self._warning(f"log_event failed for event={event}: {exc}")

    def record_r0_stage_trace(
        self,
        *,
        prompt_idx: Optional[int],
        window_index: int,
        start_block: int,
        end_block: int,
        global_video_block_id: int,
        local_block_index: int,
        actual_timestep: float,
        local_stage_index: int,
        total_local_stages: int,
        num_frame_per_block: int,
        current_num_frames: int,
        stage_values: Iterable[Any],
    ) -> None:
        """Write R0 scalar stage metadata without mutating inference state."""
        if not self.should_record_r0_trace:
            return

        try:
            payload = {
                "event": "r0_stage_trace",
                "timestamp": time.time(),
                "prompt_idx": _optional_int(prompt_idx),
                "window_index": int(window_index),
                "window_start_block": int(start_block),
                "window_end_block": int(end_block),
                "global_video_block_id": int(global_video_block_id),
                "local_block_index": int(local_block_index),
                "actual_timestep": float(actual_timestep),
                "local_stage_index": int(local_stage_index),
                "total_local_stages": int(total_local_stages),
                "num_frame_per_block": int(num_frame_per_block),
                "current_num_frames": int(current_num_frames),
                "stage_values": [float(item) for item in stage_values],
            }
            self.r0_trace_records.append(payload)
            self._write_r0_jsonl(payload)
        except Exception as exc:
            self._warning(
                "record_r0_stage_trace failed for "
                f"window_index={window_index}, block={global_video_block_id}: {exc}")
            raise

    def record_r0_rng_state(
        self,
        *,
        prompt_idx: Optional[int],
        window_index: int,
        phase: str,
        cpu_rng_state: Any,
        cuda_rng_state: Any,
        cuda_device: Optional[Any],
    ) -> None:
        """Write hashes of RNG state only; do not serialise or change the state."""
        if not (self.should_record_r0_trace and self.config.r0_trace_rng_state):
            return

        try:
            payload = {
                "event": "r0_rng_trace",
                "timestamp": time.time(),
                "prompt_idx": _optional_int(prompt_idx),
                "window_index": int(window_index),
                "phase": str(phase),
                "cpu_rng_sha256": _rng_state_sha256(cpu_rng_state),
                "cuda_rng_sha256": _rng_state_sha256(cuda_rng_state),
                "cuda_device": None if cuda_device is None else str(cuda_device),
            }
            self.r0_trace_records.append(payload)
            self._write_r0_jsonl(payload)
        except Exception as exc:
            self._warning(
                "record_r0_rng_state failed for "
                f"window_index={window_index}, phase={phase}: {exc}")
            raise

    def log_kv_range(
        self,
        kv_cache: Optional[list],
        *,
        event: str,
        window_index: int,
        start_block: int,
        end_block: int,
        rolling_window_length_blocks: int,
        num_frame_per_block: int,
        frame_seq_length: int,
        current_num_frames: Optional[int],
        max_attention_tokens: Optional[int] = None,
    ) -> None:
        if not self.should_track_metadata:
            return

        try:
            if kv_cache is None:
                self._warning(f"kv_cache is None for event={event} window_index={window_index}")
                return

            for layer_idx, cache_item in enumerate(kv_cache):
                record = self._build_kv_range_record(
                    cache_item=cache_item,
                    event=event,
                    window_index=window_index,
                    start_block=start_block,
                    end_block=end_block,
                    rolling_window_length_blocks=rolling_window_length_blocks,
                    num_frame_per_block=num_frame_per_block,
                    frame_seq_length=frame_seq_length,
                    current_num_frames=current_num_frames,
                    max_attention_tokens=max_attention_tokens,
                    layer_idx=layer_idx,
                )
                self.kv_range_records.append(record)
                self._write_jsonl(asdict(record))

                if self.should_log_metadata and self.config.log_kv_ranges:
                    print(
                        "[FlowCache][kv_range] "
                        f"event={record.event} "
                        f"window={record.window_index} "
                        f"layer={record.layer_idx} "
                        f"sink=[{record.sink_start},{record.sink_end}) "
                        f"history=[{record.history_start},{record.history_end}) "
                        f"current=[{record.current_start},{record.current_end}) "
                        f"working_history_tokens={record.working_history_tokens} "
                        f"current_tokens={record.current_tokens} "
                        f"cache_tokens={record.cache_tokens} "
                        f"total={record.total_visible_kv_tokens} "
                        f"global_end_index={record.global_end_index} "
                        f"local_end_index={record.local_end_index}"
                    )
        except Exception as exc:
            self._warning(f"log_kv_range failed for event={event} window_index={window_index}: {exc}")

    def log_kv_cache_summary(self, kv_cache: Optional[list], *, stage: str) -> None:
        if not self.should_log:
            return

        if kv_cache is None:
            print(f"[FlowCache][event] event=kv_cache_summary stage={stage} kv_cache=None")
            return

        entries = []
        for index, cache_item in enumerate(kv_cache):
            global_end = _scalar_to_int(_dict_get(cache_item, "global_end_index"))
            local_end = _scalar_to_int(_dict_get(cache_item, "local_end_index"))
            entries.append(f"{index}:g={global_end},l={local_end}")

        print(
            f"[FlowCache][event] event=kv_cache_summary stage={stage} "
            f"num_layers={len(kv_cache)} end_indices=[{'; '.join(entries)}]"
        )

    def record_attention_parts(
        self,
        *,
        window_index: Optional[int],
        event: str,
        layer_idx: Optional[int],
        attention_branch: str,
        updating_cache: bool,
        block_length: int,
        sink_tokens: int,
        anchor_tokens: int,
        anchor_value_tokens: int,
        working_tokens: int,
        working_value_tokens: int,
        current_tokens: int,
        current_value_tokens: int,
        input_tokens: int,
        input_value_tokens: int,
        current_start: Optional[int],
        cache_start: Optional[int],
        cache_end: Optional[int],
        global_end_index: Optional[int],
        local_end_index: Optional[int],
    ) -> None:
        if not self.should_track_candidates:
            return

        try:
            total_visible_kv_tokens = int(input_tokens)
            record = FlowCacheAttentionParts(
                event="attention_parts",
                timestamp=time.time(),
                source_event=event or attention_branch,
                window_index=window_index,
                layer_idx=layer_idx,
                attention_branch=attention_branch,
                updating_cache=bool(updating_cache),
                block_length=int(block_length),
                sink_tokens=int(sink_tokens),
                anchor_tokens=int(anchor_tokens),
                anchor_value_tokens=int(anchor_value_tokens),
                working_tokens=int(working_tokens),
                working_value_tokens=int(working_value_tokens),
                current_tokens=int(current_tokens),
                current_value_tokens=int(current_value_tokens),
                input_tokens=int(input_tokens),
                input_value_tokens=int(input_value_tokens),
                total_visible_kv_tokens=total_visible_kv_tokens,
                total_kv_tokens=total_visible_kv_tokens,
                current_start=_optional_int(current_start),
                cache_start=_optional_int(cache_start),
                cache_end=_optional_int(cache_end),
                global_end_index=_optional_int(global_end_index),
                local_end_index=_optional_int(local_end_index),
            )
            self.attention_part_records.append(record)
            self._write_jsonl(asdict(record))

            if self.should_log_candidates and self.config.log_attention_parts:
                print(
                    "[FlowCache][attention_parts] "
                    f"source_event={record.source_event} "
                    f"window={record.window_index} "
                    f"layer={record.layer_idx} "
                    f"branch={record.attention_branch} "
                    f"anchor={record.anchor_tokens} "
                    f"working={record.working_tokens} "
                    f"current={record.current_tokens} "
                    f"input={record.input_tokens} "
                    f"sink_tokens={record.sink_tokens} "
                    f"block_length={record.block_length} "
                    f"global_end_index={record.global_end_index} "
                    f"local_end_index={record.local_end_index}"
                )

            self.record_compression_candidate(record)
        except Exception as exc:
            self._warning(
                f"record_attention_parts failed for event={event} "
                f"window_index={window_index} layer_idx={layer_idx}: {exc}")

    def record_compression_candidate(
        self,
        attention_record: FlowCacheAttentionParts,
    ) -> None:
        if not self.should_track_candidates:
            return

        try:
            protected_sink_tokens = max(0, int(attention_record.anchor_tokens))
            protected_current_tokens = max(0, int(attention_record.current_tokens))
            compressible_history_tokens = max(0, int(attention_record.working_tokens))
            total_kv_tokens = max(0, int(attention_record.total_kv_tokens))
            candidate_ratio = (
                compressible_history_tokens / total_kv_tokens
                if total_kv_tokens > 0 else 0.0
            )
            candidate_region_start = protected_sink_tokens
            candidate_region_end = protected_sink_tokens + compressible_history_tokens

            record = FlowCacheCompressionCandidate(
                event="compression_candidate",
                timestamp=time.time(),
                source_event=attention_record.source_event,
                window_index=attention_record.window_index,
                layer_idx=attention_record.layer_idx,
                attention_branch=attention_record.attention_branch,
                block_length=attention_record.block_length,
                sink_tokens=attention_record.sink_tokens,
                anchor_tokens=attention_record.anchor_tokens,
                working_tokens=attention_record.working_tokens,
                current_tokens=attention_record.current_tokens,
                input_tokens=attention_record.input_tokens,
                total_visible_kv_tokens=attention_record.total_visible_kv_tokens,
                total_kv_tokens=total_kv_tokens,
                protected_sink_tokens=protected_sink_tokens,
                protected_current_tokens=protected_current_tokens,
                compressible_history_tokens=compressible_history_tokens,
                candidate_ratio=candidate_ratio,
                candidate_region_start=candidate_region_start,
                candidate_region_end=candidate_region_end,
                global_end_index=attention_record.global_end_index,
                local_end_index=attention_record.local_end_index,
            )
            self.compression_candidate_records.append(record)
            self._write_jsonl(asdict(record))

            if self.should_log_candidates and self.config.log_compression_candidates:
                print(
                    "[FlowCache][compression_candidate] "
                    f"source_event={record.source_event} "
                    f"window={record.window_index} "
                    f"layer={record.layer_idx} "
                    f"protected_sink={record.protected_sink_tokens} "
                    f"protected_current={record.protected_current_tokens} "
                    f"compressible_history={record.compressible_history_tokens} "
                    f"total={record.total_kv_tokens} "
                    f"ratio={record.candidate_ratio:.3f} "
                    f"candidate=[{record.candidate_region_start},{record.candidate_region_end})"
                )
            self.record_compression_dry_run(record)
        except Exception as exc:
            self._warning(
                "record_compression_candidate failed for "
                f"window_index={attention_record.window_index} "
                f"layer_idx={attention_record.layer_idx}: {exc}")

    def build_compression_dry_run_plan(
        self,
        candidate_record: FlowCacheCompressionCandidate,
    ) -> Optional[FlowCacheCompressionDryRun]:
        if not self.should_run_compression_dry_run:
            return None

        try:
            apply_to_branch, skipped_reason = self._resolve_dry_run_branch_policy(
                candidate_record)
            original_total = max(0, int(candidate_record.total_kv_tokens))
            compressible_history = max(
                0, int(candidate_record.compressible_history_tokens))
            protected_sink = max(0, int(candidate_record.protected_sink_tokens))
            protected_current = max(0, int(candidate_record.protected_current_tokens))

            if apply_to_branch and compressible_history <= self.config.kv_compress_min_candidate_tokens:
                apply_to_branch = False
                skipped_reason = "below_min_candidate_tokens"

            projected_history = self._compute_projected_tokens(
                compressible_history_tokens=compressible_history,
                apply_to_branch=apply_to_branch,
            )
            projected_total = protected_sink + projected_history + protected_current
            projected_total = min(projected_total, original_total)
            saved_tokens = max(0, original_total - projected_total)
            saving_ratio = saved_tokens / original_total if original_total else 0.0
            candidate_keep_ratio = (
                projected_history / compressible_history
                if compressible_history else 1.0
            )

            return FlowCacheCompressionDryRun(
                event="compression_dry_run",
                timestamp=time.time(),
                window_index=candidate_record.window_index,
                layer_idx=candidate_record.layer_idx,
                branch=candidate_record.attention_branch,
                attention_event=candidate_record.source_event,
                source_event=candidate_record.source_event,
                protected_sink_tokens=protected_sink,
                protected_current_tokens=protected_current,
                compressible_history_tokens=compressible_history,
                original_total_kv_tokens=original_total,
                projected_history_tokens=projected_history,
                projected_total_kv_tokens=projected_total,
                saved_tokens=saved_tokens,
                saving_ratio=saving_ratio,
                candidate_keep_ratio=candidate_keep_ratio,
                kv_compress_target_ratio=self.config.kv_compress_target_ratio,
                apply_to_branch=apply_to_branch,
                skipped_reason=skipped_reason,
                candidate_region_start=candidate_record.candidate_region_start,
                candidate_region_end=candidate_record.candidate_region_end,
            )
        except Exception as exc:
            self._warning(
                "build_compression_dry_run_plan failed for "
                f"window_index={candidate_record.window_index} "
                f"layer_idx={candidate_record.layer_idx}: {exc}")
            return None

    def record_compression_dry_run(
        self,
        candidate_record: FlowCacheCompressionCandidate,
    ) -> None:
        if not self.should_run_compression_dry_run:
            return

        try:
            record = self.build_compression_dry_run_plan(candidate_record)
            if record is None:
                return
            self.compression_dry_run_records.append(record)
            self._write_jsonl(asdict(record))

            if self.config.debug:
                print(
                    "[FlowCache][compression_dry_run] "
                    f"window={record.window_index} "
                    f"layer={record.layer_idx} "
                    f"branch={record.branch} "
                    f"attention_event={record.attention_event} "
                    f"applied={record.apply_to_branch} "
                    f"original={record.original_total_kv_tokens} "
                    f"projected={record.projected_total_kv_tokens} "
                    f"saved={record.saved_tokens} "
                    f"saving_ratio={record.saving_ratio:.3f} "
                    f"candidate_keep_ratio={record.candidate_keep_ratio:.3f} "
                    f"skipped_reason={record.skipped_reason}"
                )
        except Exception as exc:
            self._warning(
                "record_compression_dry_run failed for "
                f"window_index={candidate_record.window_index} "
                f"layer_idx={candidate_record.layer_idx}: {exc}")

    def should_apply_real_compression(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        num_history_tokens: int,
    ) -> tuple:
        if not self.should_run_real_compression:
            return False, "real_compression_disabled"

        if not _filter_contains(self.config.kv_compress_max_windows, window_index):
            return False, "window_filtered"
        if not _filter_contains(self.config.kv_compress_max_layers, layer_idx):
            return False, "layer_filtered"

        strategy = str(self.config.kv_compress_strategy).lower()
        if strategy not in {"uniform", "linspace"}:
            return False, "unsupported_strategy"

        if branch == "current_only":
            return False, "current_only_protected"

        if source_event == "clean_cache_update_attention" or branch == "clean_cache_update":
            if self.config.kv_compress_apply_to_clean_cache_update:
                return False, "clean_cache_update_not_supported_round4"
            return False, "clean_cache_update_disabled"

        if source_event == "denoise_attention" and branch == "anchor_working_current":
            if not self.config.kv_compress_apply_to_denoise:
                return False, "denoise_disabled"
            if int(num_history_tokens) <= int(self.config.kv_compress_min_candidate_tokens):
                return False, "below_min_candidate_tokens"
            return True, None

        self._warning(
            f"unknown real compression branch policy for source_event={source_event}, branch={branch}")
        return False, "unknown_branch"

    def get_real_compression_keep_tokens(self, num_history_tokens: int) -> int:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return 0
        target_ratio = max(0.0, min(1.0, float(self.config.kv_compress_target_ratio)))
        keep_tokens = int(math.ceil(history_tokens * target_ratio))
        keep_tokens = max(1, keep_tokens)
        return min(history_tokens, keep_tokens)

    def select_history_indices(
        self,
        num_history_tokens: int,
        target_ratio: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> List[int]:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return []

        ratio = (
            float(self.config.kv_compress_target_ratio)
            if target_ratio is None else float(target_ratio)
        )
        ratio = max(0.0, min(1.0, ratio))
        keep_tokens = max(1, int(math.ceil(history_tokens * ratio)))
        keep_tokens = min(history_tokens, keep_tokens)

        chosen_strategy = (strategy or self.config.kv_compress_strategy).lower()
        if chosen_strategy not in {"uniform", "linspace"}:
            self._warning(f"unsupported history index strategy={chosen_strategy}; using uniform")

        if keep_tokens == history_tokens:
            return list(range(history_tokens))
        if keep_tokens == 1:
            return [0]

        return [
            int(index * (history_tokens - 1) // (keep_tokens - 1))
            for index in range(keep_tokens)
        ]

    def build_real_compression_plan(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        protected_sink_tokens: int,
        protected_current_tokens: int,
        original_history_tokens: int,
        original_total_kv_tokens: int,
        candidate_region_start: int,
        candidate_region_end: int,
    ) -> Dict[str, Any]:
        protected_sink = max(0, int(protected_sink_tokens))
        protected_current = max(0, int(protected_current_tokens))
        original_history = max(0, int(original_history_tokens))
        original_total = max(0, int(original_total_kv_tokens))
        apply_real, skipped_reason = self.should_apply_real_compression(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            source_event=source_event,
            num_history_tokens=original_history,
        )
        compressed_history = (
            self.get_real_compression_keep_tokens(original_history)
            if apply_real else original_history
        )
        compressed_total = (
            protected_sink + compressed_history + protected_current
            if apply_real else original_total
        )
        compressed_total = min(original_total, max(0, compressed_total))
        saved_tokens = max(0, original_total - compressed_total)
        saving_ratio = saved_tokens / original_total if original_total else 0.0
        keep_ratio = (
            compressed_history / original_history
            if original_history else 1.0
        )

        return {
            "window_index": window_index,
            "layer_idx": layer_idx,
            "branch": branch,
            "attention_event": source_event,
            "source_event": source_event,
            "applied": apply_real,
            "skipped_reason": skipped_reason,
            "strategy": self.config.kv_compress_strategy,
            "kv_compress_target_ratio": self.config.kv_compress_target_ratio,
            "original_history_tokens": original_history,
            "compressed_history_tokens": compressed_history,
            "protected_sink_tokens": protected_sink,
            "protected_current_tokens": protected_current,
            "original_total_kv_tokens": original_total,
            "compressed_total_kv_tokens": compressed_total,
            "saved_tokens": saved_tokens,
            "saving_ratio": saving_ratio,
            "keep_ratio": keep_ratio,
            "candidate_region_start": int(candidate_region_start),
            "candidate_region_end": int(candidate_region_end),
        }

    def record_real_compression(self, **kwargs: Any) -> None:
        if not self.should_run_real_compression:
            return

        try:
            record = FlowCacheRealCompression(
                event="real_compression",
                timestamp=time.time(),
                window_index=kwargs.get("window_index"),
                layer_idx=kwargs.get("layer_idx"),
                branch=str(kwargs.get("branch")),
                attention_event=str(kwargs.get("attention_event")),
                source_event=str(kwargs.get("source_event", kwargs.get("attention_event"))),
                applied=bool(kwargs.get("applied")),
                skipped_reason=kwargs.get("skipped_reason"),
                strategy=str(kwargs.get("strategy", self.config.kv_compress_strategy)),
                kv_compress_target_ratio=float(
                    kwargs.get("kv_compress_target_ratio",
                               self.config.kv_compress_target_ratio)),
                original_history_tokens=max(
                    0, int(kwargs.get("original_history_tokens", 0))),
                compressed_history_tokens=max(
                    0, int(kwargs.get("compressed_history_tokens", 0))),
                protected_sink_tokens=max(
                    0, int(kwargs.get("protected_sink_tokens", 0))),
                protected_current_tokens=max(
                    0, int(kwargs.get("protected_current_tokens", 0))),
                original_total_kv_tokens=max(
                    0, int(kwargs.get("original_total_kv_tokens", 0))),
                compressed_total_kv_tokens=max(
                    0, int(kwargs.get("compressed_total_kv_tokens", 0))),
                saved_tokens=max(0, int(kwargs.get("saved_tokens", 0))),
                saving_ratio=float(kwargs.get("saving_ratio", 0.0)),
                keep_ratio=float(kwargs.get("keep_ratio", 1.0)),
                candidate_region_start=max(
                    0, int(kwargs.get("candidate_region_start", 0))),
                candidate_region_end=max(
                    0, int(kwargs.get("candidate_region_end", 0))),
                fallback=bool(kwargs.get("fallback", False)),
            )
            self.real_compression_records.append(record)
            self._write_jsonl(asdict(record))

            if self.config.debug:
                print(
                    "[FlowCache][real_compression] "
                    f"window={record.window_index} "
                    f"layer={record.layer_idx} "
                    f"branch={record.branch} "
                    f"applied={record.applied} "
                    f"history={record.original_history_tokens}->{record.compressed_history_tokens} "
                    f"total={record.original_total_kv_tokens}->{record.compressed_total_kv_tokens} "
                    f"saved={record.saved_tokens} "
                    f"ratio={record.saving_ratio:.3f} "
                    f"strategy={record.strategy} "
                    f"fallback={record.fallback} "
                    f"skipped_reason={record.skipped_reason}"
                )
        except Exception as exc:
            self._warning(f"record_real_compression failed: {exc}")

    def should_apply_cache_body_compression(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        num_history_tokens: int,
    ) -> tuple:
        if not self.should_track_cache_body_compression:
            return False, "cache_body_compression_disabled"

        if not self.config.cache_body_compress_real_enabled:
            return False, "cache_body_real_disabled"
        if not self.config.cache_body_compress_protect_sink:
            return False, "sink_protection_required"
        if not self.config.cache_body_compress_protect_current:
            return False, "current_protection_required"
        if not _filter_contains(self.config.cache_body_compress_max_windows, window_index):
            return False, "window_filtered"
        if not _filter_contains(self.config.cache_body_compress_max_layers, layer_idx):
            return False, "layer_filtered"

        strategy = str(self.config.cache_body_compress_strategy).lower()
        if strategy not in {"uniform", "linspace"}:
            return False, "unsupported_strategy"

        if branch == "current_only":
            return False, "current_only_protected"

        if source_event == "clean_cache_update_attention" or branch == "clean_cache_update":
            if self.config.cache_body_compress_apply_to_clean_cache_update:
                if int(num_history_tokens) <= int(self.config.cache_body_compress_min_candidate_tokens):
                    return False, "below_min_candidate_tokens"
                return True, None
            return False, "clean_cache_update_disabled"

        if source_event == "denoise_attention" and branch == "anchor_working_current":
            if not self.config.cache_body_compress_apply_to_denoise:
                return False, "denoise_disabled"
            if int(num_history_tokens) <= int(self.config.cache_body_compress_min_candidate_tokens):
                return False, "below_min_candidate_tokens"
            return True, None

        self._warning(
            "unknown cache-body branch policy for "
            f"source_event={source_event}, branch={branch}")
        return False, "unknown_branch"

    def get_cache_body_keep_tokens(self, num_history_tokens: int) -> int:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return 0
        target_ratio = max(
            0.0, min(1.0, float(self.config.cache_body_compress_target_ratio)))
        keep_tokens = int(math.ceil(history_tokens * target_ratio))
        keep_tokens = max(1, keep_tokens)
        return min(history_tokens, keep_tokens)

    def select_cache_body_history_indices(
        self,
        num_history_tokens: int,
        target_ratio: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> List[int]:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return []

        ratio = (
            float(self.config.cache_body_compress_target_ratio)
            if target_ratio is None else float(target_ratio)
        )
        ratio = max(0.0, min(1.0, ratio))
        keep_tokens = max(1, int(math.ceil(history_tokens * ratio)))
        keep_tokens = min(history_tokens, keep_tokens)

        chosen_strategy = (strategy or self.config.cache_body_compress_strategy).lower()
        if chosen_strategy not in {"uniform", "linspace"}:
            self._warning(
                f"unsupported cache-body index strategy={chosen_strategy}; using uniform")

        if keep_tokens == history_tokens:
            return list(range(history_tokens))
        if keep_tokens == 1:
            return [0]

        return [
            int(index * (history_tokens - 1) // (keep_tokens - 1))
            for index in range(keep_tokens)
        ]

    def build_cache_body_compression_plan(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        protected_sink_tokens: int,
        protected_current_tokens: int,
        original_history_tokens: int,
        original_visible_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        if not self.should_track_cache_body_compression:
            return None

        protected_sink = max(0, int(protected_sink_tokens))
        protected_current = max(0, int(protected_current_tokens))
        original_history = max(0, int(original_history_tokens))
        original_visible = max(0, int(original_visible_tokens))
        apply_body, skipped_reason = self.should_apply_cache_body_compression(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            source_event=source_event,
            num_history_tokens=original_history,
        )
        compressed_history = (
            self.get_cache_body_keep_tokens(original_history)
            if apply_body else original_history
        )
        compressed_visible = (
            protected_sink + compressed_history + protected_current
            if apply_body else original_visible
        )
        compressed_visible = min(original_visible, max(0, compressed_visible))
        saved_visible = max(0, original_visible - compressed_visible)
        visible_saving_ratio = (
            saved_visible / original_visible if original_visible else 0.0
        )
        keep_ratio = (
            compressed_history / original_history
            if original_history else 1.0
        )

        return {
            "window_index": window_index,
            "layer_idx": layer_idx,
            "branch": branch,
            "attention_event": source_event,
            "source_event": source_event,
            "applied": apply_body,
            "skipped_reason": skipped_reason,
            "original_history_tokens": original_history,
            "compressed_history_tokens": compressed_history,
            "protected_sink_tokens": protected_sink,
            "protected_current_tokens": protected_current,
            "original_visible_tokens": original_visible,
            "compressed_visible_tokens": compressed_visible,
            "saved_visible_tokens": saved_visible,
            "visible_saving_ratio": visible_saving_ratio,
            "strategy": self.config.cache_body_compress_strategy,
            "cache_body_compress_target_ratio": (
                self.config.cache_body_compress_target_ratio
            ),
            "keep_ratio": keep_ratio,
            "mode": "logical_read_path",
        }

    def record_cache_body_compression(self, **kwargs: Any) -> None:
        if not self.should_track_cache_body_compression:
            return

        try:
            record = FlowCacheCacheBodyCompression(
                event="cache_body_compression",
                timestamp=time.time(),
                window_index=kwargs.get("window_index"),
                layer_idx=kwargs.get("layer_idx"),
                branch=str(kwargs.get("branch")),
                attention_event=str(kwargs.get("attention_event")),
                source_event=str(kwargs.get("source_event", kwargs.get("attention_event"))),
                applied=bool(kwargs.get("applied")),
                skipped_reason=kwargs.get("skipped_reason"),
                original_history_tokens=max(
                    0, int(kwargs.get("original_history_tokens", 0))),
                compressed_history_tokens=max(
                    0, int(kwargs.get("compressed_history_tokens", 0))),
                protected_sink_tokens=max(
                    0, int(kwargs.get("protected_sink_tokens", 0))),
                protected_current_tokens=max(
                    0, int(kwargs.get("protected_current_tokens", 0))),
                original_visible_tokens=max(
                    0, int(kwargs.get("original_visible_tokens", 0))),
                compressed_visible_tokens=max(
                    0, int(kwargs.get("compressed_visible_tokens", 0))),
                saved_visible_tokens=max(
                    0, int(kwargs.get("saved_visible_tokens", 0))),
                visible_saving_ratio=float(
                    kwargs.get("visible_saving_ratio", 0.0)),
                strategy=str(
                    kwargs.get("strategy",
                               self.config.cache_body_compress_strategy)),
                cache_body_compress_target_ratio=float(
                    kwargs.get("cache_body_compress_target_ratio",
                               self.config.cache_body_compress_target_ratio)),
                keep_ratio=float(kwargs.get("keep_ratio", 1.0)),
                mode=str(kwargs.get("mode", "logical_read_path")),
                fallback=bool(kwargs.get("fallback", False)),
            )
            if self.config.cache_body_compress_debug_verify:
                self._verify_cache_body_record(record)
            self.cache_body_compression_records.append(record)
            self._write_jsonl(asdict(record))

            print(
                "[FlowCache][cache_body_compression] "
                f"window={record.window_index} "
                f"layer={record.layer_idx} "
                f"branch={record.branch} "
                f"applied={record.applied} "
                f"history={record.original_history_tokens}->{record.compressed_history_tokens} "
                f"sink={record.protected_sink_tokens} "
                f"current={record.protected_current_tokens} "
                f"visible={record.original_visible_tokens}->{record.compressed_visible_tokens} "
                f"saved_visible={record.saved_visible_tokens} "
                f"visible_saving_ratio={record.visible_saving_ratio:.4f} "
                f"strategy={record.strategy} "
                f"keep_ratio={record.keep_ratio:.4f} "
                f"mode={record.mode} "
                f"fallback={record.fallback} "
                f"skipped_reason={record.skipped_reason}"
            )
        except Exception as exc:
            self._warning(f"record_cache_body_compression failed: {exc}")

    def should_apply_persistent_cache_compression(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        num_history_tokens: int,
    ) -> tuple:
        if not self.should_track_persistent_cache_compression:
            return False, "persistent_cache_compression_disabled"

        if not self.config.persistent_cache_compress_real_enabled:
            return False, "persistent_cache_real_disabled"
        if not self.config.persistent_cache_compress_sidecar_enabled:
            return False, "persistent_cache_sidecar_disabled"
        if not self.config.persistent_cache_compress_protect_sink:
            return False, "sink_protection_required"
        if not self.config.persistent_cache_compress_protect_current:
            return False, "current_protection_required"
        if not _filter_contains(
                self.config.persistent_cache_compress_max_windows,
                window_index):
            return False, "window_filtered"
        if not _filter_contains(
                self.config.persistent_cache_compress_max_layers,
                layer_idx):
            return False, "layer_filtered"

        strategy = str(self.config.persistent_cache_compress_strategy).lower()
        if strategy not in {"uniform", "linspace"}:
            return False, "unsupported_strategy"

        if branch == "current_only":
            return False, "current_only_protected"

        if source_event == "clean_cache_update_attention" or branch == "clean_cache_update":
            if self.config.persistent_cache_compress_apply_to_clean_cache_update:
                if int(num_history_tokens) <= int(
                        self.config.persistent_cache_compress_min_candidate_tokens):
                    return False, "below_min_candidate_tokens"
                return True, None
            return False, "clean_cache_update_disabled"

        if source_event == "denoise_attention" and branch == "anchor_working_current":
            if not self.config.persistent_cache_compress_apply_to_denoise:
                return False, "denoise_disabled"
            if int(num_history_tokens) <= int(
                    self.config.persistent_cache_compress_min_candidate_tokens):
                return False, "below_min_candidate_tokens"
            return True, None

        self._warning(
            "unknown persistent-cache branch policy for "
            f"source_event={source_event}, branch={branch}")
        return False, "unknown_branch"

    def get_persistent_cache_keep_tokens(self, num_history_tokens: int) -> int:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return 0
        target_ratio = max(
            0.0,
            min(1.0, float(self.config.persistent_cache_compress_target_ratio)),
        )
        keep_tokens = int(math.ceil(history_tokens * target_ratio))
        keep_tokens = max(1, keep_tokens)
        return min(history_tokens, keep_tokens)

    def select_persistent_cache_history_indices(
        self,
        num_history_tokens: int,
        target_ratio: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> List[int]:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return []

        ratio = (
            float(self.config.persistent_cache_compress_target_ratio)
            if target_ratio is None else float(target_ratio)
        )
        ratio = max(0.0, min(1.0, ratio))
        keep_tokens = max(1, int(math.ceil(history_tokens * ratio)))
        keep_tokens = min(history_tokens, keep_tokens)

        chosen_strategy = (
            strategy or self.config.persistent_cache_compress_strategy
        ).lower()
        if chosen_strategy not in {"uniform", "linspace"}:
            self._warning(
                "unsupported persistent-cache index "
                f"strategy={chosen_strategy}; using uniform")

        if keep_tokens == history_tokens:
            return list(range(history_tokens))
        if keep_tokens == 1:
            return [0]

        return [
            int(index * (history_tokens - 1) // (keep_tokens - 1))
            for index in range(keep_tokens)
        ]

    def build_persistent_cache_compression_plan(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        protected_sink_tokens: int,
        protected_current_tokens: int,
        original_history_tokens: int,
        original_visible_tokens: int,
        source_start: Optional[int],
        source_end: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not self.should_track_persistent_cache_compression:
            return None

        protected_sink = max(0, int(protected_sink_tokens))
        protected_current = max(0, int(protected_current_tokens))
        original_history = max(0, int(original_history_tokens))
        original_visible = max(0, int(original_visible_tokens))
        apply_persistent, skipped_reason = (
            self.should_apply_persistent_cache_compression(
                window_index=window_index,
                layer_idx=layer_idx,
                branch=branch,
                source_event=source_event,
                num_history_tokens=original_history,
            )
        )
        compressed_history = (
            self.get_persistent_cache_keep_tokens(original_history)
            if apply_persistent else original_history
        )
        compressed_visible = (
            protected_sink + compressed_history + protected_current
            if apply_persistent else original_visible
        )
        compressed_visible = min(original_visible, max(0, compressed_visible))
        saved_visible = max(0, original_visible - compressed_visible)
        visible_saving_ratio = (
            saved_visible / original_visible if original_visible else 0.0
        )
        keep_ratio = (
            compressed_history / original_history
            if original_history else 1.0
        )

        return {
            "window_index": window_index,
            "layer_idx": layer_idx,
            "branch": branch,
            "attention_event": source_event,
            "source_event": source_event,
            "applied": apply_persistent,
            "skipped_reason": skipped_reason,
            "mode": "sidecar",
            "original_history_tokens": original_history,
            "compressed_history_tokens": compressed_history,
            "protected_sink_tokens": protected_sink,
            "protected_current_tokens": protected_current,
            "original_visible_tokens": original_visible,
            "compressed_visible_tokens": compressed_visible,
            "saved_visible_tokens": saved_visible,
            "visible_saving_ratio": visible_saving_ratio,
            "strategy": self.config.persistent_cache_compress_strategy,
            "target_ratio": self.config.persistent_cache_compress_target_ratio,
            "keep_ratio": keep_ratio,
            "sidecar_created": False,
            "sidecar_reused": False,
            "sidecar_invalidated": False,
            "sidecar_fallback": False,
            "sidecar_valid": False,
            "source_start": (
                None if source_start is None else max(0, int(source_start))
            ),
            "source_end": (
                None if source_end is None else max(0, int(source_end))
            ),
        }

    def record_persistent_cache_compression(self, **kwargs: Any) -> None:
        if not self.should_track_persistent_cache_compression:
            return

        try:
            record = FlowCachePersistentCacheCompression(
                event="persistent_cache_compression",
                timestamp=time.time(),
                window_index=kwargs.get("window_index"),
                layer_idx=kwargs.get("layer_idx"),
                branch=str(kwargs.get("branch")),
                attention_event=str(kwargs.get("attention_event")),
                source_event=str(kwargs.get("source_event", kwargs.get("attention_event"))),
                applied=bool(kwargs.get("applied")),
                skipped_reason=kwargs.get("skipped_reason"),
                mode=str(kwargs.get("mode", "sidecar")),
                original_history_tokens=max(
                    0, int(kwargs.get("original_history_tokens", 0))),
                compressed_history_tokens=max(
                    0, int(kwargs.get("compressed_history_tokens", 0))),
                protected_sink_tokens=max(
                    0, int(kwargs.get("protected_sink_tokens", 0))),
                protected_current_tokens=max(
                    0, int(kwargs.get("protected_current_tokens", 0))),
                original_visible_tokens=max(
                    0, int(kwargs.get("original_visible_tokens", 0))),
                compressed_visible_tokens=max(
                    0, int(kwargs.get("compressed_visible_tokens", 0))),
                saved_visible_tokens=max(
                    0, int(kwargs.get("saved_visible_tokens", 0))),
                visible_saving_ratio=float(
                    kwargs.get("visible_saving_ratio", 0.0)),
                strategy=str(
                    kwargs.get("strategy",
                               self.config.persistent_cache_compress_strategy)),
                target_ratio=float(
                    kwargs.get("target_ratio",
                               self.config.persistent_cache_compress_target_ratio)),
                keep_ratio=float(kwargs.get("keep_ratio", 1.0)),
                sidecar_created=bool(kwargs.get("sidecar_created", False)),
                sidecar_reused=bool(kwargs.get("sidecar_reused", False)),
                sidecar_invalidated=bool(kwargs.get("sidecar_invalidated", False)),
                sidecar_fallback=bool(kwargs.get("sidecar_fallback", False)),
                sidecar_valid=bool(kwargs.get("sidecar_valid", False)),
                source_start=_optional_int(kwargs.get("source_start")),
                source_end=_optional_int(kwargs.get("source_end")),
                fallback=bool(kwargs.get("fallback", False)),
            )
            if self.config.persistent_cache_compress_debug_verify:
                self._verify_persistent_cache_record(record)
            self.persistent_cache_compression_records.append(record)
            self._write_jsonl(asdict(record))

            print(
                "[FlowCache][persistent_cache_compression] "
                f"window={record.window_index} "
                f"layer={record.layer_idx} "
                f"branch={record.branch} "
                f"applied={record.applied} "
                f"history={record.original_history_tokens}->{record.compressed_history_tokens} "
                f"sink={record.protected_sink_tokens} "
                f"current={record.protected_current_tokens} "
                f"visible={record.original_visible_tokens}->{record.compressed_visible_tokens} "
                f"saved_visible={record.saved_visible_tokens} "
                f"visible_saving_ratio={record.visible_saving_ratio:.4f} "
                f"strategy={record.strategy} "
                f"target_ratio={record.target_ratio:.4f} "
                f"keep_ratio={record.keep_ratio:.4f} "
                f"mode={record.mode} "
                f"sidecar_created={record.sidecar_created} "
                f"sidecar_reused={record.sidecar_reused} "
                f"sidecar_invalidated={record.sidecar_invalidated} "
                f"sidecar_fallback={record.sidecar_fallback} "
                f"sidecar_valid={record.sidecar_valid} "
                f"source={record.source_start}:{record.source_end} "
                f"fallback={record.fallback} "
                f"skipped_reason={record.skipped_reason}"
            )
        except Exception as exc:
            self._warning(f"record_persistent_cache_compression failed: {exc}")

    def record_persistent_cache_sidecar(
        self,
        *,
        action: str,
        window_index: Optional[int],
        layer_idx: Optional[int],
        source_start: Optional[int],
        source_end: Optional[int],
        compressed_tokens: int,
        reason: Optional[str],
    ) -> None:
        if not self.should_track_persistent_cache_compression:
            return

        try:
            action_name = str(action)
            record = FlowCachePersistentCacheSidecar(
                event="persistent_cache_sidecar",
                timestamp=time.time(),
                action=action_name,
                window_index=window_index,
                layer_idx=layer_idx,
                source_start=_optional_int(source_start),
                source_end=_optional_int(source_end),
                compressed_tokens=max(0, int(compressed_tokens)),
                reason=reason,
            )
            self.persistent_cache_sidecar_records.append(record)
            self._persistent_sidecar_action_counts[action_name] = (
                self._persistent_sidecar_action_counts.get(action_name, 0) + 1
            )
            self._write_jsonl(asdict(record))

            print(
                "[FlowCache][persistent_cache_sidecar] "
                f"action={record.action} "
                f"window={record.window_index} "
                f"layer={record.layer_idx} "
                f"source={record.source_start}:{record.source_end} "
                f"compressed_tokens={record.compressed_tokens} "
                f"reason={record.reason}"
            )
        except Exception as exc:
            self._warning(f"record_persistent_cache_sidecar failed: {exc}")

    def validate_persistent_cache_sidecar_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        layer_idx: Optional[int],
        source_start: Optional[int],
        source_end: Optional[int],
        target_ratio: Optional[float] = None,
        strategy: Optional[str] = None,
        original_history_tokens: Optional[int] = None,
        compressed_tokens: Optional[int] = None,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
        global_end_index: Optional[int] = None,
        local_end_index: Optional[int] = None,
    ) -> tuple:
        if not metadata:
            return False, "missing_metadata"
        if not bool(metadata.get("valid", False)):
            return False, "sidecar_invalid_flag"

        expected_source_start = _optional_int(source_start)
        expected_source_end = _optional_int(source_end)
        if _optional_int(metadata.get("layer_idx")) != _optional_int(layer_idx):
            return False, "layer_mismatch"
        if _optional_int(metadata.get("source_start")) != expected_source_start:
            return False, "source_start_mismatch"
        if _optional_int(metadata.get("source_end")) != expected_source_end:
            return False, "source_end_mismatch"

        expected_strategy = (
            strategy or self.config.persistent_cache_compress_strategy
        )
        if str(metadata.get("strategy", "")).lower() != str(expected_strategy).lower():
            return False, "strategy_mismatch"

        expected_ratio = (
            self.config.persistent_cache_compress_target_ratio
            if target_ratio is None else float(target_ratio)
        )
        actual_ratio = float(metadata.get("target_ratio", -1.0))
        if abs(actual_ratio - float(expected_ratio)) > 1e-12:
            return False, "target_ratio_mismatch"

        if global_end_index is not None and (
                _optional_int(metadata.get("global_end_index")) !=
                _optional_int(global_end_index)):
            return False, "global_end_index_mismatch"
        if local_end_index is not None and (
                _optional_int(metadata.get("local_end_index")) !=
                _optional_int(local_end_index)):
            return False, "local_end_index_mismatch"

        actual_compressed = _optional_int(metadata.get("compressed_tokens"))
        if compressed_tokens is not None and (
                actual_compressed != _optional_int(compressed_tokens)):
            return False, "compressed_tokens_mismatch"
        if original_history_tokens is not None and actual_compressed is not None:
            if actual_compressed > int(original_history_tokens):
                return False, "compressed_history_exceeds_original"

        if device is not None and metadata.get("device") is not None:
            if str(metadata.get("device")) != str(device):
                return False, "device_mismatch"
        if dtype is not None and metadata.get("dtype") is not None:
            if str(metadata.get("dtype")) != str(dtype):
                return False, "dtype_mismatch"

        return True, None

    def should_invalidate_persistent_cache_sidecar_for_write(
        self,
        metadata: Dict[str, Any],
        *,
        write_start: Optional[int],
        write_end: Optional[int],
        eviction_happened: bool = False,
    ) -> tuple:
        if not metadata or not bool(metadata.get("valid", False)):
            return False, "sidecar_invalid_or_missing"

        if eviction_happened:
            return True, "cache_write_after_eviction"

        source_start = _optional_int(metadata.get("source_start"))
        source_end = _optional_int(metadata.get("source_end"))
        write_start = _optional_int(write_start)
        write_end = _optional_int(write_end)

        if source_start is None or source_end is None:
            return True, "missing_sidecar_source_range"
        if write_start is None or write_end is None:
            return True, "missing_cache_write_range"
        if write_end <= write_start:
            return False, "empty_cache_write_range"
        if source_end <= source_start:
            return True, "invalid_sidecar_source_range"

        if write_end <= source_start or source_end <= write_start:
            return False, "cache_write_disjoint_from_sidecar_source"
        return True, "cache_write_overlaps_sidecar_source"

    def should_apply_compacted_kv(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        num_history_tokens: int,
    ) -> tuple:
        if not self.should_track_compacted_kv:
            return False, "compacted_kv_disabled"
        if not self.config.compacted_kv_real_enabled:
            return False, "compacted_kv_real_disabled"
        if not self.config.compacted_kv_protect_sink:
            return False, "sink_protection_required"
        if not self.config.compacted_kv_protect_current:
            return False, "current_protection_required"

        max_events = self.config.compacted_kv_max_events
        if max_events is not None:
            applied_count = sum(1 for record in self.compacted_kv_records
                                if record.applied)
            if applied_count >= int(max_events):
                return False, "max_events_reached"

        strategy = str(self.config.compacted_kv_strategy).lower()
        if strategy not in {"uniform", "linspace"}:
            return False, "unsupported_strategy"

        if branch == "current_only":
            return False, "current_only_protected"

        if source_event == "clean_cache_update_attention" or branch == "clean_cache_update":
            if self.config.compacted_kv_apply_to_clean_cache_update:
                if int(num_history_tokens) <= int(
                        self.config.compacted_kv_min_history_tokens):
                    return False, "below_min_history_tokens"
                return True, None
            return False, "clean_cache_update_disabled"

        if source_event == "denoise_attention" and branch == "anchor_working_current":
            if not self.config.compacted_kv_apply_to_denoise:
                return False, "denoise_disabled"
            if int(num_history_tokens) <= int(
                    self.config.compacted_kv_min_history_tokens):
                return False, "below_min_history_tokens"
            return True, None

        self._warning(
            "unknown compacted-kv branch policy for "
            f"source_event={source_event}, branch={branch}")
        return False, "unknown_branch"

    def get_compacted_kv_keep_tokens(self, num_history_tokens: int) -> int:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return 0
        target_ratio = max(
            0.0,
            min(1.0, float(self.config.compacted_kv_target_ratio)),
        )
        keep_tokens = int(math.ceil(history_tokens * target_ratio))
        keep_tokens = max(1, keep_tokens)
        return min(history_tokens, keep_tokens)

    def select_compacted_kv_history_indices(
        self,
        num_history_tokens: int,
        target_ratio: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> List[int]:
        history_tokens = max(0, int(num_history_tokens))
        if history_tokens == 0:
            return []

        ratio = (
            float(self.config.compacted_kv_target_ratio)
            if target_ratio is None else float(target_ratio)
        )
        ratio = max(0.0, min(1.0, ratio))
        keep_tokens = max(1, int(math.ceil(history_tokens * ratio)))
        keep_tokens = min(history_tokens, keep_tokens)

        chosen_strategy = (strategy or self.config.compacted_kv_strategy).lower()
        if chosen_strategy not in {"uniform", "linspace"}:
            self._warning(
                f"unsupported compacted-kv index strategy={chosen_strategy}; using uniform")

        if keep_tokens == history_tokens:
            return list(range(history_tokens))
        if keep_tokens == 1:
            return [0]

        return [
            int(index * (history_tokens - 1) // (keep_tokens - 1))
            for index in range(keep_tokens)
        ]

    def build_compacted_kv_plan(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        source_event: str,
        protected_sink_tokens: int,
        protected_current_tokens: int,
        original_history_tokens: int,
        original_visible_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        if not self.should_track_compacted_kv:
            return None

        protected_sink = max(0, int(protected_sink_tokens))
        protected_current = max(0, int(protected_current_tokens))
        original_history = max(0, int(original_history_tokens))
        original_visible = max(0, int(original_visible_tokens))
        apply_compacted, skipped_reason = self.should_apply_compacted_kv(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            source_event=source_event,
            num_history_tokens=original_history,
        )
        compacted_history = (
            self.get_compacted_kv_keep_tokens(original_history)
            if apply_compacted else original_history
        )
        compacted_visible = (
            protected_sink + compacted_history + protected_current
            if apply_compacted else original_visible
        )
        compacted_visible = min(original_visible, max(0, compacted_visible))
        saved_visible = max(0, original_visible - compacted_visible)
        visible_saving_ratio = (
            saved_visible / original_visible if original_visible else 0.0
        )
        keep_ratio = (
            compacted_history / original_history
            if original_history else 1.0
        )

        return {
            "prompt_idx": None,
            "sample_idx": None,
            "window_index": window_index,
            "layer_idx": layer_idx,
            "branch": branch,
            "attention_event": source_event,
            "source_event": source_event,
            "applied": apply_compacted,
            "skipped_reason": skipped_reason,
            "fallback": False,
            "fallback_reason": None,
            "original_history_tokens": original_history,
            "compacted_history_tokens": compacted_history,
            "protected_sink_tokens": protected_sink,
            "protected_current_tokens": protected_current,
            "original_visible_tokens": original_visible,
            "compacted_visible_tokens": compacted_visible,
            "saved_visible_tokens": saved_visible,
            "visible_saving_ratio": visible_saving_ratio,
            "target_ratio": self.config.compacted_kv_target_ratio,
            "strategy": self.config.compacted_kv_strategy,
            "keep_ratio": keep_ratio,
            "dtype": None,
            "device": None,
            "elapsed_ms": None,
        }

    def record_compacted_kv(self, **kwargs: Any) -> None:
        if not self.should_track_compacted_kv:
            return

        try:
            record = FlowCacheCompactedKV(
                event="compacted_kv",
                timestamp=time.time(),
                prompt_idx=_optional_int(kwargs.get("prompt_idx")),
                sample_idx=_optional_int(kwargs.get("sample_idx")),
                window_index=kwargs.get("window_index"),
                layer_idx=kwargs.get("layer_idx"),
                branch=str(kwargs.get("branch")),
                attention_event=str(kwargs.get("attention_event")),
                source_event=str(kwargs.get("source_event", kwargs.get("attention_event"))),
                applied=bool(kwargs.get("applied")),
                skipped_reason=kwargs.get("skipped_reason"),
                fallback=bool(kwargs.get("fallback", False)),
                fallback_reason=kwargs.get("fallback_reason"),
                original_history_tokens=max(
                    0, int(kwargs.get("original_history_tokens", 0))),
                compacted_history_tokens=max(
                    0, int(kwargs.get("compacted_history_tokens", 0))),
                protected_sink_tokens=max(
                    0, int(kwargs.get("protected_sink_tokens", 0))),
                protected_current_tokens=max(
                    0, int(kwargs.get("protected_current_tokens", 0))),
                original_visible_tokens=max(
                    0, int(kwargs.get("original_visible_tokens", 0))),
                compacted_visible_tokens=max(
                    0, int(kwargs.get("compacted_visible_tokens", 0))),
                saved_visible_tokens=max(
                    0, int(kwargs.get("saved_visible_tokens", 0))),
                visible_saving_ratio=float(
                    kwargs.get("visible_saving_ratio", 0.0)),
                target_ratio=float(
                    kwargs.get("target_ratio",
                               self.config.compacted_kv_target_ratio)),
                strategy=str(
                    kwargs.get("strategy",
                               self.config.compacted_kv_strategy)),
                keep_ratio=float(kwargs.get("keep_ratio", 1.0)),
                dtype=kwargs.get("dtype"),
                device=kwargs.get("device"),
                elapsed_ms=(
                    None if kwargs.get("elapsed_ms") is None
                    else float(kwargs.get("elapsed_ms"))
                ),
            )
            if self.config.compacted_kv_debug_verify:
                self._verify_compacted_kv_record(record)
            self.compacted_kv_records.append(record)
            self._write_jsonl(asdict(record))

            if self.config.compacted_kv_log_events:
                print(
                    "[FlowCache][compacted_kv] "
                    f"window={record.window_index} "
                    f"layer={record.layer_idx} "
                    f"branch={record.branch} "
                    f"applied={record.applied} "
                    f"history={record.original_history_tokens}->{record.compacted_history_tokens} "
                    f"sink={record.protected_sink_tokens} "
                    f"current={record.protected_current_tokens} "
                    f"visible={record.original_visible_tokens}->{record.compacted_visible_tokens} "
                    f"saved_visible={record.saved_visible_tokens} "
                    f"visible_saving_ratio={record.visible_saving_ratio:.4f} "
                    f"strategy={record.strategy} "
                    f"target_ratio={record.target_ratio:.4f} "
                    f"keep_ratio={record.keep_ratio:.4f} "
                    f"fallback={record.fallback} "
                    f"fallback_reason={record.fallback_reason} "
                    f"skipped_reason={record.skipped_reason} "
                    f"elapsed_ms={_format_metric_value(record.elapsed_ms)}"
                )
        except Exception as exc:
            self._warning(f"record_compacted_kv failed: {exc}")

    def record_attention_output_diff(
        self,
        *,
        window_index: Optional[int],
        layer_idx: Optional[int],
        branch: str,
        attention_event: str,
        source_event: str,
        mean_abs_diff: float,
        max_abs_diff: float,
        relative_l1: float,
        cosine_similarity: Optional[float],
    ) -> None:
        if not self.should_run_real_compression:
            return

        try:
            record = FlowCacheAttentionOutputDiff(
                event="attention_output_diff",
                timestamp=time.time(),
                window_index=window_index,
                layer_idx=layer_idx,
                branch=branch,
                attention_event=attention_event,
                source_event=source_event,
                mean_abs_diff=float(mean_abs_diff),
                max_abs_diff=float(max_abs_diff),
                relative_l1=float(relative_l1),
                cosine_similarity=(
                    None if cosine_similarity is None else float(cosine_similarity)
                ),
            )
            self.attention_output_diff_records.append(record)
            self._write_jsonl(asdict(record))

            if self.config.debug:
                print(
                    "[FlowCache][attention_output_diff] "
                    f"window={record.window_index} "
                    f"layer={record.layer_idx} "
                    f"branch={record.branch} "
                    f"mean_abs_diff={record.mean_abs_diff:.6e} "
                    f"max_abs_diff={record.max_abs_diff:.6e} "
                    f"relative_l1={record.relative_l1:.6e} "
                    f"cosine_similarity={record.cosine_similarity}"
                )
        except Exception as exc:
            self._warning(f"record_attention_output_diff failed: {exc}")

    def record_output_reuse_dry_run(
        self,
        *,
        previous_tensor: Any,
        current_tensor: Any,
        prompt_idx: Optional[int],
        window_index: int,
        step_idx: Optional[int],
        timestep: Optional[float],
        previous_step_idx: Optional[int],
        previous_timestep: Optional[float],
        block_idx: int,
        chunk_idx: int,
        start_block: int,
        end_block: int,
        fallback_reason: Optional[str] = None,
    ) -> None:
        if not self.should_run_output_reuse_dry_run:
            return

        try:
            metric = self.config.output_reuse_metric.lower().strip()
            thresholds = tuple(float(value) for value in self.config.output_reuse_thresholds)
            if not thresholds:
                thresholds = (0.0,)

            shape_text = _shape_to_string(getattr(current_tensor, "shape", None))
            current = current_tensor.detach()
            previous = previous_tensor.detach()
            batch_size = int(current.shape[0]) if len(current.shape) > 0 else 1
            self._output_reuse_total_candidates += batch_size

            warning_reason = None
            l1rel_values: List[Optional[float]] = [None] * batch_size
            fallback = bool(fallback_reason)

            if metric != "l1rel":
                fallback = True
                fallback_reason = fallback_reason or f"unsupported_metric:{metric}"
            elif previous_step_idx is None:
                fallback = True
                fallback_reason = fallback_reason or "no_previous_output"
            elif tuple(previous.shape) != tuple(current.shape):
                fallback = True
                fallback_reason = (
                    fallback_reason or
                    f"shape_mismatch:{_shape_to_string(previous.shape)}!={shape_text}"
                )
            elif step_idx is None:
                fallback = True
                fallback_reason = fallback_reason or "missing_step_idx"
            else:
                import torch

                with torch.no_grad():
                    current_float = current.float().flatten(1)
                    previous_float = previous.float().flatten(1)
                    denom = previous_float.abs().mean(dim=1).clamp_min(1.0e-8)
                    l1rel_tensor = (
                        (current_float - previous_float).abs().mean(dim=1) / denom
                    )
                    l1rel_values = [
                        float(value) for value in l1rel_tensor.detach().cpu().tolist()
                    ]
                    self._output_reuse_valid_metric_count += len(l1rel_values)
                    self._output_reuse_l1rel_values.extend(l1rel_values)

            for sample_idx, l1rel in enumerate(l1rel_values):
                for threshold in thresholds:
                    reusable = (
                        l1rel is not None and
                        not fallback and
                        l1rel <= float(threshold)
                    )
                    threshold_key = _format_threshold_key(threshold)
                    if reusable:
                        self._output_reuse_reusable_count_by_threshold[threshold_key] = (
                            self._output_reuse_reusable_count_by_threshold.get(
                                threshold_key, 0) + 1
                        )

                    record = FlowCacheOutputReuseDryRun(
                        event="output_reuse_dry_run",
                        timestamp=time.time(),
                        prompt_idx=_optional_int(prompt_idx),
                        sample_idx=sample_idx,
                        window_idx=window_index,
                        window_index=window_index,
                        step_idx=_optional_int(step_idx),
                        previous_step_idx=_optional_int(previous_step_idx),
                        timestep=(
                            None if timestep is None else float(timestep)
                        ),
                        previous_timestep=(
                            None if previous_timestep is None else float(previous_timestep)
                        ),
                        block_idx=block_idx,
                        chunk_idx=chunk_idx,
                        block_id=f"block_{block_idx}",
                        chunk_id=f"window_{window_index}_chunk_{chunk_idx}",
                        start_block=start_block,
                        end_block=end_block,
                        metric=metric,
                        l1rel=None if l1rel is None else float(l1rel),
                        threshold=float(threshold),
                        reusable_by_threshold=bool(reusable),
                        tensor_shape=shape_text,
                        tensor_like=0,
                        fallback=bool(fallback),
                        fallback_reason=fallback_reason,
                        warning_reason=warning_reason,
                    )
                    self.output_reuse_dry_run_records.append(record)
                    if self.config.output_reuse_log_jsonl:
                        self._write_jsonl(asdict(record))

                    print(
                        "[FlowCache][output_reuse_dry_run] "
                        f"prompt_idx={record.prompt_idx} "
                        f"sample_idx={record.sample_idx} "
                        f"window_idx={record.window_idx} "
                        f"block={record.block_id} "
                        f"chunk={record.chunk_id} "
                        f"step_idx={record.step_idx} "
                        f"timestep={record.timestep} "
                        f"l1rel={_format_optional_float(record.l1rel)} "
                        f"threshold={record.threshold:.4f} "
                        f"reusable_by_threshold={record.reusable_by_threshold} "
                        f"fallback={record.fallback} "
                        f"fallback_reason={record.fallback_reason}"
                    )

                    if self.config.output_reuse_debug_verify:
                        self._verify_output_reuse_record(record)
        except Exception as exc:
            self._output_reuse_parse_errors += 1
            self._warning(f"record_output_reuse_dry_run failed: {exc}")

    def summarize(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.config.metadata_enabled:
            return

        try:
            total_window_time = sum(
                record.elapsed_sec or 0.0 for record in self.window_records)
            window_count = len(self.window_records)
            kv_tokens = [
                record.total_visible_kv_tokens
                for record in self.kv_range_records
                if record.total_visible_kv_tokens is not None
            ]
            avg_window_time = total_window_time / window_count if window_count else 0.0
            avg_kv_tokens = sum(kv_tokens) / len(kv_tokens) if kv_tokens else 0.0
            max_kv_tokens = max(kv_tokens) if kv_tokens else 0
            layer_count = _count_unique(
                record.layer_idx for record in self.kv_range_records)

            print(
                "[FlowCache][summary] "
                f"windows={window_count} "
                f"layers={layer_count} "
                f"kv_records={len(self.kv_range_records)} "
                f"events={len(self.event_records)} "
                f"avg_kv_tokens={avg_kv_tokens:.2f} "
                f"max_kv_tokens={max_kv_tokens} "
                f"total_window_time_sec={total_window_time:.6f} "
                f"avg_window_time_sec={avg_window_time:.6f} "
                f"metadata_output_path={self.config.metadata_output_path}"
            )
        except Exception as exc:
            self._warning(f"summarize failed: {exc}")

    def summarize_output_reuse_dry_run(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.config.output_reuse_dry_run_enabled:
            return

        try:
            thresholds = tuple(float(value) for value in self.config.output_reuse_thresholds)
            if not thresholds:
                thresholds = (0.0,)
            l1rel_values = self._output_reuse_l1rel_values
            valid_count = self._output_reuse_valid_metric_count
            reusable_counts = {
                _format_threshold_key(threshold): int(
                    self._output_reuse_reusable_count_by_threshold.get(
                        _format_threshold_key(threshold), 0)
                )
                for threshold in thresholds
            }
            reusable_ratios = {
                key: (count / valid_count if valid_count else 0.0)
                for key, count in reusable_counts.items()
            }
            peak_allocated, peak_reserved = self._collect_cuda_peak_memory()
            summary_payload = {
                "event": "output_reuse_summary",
                "timestamp": time.time(),
                "flowcache_run_id": self._flowcache_run_id,
                "total_candidates": self._output_reuse_total_candidates,
                "valid_metric_count": valid_count,
                "reusable_count_by_threshold": reusable_counts,
                "reusable_ratio_by_threshold": reusable_ratios,
                "min_l1rel": min(l1rel_values) if l1rel_values else None,
                "avg_l1rel": _avg(l1rel_values) if l1rel_values else None,
                "max_l1rel": max(l1rel_values) if l1rel_values else None,
                "warning_count": self._warning_count,
                "parse_errors": self._output_reuse_parse_errors,
                "runtime_sec": time.time() - self._created_timestamp,
                "peak_cuda_allocated_gb": peak_allocated,
                "peak_cuda_reserved_gb": peak_reserved,
                "tensor_like": 0,
                "metadata_output_path": self.config.metadata_output_path,
            }
            if self.config.output_reuse_log_jsonl:
                self._write_jsonl(summary_payload)

            print(
                "[FlowCache][output_reuse_summary] "
                f"total_candidates={summary_payload['total_candidates']} "
                f"valid_metric_count={summary_payload['valid_metric_count']} "
                f"reusable_count_by_threshold={_format_count_dict(reusable_counts)} "
                f"reusable_ratio_by_threshold={_format_ratio_dict(reusable_ratios)} "
                f"min_l1rel={_format_optional_float(summary_payload['min_l1rel'])} "
                f"avg_l1rel={_format_optional_float(summary_payload['avg_l1rel'])} "
                f"max_l1rel={_format_optional_float(summary_payload['max_l1rel'])} "
                f"warning_count={summary_payload['warning_count']} "
                f"parse_errors={summary_payload['parse_errors']} "
                f"runtime_sec={summary_payload['runtime_sec']:.6f} "
                f"peak_cuda_allocated_gb={_format_metric_value(peak_allocated)} "
                f"peak_cuda_reserved_gb={_format_metric_value(peak_reserved)} "
                f"tensor_like=0 "
                f"metadata_output_path={self.config.metadata_output_path}"
            )
        except Exception as exc:
            self._warning(f"summarize_output_reuse_dry_run failed: {exc}")

    def summarize_candidates(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.config.compression_candidate_enabled:
            return
        if not (
            self.config.metadata_enabled or
            self.config.metadata_output_path or
            self.config.debug or
            self.config.log_attention_parts or
            self.config.log_compression_candidates
        ):
            return

        try:
            records = self.compression_candidate_records
            ratios = [record.candidate_ratio for record in records]
            candidate_tokens = [
                record.compressible_history_tokens for record in records]
            total_tokens = [record.total_kv_tokens for record in records]
            protected_sink_tokens = [
                record.protected_sink_tokens for record in records]
            protected_current_tokens = [
                record.protected_current_tokens for record in records]

            avg_candidate_ratio = sum(ratios) / len(ratios) if ratios else 0.0
            max_candidate_ratio = max(ratios) if ratios else 0.0
            avg_candidate_tokens = (
                sum(candidate_tokens) / len(candidate_tokens)
                if candidate_tokens else 0.0
            )
            max_candidate_tokens = max(candidate_tokens) if candidate_tokens else 0
            avg_total_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0.0
            layer_count = _count_unique(
                record.layer_idx for record in records if record.layer_idx is not None)
            window_count = _count_unique(
                record.window_index for record in records if record.window_index is not None)

            print(
                "[FlowCache][candidate_summary] "
                f"windows={window_count} "
                f"layers={layer_count} "
                f"attention_part_records={len(self.attention_part_records)} "
                f"candidate_records={len(records)} "
                f"avg_candidate_ratio={avg_candidate_ratio:.4f} "
                f"max_candidate_ratio={max_candidate_ratio:.4f} "
                f"avg_candidate_tokens={avg_candidate_tokens:.2f} "
                f"max_candidate_tokens={max_candidate_tokens} "
                f"avg_total_kv_tokens={avg_total_tokens:.2f} "
                f"max_total_kv_tokens={max(total_tokens) if total_tokens else 0} "
                f"max_protected_sink_tokens={max(protected_sink_tokens) if protected_sink_tokens else 0} "
                f"max_protected_current_tokens={max(protected_current_tokens) if protected_current_tokens else 0} "
                f"metadata_output_path={self.config.metadata_output_path}"
            )
        except Exception as exc:
            self._warning(f"summarize_candidates failed: {exc}")

    def summarize_compression_dry_run(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.config.kv_compress_enabled or not self.config.kv_compress_dry_run:
            return

        try:
            records = self.compression_dry_run_records
            applied_records = [record for record in records if record.apply_to_branch]
            skipped_records = [record for record in records if not record.apply_to_branch]
            total_original = sum(record.original_total_kv_tokens for record in records)
            total_projected = sum(record.projected_total_kv_tokens for record in records)
            total_saved = sum(record.saved_tokens for record in records)
            saving_ratios = [record.saving_ratio for record in records]
            keep_ratios = [
                record.candidate_keep_ratio for record in records
                if record.compressible_history_tokens > 0
            ]

            print(
                "[FlowCache][compression_dry_run_summary] "
                f"total_events={len(records)} "
                f"applied_events={len(applied_records)} "
                f"skipped_events={len(skipped_records)} "
                f"total_original_tokens={total_original} "
                f"total_projected_tokens={total_projected} "
                f"total_saved_tokens={total_saved} "
                f"avg_saving_ratio={_avg(saving_ratios):.4f} "
                f"max_saving_ratio={max(saving_ratios) if saving_ratios else 0.0:.4f} "
                f"avg_candidate_keep_ratio={_avg(keep_ratios):.4f} "
                f"kv_compress_target_ratio={self.config.kv_compress_target_ratio:.4f} "
                f"by_branch={self._summarize_dry_run_groups(records, 'branch')} "
                f"by_event={self._summarize_dry_run_groups(records, 'attention_event')}"
            )
        except Exception as exc:
            self._warning(f"summarize_compression_dry_run failed: {exc}")

    def summarize_real_compression(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.should_run_real_compression:
            return

        try:
            records = self.real_compression_records
            applied_records = [record for record in records if record.applied]
            skipped_records = [record for record in records if not record.applied]

            applied_original = sum(
                record.original_total_kv_tokens for record in applied_records)
            applied_compressed = sum(
                record.compressed_total_kv_tokens for record in applied_records)
            applied_saved = sum(record.saved_tokens for record in applied_records)
            overall_original = sum(
                record.original_total_kv_tokens for record in records)
            overall_compressed = sum(
                record.compressed_total_kv_tokens for record in records)
            overall_saved = sum(record.saved_tokens for record in records)

            saving_ratios = [record.saving_ratio for record in records]
            applied_weighted_saving_ratio = (
                applied_saved / applied_original if applied_original else 0.0
            )
            overall_weighted_saving_ratio = (
                overall_saved / overall_original if overall_original else 0.0
            )
            fallback_count = sum(1 for record in records if record.fallback)

            diff_records = self.attention_output_diff_records
            relative_l1_values = [record.relative_l1 for record in diff_records]
            cosine_values = [
                record.cosine_similarity for record in diff_records
                if record.cosine_similarity is not None
            ]
            eval_metrics = self._collect_eval_metrics()
            applied_by_branch = _counter_dict(record.branch for record in applied_records)
            skipped_by_reason = _counter_dict(
                record.skipped_reason or "unknown" for record in skipped_records)

            summary_payload = {
                "event": "real_compression_summary",
                "timestamp": time.time(),
                "total_events": len(records),
                "applied_events": len(applied_records),
                "skipped_events": len(skipped_records),
                "applied_original_tokens": applied_original,
                "applied_compressed_tokens": applied_compressed,
                "applied_saved_tokens": applied_saved,
                "overall_original_tokens": overall_original,
                "overall_compressed_tokens": overall_compressed,
                "overall_saved_tokens": overall_saved,
                "applied_weighted_saving_ratio": applied_weighted_saving_ratio,
                "overall_weighted_saving_ratio": overall_weighted_saving_ratio,
                "avg_saving_ratio": _avg(saving_ratios),
                "max_saving_ratio": max(saving_ratios) if saving_ratios else 0.0,
                "fallback_count": fallback_count,
                "warning_count": self._warning_count,
                "real_compression_applied_by_branch": applied_by_branch,
                "real_compression_skipped_by_reason": skipped_by_reason,
                "attention_output_diff_events": len(diff_records),
                "attention_output_diff_avg_relative_l1": _avg(relative_l1_values),
                "attention_output_diff_max_relative_l1": (
                    max(relative_l1_values) if relative_l1_values else 0.0
                ),
                "attention_output_diff_avg_cosine": _avg(cosine_values),
                "attention_output_diff_min_cosine": (
                    min(cosine_values) if cosine_values else 0.0
                ),
                "by_branch": self._summarize_real_groups(records, "branch"),
                "by_layer": self._summarize_real_groups(records, "layer_idx"),
            }
            summary_payload.update(eval_metrics)
            self._write_jsonl(summary_payload)

            eval_summary = " ".join(
                f"{key}={_format_metric_value(value)}"
                for key, value in eval_metrics.items()
            )
            if eval_summary:
                eval_summary = f" {eval_summary}"

            print(
                "[FlowCache][real_compression_summary] "
                f"total_events={summary_payload['total_events']} "
                f"applied_events={summary_payload['applied_events']} "
                f"skipped_events={summary_payload['skipped_events']} "
                f"applied_original_tokens={summary_payload['applied_original_tokens']} "
                f"applied_compressed_tokens={summary_payload['applied_compressed_tokens']} "
                f"applied_saved_tokens={summary_payload['applied_saved_tokens']} "
                f"overall_original_tokens={summary_payload['overall_original_tokens']} "
                f"overall_compressed_tokens={summary_payload['overall_compressed_tokens']} "
                f"overall_saved_tokens={summary_payload['overall_saved_tokens']} "
                f"applied_weighted_saving_ratio={applied_weighted_saving_ratio:.4f} "
                f"overall_weighted_saving_ratio={overall_weighted_saving_ratio:.4f} "
                f"avg_saving_ratio={summary_payload['avg_saving_ratio']:.4f} "
                f"max_saving_ratio={summary_payload['max_saving_ratio']:.4f} "
                f"fallback_count={fallback_count} "
                f"warning_count={self._warning_count} "
                f"real_compression_applied_by_branch={_format_count_dict(applied_by_branch)} "
                f"real_compression_skipped_by_reason={_format_count_dict(skipped_by_reason)} "
                f"attention_output_diff_events={len(diff_records)} "
                f"attention_output_diff_avg_relative_l1={summary_payload['attention_output_diff_avg_relative_l1']:.6e} "
                f"attention_output_diff_max_relative_l1={summary_payload['attention_output_diff_max_relative_l1']:.6e} "
                f"attention_output_diff_avg_cosine={summary_payload['attention_output_diff_avg_cosine']:.6f} "
                f"attention_output_diff_min_cosine={summary_payload['attention_output_diff_min_cosine']:.6f} "
                f"by_branch={summary_payload['by_branch']} "
                f"by_layer={summary_payload['by_layer']}"
                f"{eval_summary}"
            )
        except Exception as exc:
            self._warning(f"summarize_real_compression failed: {exc}")

    def summarize_cache_body_compression(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.should_track_cache_body_compression:
            return

        try:
            records = self.cache_body_compression_records
            applied_records = [record for record in records if record.applied]
            skipped_records = [record for record in records if not record.applied]

            applied_original = sum(
                record.original_visible_tokens for record in applied_records)
            applied_compressed = sum(
                record.compressed_visible_tokens for record in applied_records)
            applied_saved = sum(
                record.saved_visible_tokens for record in applied_records)
            overall_original = sum(
                record.original_visible_tokens for record in records)
            overall_compressed = sum(
                record.compressed_visible_tokens for record in records)
            overall_saved = sum(
                record.saved_visible_tokens for record in records)

            applied_weighted_visible_saving_ratio = (
                applied_saved / applied_original if applied_original else 0.0
            )
            overall_weighted_visible_saving_ratio = (
                overall_saved / overall_original if overall_original else 0.0
            )
            fallback_count = sum(1 for record in records if record.fallback)
            current_only_applied = sum(
                1 for record in applied_records if record.branch == "current_only")
            clean_cache_update_applied = sum(
                1 for record in applied_records
                if record.branch == "clean_cache_update")
            applied_by_branch = _counter_dict(
                record.branch for record in applied_records)
            skipped_by_reason = _counter_dict(
                record.skipped_reason or "unknown" for record in skipped_records)
            peak_allocated, peak_reserved = self._collect_cuda_peak_memory()

            summary_payload = {
                "event": "cache_body_summary",
                "timestamp": time.time(),
                "total_events": len(records),
                "applied_events": len(applied_records),
                "skipped_events": len(skipped_records),
                "current_only_applied": current_only_applied,
                "clean_cache_update_applied": clean_cache_update_applied,
                "fallback_count": fallback_count,
                "warning_count": self._warning_count,
                "total_original_visible_tokens": overall_original,
                "total_compressed_visible_tokens": overall_compressed,
                "total_saved_visible_tokens": overall_saved,
                "applied_original_visible_tokens": applied_original,
                "applied_compressed_visible_tokens": applied_compressed,
                "applied_saved_visible_tokens": applied_saved,
                "applied_weighted_visible_saving_ratio": (
                    applied_weighted_visible_saving_ratio
                ),
                "overall_weighted_visible_saving_ratio": (
                    overall_weighted_visible_saving_ratio
                ),
                "runtime_sec": time.time() - self._created_timestamp,
                "peak_cuda_allocated_gb": peak_allocated,
                "peak_cuda_reserved_gb": peak_reserved,
                "cache_body_applied_by_branch": applied_by_branch,
                "cache_body_skipped_by_reason": skipped_by_reason,
                "by_branch": self._summarize_cache_body_groups(records, "branch"),
                "by_layer": self._summarize_cache_body_groups(records, "layer_idx"),
            }
            self._write_jsonl(summary_payload)

            print(
                "[FlowCache][cache_body_summary] "
                f"total_events={summary_payload['total_events']} "
                f"applied_events={summary_payload['applied_events']} "
                f"skipped_events={summary_payload['skipped_events']} "
                f"current_only_applied={current_only_applied} "
                f"clean_cache_update_applied={clean_cache_update_applied} "
                f"fallback_count={fallback_count} "
                f"warning_count={self._warning_count} "
                f"total_original_visible_tokens={overall_original} "
                f"total_compressed_visible_tokens={overall_compressed} "
                f"total_saved_visible_tokens={overall_saved} "
                f"applied_weighted_visible_saving_ratio={applied_weighted_visible_saving_ratio:.4f} "
                f"overall_weighted_visible_saving_ratio={overall_weighted_visible_saving_ratio:.4f} "
                f"runtime_sec={summary_payload['runtime_sec']:.6f} "
                f"peak_cuda_allocated_gb={_format_metric_value(peak_allocated)} "
                f"peak_cuda_reserved_gb={_format_metric_value(peak_reserved)} "
                f"cache_body_applied_by_branch={_format_count_dict(applied_by_branch)} "
                f"cache_body_skipped_by_reason={_format_count_dict(skipped_by_reason)} "
                f"by_branch={summary_payload['by_branch']} "
                f"by_layer={summary_payload['by_layer']}"
            )
        except Exception as exc:
            self._warning(f"summarize_cache_body_compression failed: {exc}")

    def summarize_persistent_cache_compression(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.should_track_persistent_cache_compression:
            return

        try:
            records = self.persistent_cache_compression_records
            sidecar_records = self.persistent_cache_sidecar_records
            applied_records = [record for record in records if record.applied]
            skipped_records = [record for record in records if not record.applied]

            applied_original = sum(
                record.original_visible_tokens for record in applied_records)
            applied_compressed = sum(
                record.compressed_visible_tokens for record in applied_records)
            applied_saved = sum(
                record.saved_visible_tokens for record in applied_records)
            overall_original = sum(
                record.original_visible_tokens for record in records)
            overall_compressed = sum(
                record.compressed_visible_tokens for record in records)
            overall_saved = sum(
                record.saved_visible_tokens for record in records)

            applied_weighted_visible_saving_ratio = (
                applied_saved / applied_original if applied_original else 0.0
            )
            overall_weighted_visible_saving_ratio = (
                overall_saved / overall_original if overall_original else 0.0
            )
            fallback_count = sum(1 for record in records if record.fallback)
            current_only_applied = sum(
                1 for record in applied_records if record.branch == "current_only")
            clean_cache_update_applied = sum(
                1 for record in applied_records
                if record.branch == "clean_cache_update")
            sidecar_action_counts = dict(sorted(
                self._persistent_sidecar_action_counts.items()))
            sidecar_record_action_counts = _counter_dict(
                record.action for record in sidecar_records)
            sidecar_action_count_consistent = (
                sidecar_action_counts == sidecar_record_action_counts
            )
            if not sidecar_action_count_consistent:
                sidecar_action_counts = sidecar_record_action_counts
            sidecar_create_count = sidecar_action_counts.get("create", 0)
            sidecar_reuse_count = sidecar_action_counts.get("reuse", 0)
            sidecar_invalidate_count = sidecar_action_counts.get("invalidate", 0)
            sidecar_fallback_count = sidecar_action_counts.get("fallback", 0)
            sidecar_keep_valid_count = sidecar_action_counts.get("keep_valid", 0)
            applied_by_branch = _counter_dict(
                record.branch for record in applied_records)
            skipped_by_reason = _counter_dict(
                record.skipped_reason or "unknown" for record in skipped_records)
            peak_allocated, peak_reserved = self._collect_cuda_peak_memory()
            self._persistent_summary_index += 1

            summary_payload = {
                "event": "persistent_cache_summary",
                "timestamp": time.time(),
                "summary_scope": "manager_lifetime_cumulative",
                "summary_index": self._persistent_summary_index,
                "sidecar_action_count_consistent": sidecar_action_count_consistent,
                "sidecar_action_counts": sidecar_action_counts,
                "total_events": len(records),
                "applied_events": len(applied_records),
                "skipped_events": len(skipped_records),
                "sidecar_create_count": sidecar_create_count,
                "sidecar_reuse_count": sidecar_reuse_count,
                "sidecar_invalidate_count": sidecar_invalidate_count,
                "sidecar_fallback_count": sidecar_fallback_count,
                "sidecar_keep_valid_count": sidecar_keep_valid_count,
                "current_only_applied": current_only_applied,
                "clean_cache_update_applied": clean_cache_update_applied,
                "fallback_count": fallback_count,
                "warning_count": self._warning_count,
                "total_original_visible_tokens": overall_original,
                "total_compressed_visible_tokens": overall_compressed,
                "total_saved_visible_tokens": overall_saved,
                "applied_original_visible_tokens": applied_original,
                "applied_compressed_visible_tokens": applied_compressed,
                "applied_saved_visible_tokens": applied_saved,
                "applied_weighted_visible_saving_ratio": (
                    applied_weighted_visible_saving_ratio
                ),
                "overall_weighted_visible_saving_ratio": (
                    overall_weighted_visible_saving_ratio
                ),
                "runtime_sec": time.time() - self._created_timestamp,
                "peak_cuda_allocated_gb": peak_allocated,
                "peak_cuda_reserved_gb": peak_reserved,
                "persistent_cache_applied_by_branch": applied_by_branch,
                "persistent_cache_skipped_by_reason": skipped_by_reason,
                "by_branch": self._summarize_persistent_cache_groups(
                    records, "branch"),
                "by_layer": self._summarize_persistent_cache_groups(
                    records, "layer_idx"),
            }
            self._write_jsonl(summary_payload)

            print(
                "[FlowCache][persistent_cache_summary] "
                f"flowcache_run_id={self._flowcache_run_id} "
                f"summary_scope={summary_payload['summary_scope']} "
                f"summary_index={summary_payload['summary_index']} "
                f"total_events={summary_payload['total_events']} "
                f"applied_events={summary_payload['applied_events']} "
                f"skipped_events={summary_payload['skipped_events']} "
                f"sidecar_create_count={sidecar_create_count} "
                f"sidecar_reuse_count={sidecar_reuse_count} "
                f"sidecar_invalidate_count={sidecar_invalidate_count} "
                f"sidecar_fallback_count={sidecar_fallback_count} "
                f"sidecar_keep_valid_count={sidecar_keep_valid_count} "
                f"sidecar_action_counts={_format_count_dict(sidecar_action_counts)} "
                f"sidecar_action_count_consistent={sidecar_action_count_consistent} "
                f"current_only_applied={current_only_applied} "
                f"clean_cache_update_applied={clean_cache_update_applied} "
                f"fallback_count={fallback_count} "
                f"warning_count={self._warning_count} "
                f"total_original_visible_tokens={overall_original} "
                f"total_compressed_visible_tokens={overall_compressed} "
                f"total_saved_visible_tokens={overall_saved} "
                f"applied_weighted_visible_saving_ratio={applied_weighted_visible_saving_ratio:.4f} "
                f"overall_weighted_visible_saving_ratio={overall_weighted_visible_saving_ratio:.4f} "
                f"runtime_sec={summary_payload['runtime_sec']:.6f} "
                f"peak_cuda_allocated_gb={_format_metric_value(peak_allocated)} "
                f"peak_cuda_reserved_gb={_format_metric_value(peak_reserved)} "
                f"persistent_cache_applied_by_branch={_format_count_dict(applied_by_branch)} "
                f"persistent_cache_skipped_by_reason={_format_count_dict(skipped_by_reason)} "
                f"by_branch={summary_payload['by_branch']} "
                f"by_layer={summary_payload['by_layer']}"
            )
        except Exception as exc:
            self._warning(f"summarize_persistent_cache_compression failed: {exc}")

    def summarize_compacted_kv(self) -> None:
        if not self.config.enabled or not self.config.log_summary:
            return
        if not self.should_track_compacted_kv:
            return

        try:
            records = self.compacted_kv_records
            applied_records = [record for record in records if record.applied]
            skipped_records = [record for record in records if not record.applied]

            applied_original = sum(
                record.original_visible_tokens for record in applied_records)
            applied_compacted = sum(
                record.compacted_visible_tokens for record in applied_records)
            applied_saved = sum(
                record.saved_visible_tokens for record in applied_records)
            overall_original = sum(
                record.original_visible_tokens for record in records)
            overall_compacted = sum(
                record.compacted_visible_tokens for record in records)
            overall_saved = sum(
                record.saved_visible_tokens for record in records)

            applied_weighted_visible_saving_ratio = (
                applied_saved / applied_original if applied_original else 0.0
            )
            overall_weighted_visible_saving_ratio = (
                overall_saved / overall_original if overall_original else 0.0
            )
            fallback_count = sum(1 for record in records if record.fallback)
            current_only_applied = sum(
                1 for record in applied_records if record.branch == "current_only")
            clean_cache_update_applied = sum(
                1 for record in applied_records
                if record.branch == "clean_cache_update")
            applied_by_branch = _counter_dict(
                record.branch for record in applied_records)
            skipped_by_reason = _counter_dict(
                record.skipped_reason or "unknown" for record in skipped_records)
            fallback_by_reason = _counter_dict(
                record.fallback_reason or "unknown"
                for record in records if record.fallback)
            original_history_tokens = [
                record.original_history_tokens for record in applied_records
            ]
            compacted_history_tokens = [
                record.compacted_history_tokens for record in applied_records
            ]
            avg_original_history = _avg(original_history_tokens)
            avg_compacted_history = _avg(compacted_history_tokens)
            history_token_ratio = (
                sum(compacted_history_tokens) / sum(original_history_tokens)
                if sum(original_history_tokens) else 0.0
            )
            peak_allocated, peak_reserved = self._collect_cuda_peak_memory()

            summary_payload = {
                "event": "compacted_kv_summary",
                "timestamp": time.time(),
                "summary_scope": "manager_lifetime_cumulative",
                "total_events": len(records),
                "applied_events": len(applied_records),
                "skipped_events": len(skipped_records),
                "fallback_count": fallback_count,
                "warning_count": self._warning_count,
                "current_only_applied": current_only_applied,
                "clean_cache_update_applied": clean_cache_update_applied,
                "total_original_visible_tokens": overall_original,
                "total_compacted_visible_tokens": overall_compacted,
                "total_saved_visible_tokens": overall_saved,
                "applied_original_visible_tokens": applied_original,
                "applied_compacted_visible_tokens": applied_compacted,
                "applied_saved_visible_tokens": applied_saved,
                "applied_weighted_visible_saving_ratio": (
                    applied_weighted_visible_saving_ratio
                ),
                "overall_weighted_visible_saving_ratio": (
                    overall_weighted_visible_saving_ratio
                ),
                "avg_original_history_tokens": avg_original_history,
                "avg_compacted_history_tokens": avg_compacted_history,
                "compacted_original_history_token_ratio": history_token_ratio,
                "runtime_sec": time.time() - self._created_timestamp,
                "peak_cuda_allocated_gb": peak_allocated,
                "peak_cuda_reserved_gb": peak_reserved,
                "target_ratio": self.config.compacted_kv_target_ratio,
                "strategy": self.config.compacted_kv_strategy,
                "compacted_kv_applied_by_branch": applied_by_branch,
                "compacted_kv_skipped_by_reason": skipped_by_reason,
                "compacted_kv_fallback_by_reason": fallback_by_reason,
            }
            self._write_jsonl(summary_payload)

            print(
                "[FlowCache][compacted_kv_summary] "
                f"flowcache_run_id={self._flowcache_run_id} "
                f"summary_scope={summary_payload['summary_scope']} "
                f"total_events={summary_payload['total_events']} "
                f"applied_events={summary_payload['applied_events']} "
                f"skipped_events={summary_payload['skipped_events']} "
                f"fallback_count={fallback_count} "
                f"warning_count={self._warning_count} "
                f"current_only_applied={current_only_applied} "
                f"clean_cache_update_applied={clean_cache_update_applied} "
                f"total_original_visible_tokens={overall_original} "
                f"total_compacted_visible_tokens={overall_compacted} "
                f"total_saved_visible_tokens={overall_saved} "
                f"applied_weighted_visible_saving_ratio={applied_weighted_visible_saving_ratio:.4f} "
                f"overall_weighted_visible_saving_ratio={overall_weighted_visible_saving_ratio:.4f} "
                f"avg_original_history_tokens={avg_original_history:.2f} "
                f"avg_compacted_history_tokens={avg_compacted_history:.2f} "
                f"compacted_original_history_token_ratio={history_token_ratio:.4f} "
                f"target_ratio={self.config.compacted_kv_target_ratio:.4f} "
                f"strategy={self.config.compacted_kv_strategy} "
                f"runtime_sec={summary_payload['runtime_sec']:.6f} "
                f"peak_cuda_allocated_gb={_format_metric_value(peak_allocated)} "
                f"peak_cuda_reserved_gb={_format_metric_value(peak_reserved)} "
                f"compacted_kv_applied_by_branch={_format_count_dict(applied_by_branch)} "
                f"compacted_kv_skipped_by_reason={_format_count_dict(skipped_by_reason)} "
                f"compacted_kv_fallback_by_reason={_format_count_dict(fallback_by_reason)}"
            )
        except Exception as exc:
            self._warning(f"summarize_compacted_kv failed: {exc}")

    def _resolve_dry_run_branch_policy(
        self,
        candidate_record: FlowCacheCompressionCandidate,
    ) -> tuple:
        source_event = candidate_record.source_event
        branch = candidate_record.attention_branch

        if branch == "current_only":
            return False, "current_only_protected"

        if source_event == "clean_cache_update_attention" or branch == "clean_cache_update":
            if self.config.kv_compress_apply_to_clean_cache_update:
                return True, None
            return False, "clean_cache_update_disabled"

        if source_event == "denoise_attention" and branch == "anchor_working_current":
            if self.config.kv_compress_apply_to_denoise:
                return True, None
            return False, "denoise_disabled"

        self._warning(
            f"unknown dry-run branch policy for source_event={source_event}, branch={branch}")
        return False, "unknown_branch"

    def _compute_projected_tokens(
        self,
        *,
        compressible_history_tokens: int,
        apply_to_branch: bool,
    ) -> int:
        if not apply_to_branch:
            return compressible_history_tokens
        return int(math.ceil(
            compressible_history_tokens * self.config.kv_compress_target_ratio))

    def _summarize_dry_run_groups(
        self,
        records: List[FlowCacheCompressionDryRun],
        key: str,
    ) -> str:
        groups: Dict[str, Dict[str, Any]] = {}
        for record in records:
            group_name = str(getattr(record, key))
            group = groups.setdefault(
                group_name,
                {
                    "events": 0,
                    "applied": 0,
                    "skipped": 0,
                    "original": 0,
                    "projected": 0,
                    "saved": 0,
                },
            )
            group["events"] += 1
            group["applied"] += int(record.apply_to_branch)
            group["skipped"] += int(not record.apply_to_branch)
            group["original"] += record.original_total_kv_tokens
            group["projected"] += record.projected_total_kv_tokens
            group["saved"] += record.saved_tokens

        parts = []
        for group_name in sorted(groups):
            group = groups[group_name]
            ratio = group["saved"] / group["original"] if group["original"] else 0.0
            parts.append(
                f"{group_name}:events={group['events']},applied={group['applied']},"
                f"skipped={group['skipped']},saved={group['saved']},"
                f"ratio={ratio:.4f}"
            )
        return "|".join(parts)

    def _summarize_cache_body_groups(
        self,
        records: List[FlowCacheCacheBodyCompression],
        key: str,
    ) -> str:
        groups: Dict[str, Dict[str, Any]] = {}
        for record in records:
            group_name = str(getattr(record, key))
            group = groups.setdefault(
                group_name,
                {
                    "events": 0,
                    "applied": 0,
                    "skipped": 0,
                    "fallbacks": 0,
                    "original_visible": 0,
                    "compressed_visible": 0,
                    "saved_visible": 0,
                },
            )
            group["events"] += 1
            group["applied"] += int(record.applied)
            group["skipped"] += int(not record.applied)
            group["fallbacks"] += int(record.fallback)
            if record.applied:
                group["original_visible"] += record.original_visible_tokens
                group["compressed_visible"] += record.compressed_visible_tokens
                group["saved_visible"] += record.saved_visible_tokens

        parts = []
        for group_name in sorted(groups):
            group = groups[group_name]
            ratio = (
                group["saved_visible"] / group["original_visible"]
                if group["original_visible"] else 0.0
            )
            parts.append(
                f"{group_name}:events={group['events']},applied={group['applied']},"
                f"skipped={group['skipped']},fallbacks={group['fallbacks']},"
                f"saved_visible={group['saved_visible']},ratio={ratio:.4f}"
            )
        return "|".join(parts)

    def _summarize_persistent_cache_groups(
        self,
        records: List[FlowCachePersistentCacheCompression],
        key: str,
    ) -> str:
        groups: Dict[str, Dict[str, Any]] = {}
        for record in records:
            group_name = str(getattr(record, key))
            group = groups.setdefault(
                group_name,
                {
                    "events": 0,
                    "applied": 0,
                    "skipped": 0,
                    "fallbacks": 0,
                    "created": 0,
                    "reused": 0,
                    "invalidated": 0,
                    "sidecar_fallbacks": 0,
                    "original_visible": 0,
                    "compressed_visible": 0,
                    "saved_visible": 0,
                },
            )
            group["events"] += 1
            group["applied"] += int(record.applied)
            group["skipped"] += int(not record.applied)
            group["fallbacks"] += int(record.fallback)
            group["created"] += int(record.sidecar_created)
            group["reused"] += int(record.sidecar_reused)
            group["invalidated"] += int(record.sidecar_invalidated)
            group["sidecar_fallbacks"] += int(record.sidecar_fallback)
            if record.applied:
                group["original_visible"] += record.original_visible_tokens
                group["compressed_visible"] += record.compressed_visible_tokens
                group["saved_visible"] += record.saved_visible_tokens

        parts = []
        for group_name in sorted(groups):
            group = groups[group_name]
            ratio = (
                group["saved_visible"] / group["original_visible"]
                if group["original_visible"] else 0.0
            )
            parts.append(
                f"{group_name}:events={group['events']},applied={group['applied']},"
                f"skipped={group['skipped']},fallbacks={group['fallbacks']},"
                f"created={group['created']},reused={group['reused']},"
                f"invalidated={group['invalidated']},"
                f"sidecar_fallbacks={group['sidecar_fallbacks']},"
                f"saved_visible={group['saved_visible']},ratio={ratio:.4f}"
            )
        return "|".join(parts)

    def _summarize_real_groups(
        self,
        records: List[FlowCacheRealCompression],
        key: str,
    ) -> str:
        groups: Dict[str, Dict[str, Any]] = {}
        for record in records:
            group_name = str(getattr(record, key))
            group = groups.setdefault(
                group_name,
                {
                    "events": 0,
                    "applied": 0,
                    "skipped": 0,
                    "fallbacks": 0,
                    "original": 0,
                    "compressed": 0,
                    "saved": 0,
                },
            )
            group["events"] += 1
            group["applied"] += int(record.applied)
            group["skipped"] += int(not record.applied)
            group["fallbacks"] += int(record.fallback)
            if record.applied:
                group["original"] += record.original_total_kv_tokens
                group["compressed"] += record.compressed_total_kv_tokens
                group["saved"] += record.saved_tokens

        parts = []
        for group_name in sorted(groups):
            group = groups[group_name]
            ratio = group["saved"] / group["original"] if group["original"] else 0.0
            parts.append(
                f"{group_name}:events={group['events']},applied={group['applied']},"
                f"skipped={group['skipped']},fallbacks={group['fallbacks']},"
                f"saved={group['saved']},ratio={ratio:.4f}"
            )
        return "|".join(parts)

    def _collect_eval_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        collect_runtime = (
            self.config.eval_metrics_enabled or self.config.eval_runtime_enabled)
        collect_memory = (
            self.config.eval_metrics_enabled or self.config.eval_memory_enabled)

        if collect_runtime:
            metrics["total_runtime_sec"] = time.time() - self._created_timestamp

        if collect_memory:
            metrics["peak_cuda_allocated_gb"] = None
            metrics["peak_cuda_reserved_gb"] = None
            try:
                import torch

                if torch.cuda.is_available():
                    scale = 1024 ** 3
                    metrics["peak_cuda_allocated_gb"] = (
                        torch.cuda.max_memory_allocated() / scale
                    )
                    metrics["peak_cuda_reserved_gb"] = (
                        torch.cuda.max_memory_reserved() / scale
                    )
            except Exception as exc:
                self._warning(f"failed to collect cuda memory metrics: {exc}")

        return metrics

    def _collect_cuda_peak_memory(self) -> tuple:
        try:
            import torch

            if torch.cuda.is_available():
                scale = 1024 ** 3
                return (
                    torch.cuda.max_memory_allocated() / scale,
                    torch.cuda.max_memory_reserved() / scale,
                )
        except Exception as exc:
                self._warning(f"failed to collect cuda peak memory: {exc}")
        return None, None

    def _verify_output_reuse_record(
        self,
        record: FlowCacheOutputReuseDryRun,
    ) -> None:
        issues = []
        if record.tensor_like != 0:
            issues.append("tensor_like_nonzero")
        if record.metric != "l1rel":
            issues.append(f"unsupported_metric:{record.metric}")
        if record.l1rel is not None and record.l1rel < 0:
            issues.append("negative_l1rel")
        expected_reusable = (
            record.l1rel is not None and
            not record.fallback and
            record.l1rel <= record.threshold
        )
        if record.reusable_by_threshold != expected_reusable:
            issues.append("reusable_threshold_mismatch")

        for issue in issues:
            self._warning(f"output-reuse dry-run verify failed: {issue}")

    def _verify_cache_body_record(
        self,
        record: FlowCacheCacheBodyCompression,
    ) -> None:
        issues = []
        expected_original = (
            record.protected_sink_tokens +
            record.original_history_tokens +
            record.protected_current_tokens
        )
        expected_compressed = (
            record.protected_sink_tokens +
            record.compressed_history_tokens +
            record.protected_current_tokens
        )

        if record.original_visible_tokens != expected_original:
            issues.append(
                "original_visible_formula "
                f"actual={record.original_visible_tokens} expected={expected_original}")
        if record.compressed_visible_tokens != expected_compressed:
            issues.append(
                "compressed_visible_formula "
                f"actual={record.compressed_visible_tokens} expected={expected_compressed}")
        if record.compressed_history_tokens > record.original_history_tokens:
            issues.append(
                "compressed_history_exceeds_original "
                f"{record.compressed_history_tokens}>{record.original_history_tokens}")
        if record.compressed_visible_tokens > record.original_visible_tokens:
            issues.append(
                "compressed_visible_exceeds_original "
                f"{record.compressed_visible_tokens}>{record.original_visible_tokens}")
        if record.saved_visible_tokens < 0:
            issues.append("negative_saved_visible_tokens")
        if record.branch == "current_only" and record.applied:
            issues.append("current_only_applied")
        if (
            record.branch == "clean_cache_update" and
            record.applied and
            not self.config.cache_body_compress_apply_to_clean_cache_update
        ):
            issues.append("clean_cache_update_applied_while_disabled")

        for issue in issues:
            self._warning(f"cache-body verify failed: {issue}")

        print(
            "[FlowCache][cache_body_verify] "
            f"window={record.window_index} "
            f"layer={record.layer_idx} "
            f"branch={record.branch} "
            f"passed={not issues} "
            f"issues={';'.join(issues) if issues else 'none'}"
        )

    def _verify_persistent_cache_record(
        self,
        record: FlowCachePersistentCacheCompression,
    ) -> None:
        issues = []
        expected_original = (
            record.protected_sink_tokens +
            record.original_history_tokens +
            record.protected_current_tokens
        )
        expected_compressed = (
            record.protected_sink_tokens +
            record.compressed_history_tokens +
            record.protected_current_tokens
        )

        if record.original_visible_tokens != expected_original:
            issues.append(
                "original_visible_formula "
                f"actual={record.original_visible_tokens} expected={expected_original}")
        if record.compressed_visible_tokens != expected_compressed:
            issues.append(
                "compressed_visible_formula "
                f"actual={record.compressed_visible_tokens} expected={expected_compressed}")
        if record.compressed_history_tokens > record.original_history_tokens:
            issues.append(
                "compressed_history_exceeds_original "
                f"{record.compressed_history_tokens}>{record.original_history_tokens}")
        if record.compressed_visible_tokens > record.original_visible_tokens:
            issues.append(
                "compressed_visible_exceeds_original "
                f"{record.compressed_visible_tokens}>{record.original_visible_tokens}")
        if record.saved_visible_tokens < 0:
            issues.append("negative_saved_visible_tokens")
        if record.branch == "current_only" and record.applied:
            issues.append("current_only_applied")
        if (
            record.branch == "clean_cache_update" and
            record.applied and
            not self.config.persistent_cache_compress_apply_to_clean_cache_update
        ):
            issues.append("clean_cache_update_applied_while_disabled")
        if record.applied and not record.sidecar_valid:
            issues.append("applied_without_valid_sidecar")
        if record.source_start is not None and record.source_end is not None:
            if record.source_end < record.source_start:
                issues.append("invalid_source_range")

        for issue in issues:
            self._warning(f"persistent-cache verify failed: {issue}")

        print(
            "[FlowCache][persistent_cache_verify] "
            f"window={record.window_index} "
            f"layer={record.layer_idx} "
            f"branch={record.branch} "
            f"passed={not issues} "
            f"issues={';'.join(issues) if issues else 'none'}"
        )

    def _verify_compacted_kv_record(
        self,
        record: FlowCacheCompactedKV,
    ) -> None:
        issues = []
        expected_original = (
            record.protected_sink_tokens +
            record.original_history_tokens +
            record.protected_current_tokens
        )
        expected_compacted = (
            record.protected_sink_tokens +
            record.compacted_history_tokens +
            record.protected_current_tokens
        )

        if record.original_visible_tokens != expected_original:
            issues.append(
                "original_visible_formula "
                f"actual={record.original_visible_tokens} expected={expected_original}")
        if record.compacted_visible_tokens != expected_compacted:
            issues.append(
                "compacted_visible_formula "
                f"actual={record.compacted_visible_tokens} expected={expected_compacted}")
        if record.compacted_history_tokens > record.original_history_tokens:
            issues.append(
                "compacted_history_exceeds_original "
                f"{record.compacted_history_tokens}>{record.original_history_tokens}")
        if record.compacted_visible_tokens > record.original_visible_tokens:
            issues.append(
                "compacted_visible_exceeds_original "
                f"{record.compacted_visible_tokens}>{record.original_visible_tokens}")
        if record.saved_visible_tokens < 0:
            issues.append("negative_saved_visible_tokens")
        if record.branch == "current_only" and record.applied:
            issues.append("current_only_applied")
        if (
            record.branch == "clean_cache_update" and
            record.applied and
            not self.config.compacted_kv_apply_to_clean_cache_update
        ):
            issues.append("clean_cache_update_applied_while_disabled")
        if record.fallback and record.applied:
            issues.append("fallback_record_marked_applied")

        for issue in issues:
            self._warning(f"compacted-kv verify failed: {issue}")

        print(
            "[FlowCache][compacted_kv_verify] "
            f"window={record.window_index} "
            f"layer={record.layer_idx} "
            f"branch={record.branch} "
            f"passed={not issues} "
            f"issues={';'.join(issues) if issues else 'none'}"
        )

    def _build_kv_range_record(
        self,
        *,
        cache_item: Any,
        event: str,
        window_index: int,
        start_block: int,
        end_block: int,
        rolling_window_length_blocks: int,
        num_frame_per_block: int,
        frame_seq_length: int,
        current_num_frames: Optional[int],
        max_attention_tokens: Optional[int],
        layer_idx: int,
    ) -> FlowCacheKVRange:
        global_end_index = _scalar_to_int(_dict_get(cache_item, "global_end_index"))
        local_end_index = _scalar_to_int(_dict_get(cache_item, "local_end_index"))
        block_length = int(num_frame_per_block) * int(frame_seq_length)
        cache_tokens = max(0, int(local_end_index or 0))
        query_frames = int(current_num_frames or 0)
        current_tokens = max(0, query_frames * int(frame_seq_length))
        if max_attention_tokens is None:
            max_attention_tokens = 21 * int(frame_seq_length)
        max_attention_tokens = int(max_attention_tokens)

        if cache_tokens <= block_length:
            sink_tokens = 0
            working_history_tokens = 0
        else:
            sink_tokens = min(block_length, cache_tokens)
            local_start_index = max(0, cache_tokens - block_length)
            working_cache_max_length = max(
                0, max_attention_tokens - current_tokens - block_length)
            extract_cache_end = local_start_index
            extract_cache_start = max(
                block_length, local_start_index - working_cache_max_length)
            working_history_tokens = max(0, extract_cache_end - extract_cache_start)

        sink_start = 0
        sink_end = sink_tokens
        history_start = sink_end
        history_end = history_start + working_history_tokens
        current_start = history_end
        current_end = current_start + current_tokens
        total_visible_kv_tokens = current_end

        return FlowCacheKVRange(
            event=event,
            timestamp=time.time(),
            window_index=window_index,
            start_block=start_block,
            end_block=end_block,
            num_frame_per_block=num_frame_per_block,
            rolling_window_length_blocks=rolling_window_length_blocks,
            layer_idx=layer_idx,
            sink_start=sink_start,
            sink_end=sink_end,
            history_start=history_start,
            history_end=history_end,
            current_start=current_start,
            current_end=current_end,
            global_end_index=global_end_index,
            local_end_index=local_end_index,
            cache_tokens=cache_tokens,
            working_history_tokens=working_history_tokens,
            current_tokens=current_tokens,
            total_visible_kv_tokens=total_visible_kv_tokens,
            total_kv_tokens=total_visible_kv_tokens,
            block_length=block_length,
            frame_seq_length=frame_seq_length,
            current_num_frames=current_num_frames,
            max_attention_tokens=max_attention_tokens,
        )

    def _record_profiler_payload(self, payload: Dict[str, Any]) -> None:
        safe_payload = self._profiler_json_safe(payload)
        phase = str(safe_payload.get("phase", "unknown"))
        elapsed_sec = float(safe_payload.get("elapsed_sec", 0.0) or 0.0)
        count = int(safe_payload.get("count", 1) or 1)
        stats = self._profiler_phase_stats.setdefault(
            phase,
            {"elapsed_sec": 0.0, "count": 0})
        stats["elapsed_sec"] += elapsed_sec
        stats["count"] += count
        if self._profiler_is_sampled_phase(phase):
            self._profiler_add_breakdown(
                self._profiler_sampled_layer_stats,
                phase,
                safe_payload.get("layer_idx"),
                elapsed_sec,
                count)
            self._profiler_add_breakdown(
                self._profiler_sampled_step_stats,
                phase,
                safe_payload.get("window_index"),
                elapsed_sec,
                count)
        if self.config.profiler_log_events:
            self._profiler_events.append(safe_payload)
            self._write_profiler_jsonl(safe_payload)

    def _profiler_is_sampled_phase(self, phase: str) -> bool:
        return str(phase).endswith("_sampled")

    def _profiler_add_breakdown(
        self,
        target: Dict[str, Dict[str, Dict[str, Any]]],
        phase: str,
        key: Any,
        elapsed_sec: float,
        count: int,
    ) -> None:
        key_text = "none" if key is None else str(key)
        phase_stats = target.setdefault(phase, {})
        stats = phase_stats.setdefault(
            key_text,
            {"elapsed_sec": 0.0, "count": 0})
        stats["elapsed_sec"] += float(elapsed_sec)
        stats["count"] += int(count)

    def _profiler_format_breakdown(
        self,
        source: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        formatted: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for phase, groups in sorted(source.items()):
            formatted[phase] = {}
            for key, stats in sorted(groups.items(), key=lambda item: item[0]):
                count = int(stats.get("count", 0) or 0)
                elapsed_sec = float(stats.get("elapsed_sec", 0.0) or 0.0)
                formatted[phase][key] = {
                    "elapsed_sec": elapsed_sec,
                    "count": count,
                    "avg_ms": elapsed_sec / count * 1000.0 if count else 0.0,
                }
        return formatted

    def _profiler_top_phases(
        self,
        phase_time_sec: Dict[str, float],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        rows = [
            {"phase": phase, "elapsed_sec": elapsed}
            for phase, elapsed in phase_time_sec.items()
            if phase not in {"total_inference", "total_runtime", "prompt_total"}
        ]
        rows.sort(key=lambda item: item["elapsed_sec"], reverse=True)
        return rows[:limit]

    def _profiler_next_step_recommendation(
        self,
        phase_time_ratio: Dict[str, float],
        sampled_phase_time_sec: Dict[str, float],
        sampled_avg_ms_by_phase: Dict[str, float],
    ) -> str:
        if not sampled_phase_time_sec:
            return (
                "fine profiling 未采到 sampled phase；请确认 profiler_fine_enabled、"
                "profiler_sample_layers 和 profiler_sample_every_n_steps。"
            )

        attention_time = sum(
            sampled_phase_time_sec.get(phase, 0.0)
            for phase in [
                "attention_forward_sampled",
                "attention_qkv_sampled",
                "attention_compute_sampled",
                "attention_output_sampled",
                "cross_attention_sampled",
            ])
        mlp_time = sampled_phase_time_sec.get("mlp_forward_sampled", 0.0)
        kv_time = sum(
            sampled_phase_time_sec.get(phase, 0.0)
            for phase in [
                "kv_cache_read_sampled",
                "kv_cache_write_sampled",
                "kv_eviction_sampled",
            ])
        vae_ratio = phase_time_ratio.get("vae_decode_total", 0.0)
        save_ratio = phase_time_ratio.get("video_save_total", 0.0)

        if attention_time >= mlp_time and attention_time >= kv_time:
            return "Round 8 优先拆 attention；同时保留 VAE/video save 作为端到端备选。"
        if mlp_time >= attention_time and mlp_time >= kv_time:
            return "Round 8 优先拆 MLP/FFN；attention 作为对照项。"
        if kv_time > 0:
            return "Round 8 可继续细拆 KV read/write/eviction，但只在其占比稳定较高时推进。"
        if vae_ratio >= 0.15:
            return "Round 8 可转向 VAE decode，model forward fine 数据作为背景。"
        if save_ratio >= 0.10:
            return "Round 8 可转向 video save/I-O 路径，尤其关注端到端耗时。"
        return "Round 8 继续按 sampled top phase 追加更细粒度 profiling。"

    def _record_attention_profiler_payload(
        self,
        payload: Dict[str, Any],
    ) -> None:
        max_events = int(self.config.attention_profiler_max_events)
        if max_events > 0 and len(self._attention_profiler_events) >= max_events:
            self._attention_profiler_max_reached = True
            return

        safe_payload = self._profiler_json_safe(payload)
        phase = str(safe_payload.get("phase", "unknown"))
        elapsed_ms = float(safe_payload.get("elapsed_ms", 0.0) or 0.0)
        stats = self._attention_profiler_phase_stats.setdefault(
            phase,
            {"elapsed_ms": 0.0, "count": 0, "values_ms": []})
        stats["elapsed_ms"] += elapsed_ms
        stats["count"] += 1
        stats["values_ms"].append(elapsed_ms)

        self._attention_add_breakdown(
            self._attention_profiler_layer_stats,
            phase,
            safe_payload.get("layer_idx"),
            elapsed_ms)
        self._attention_add_breakdown(
            self._attention_profiler_window_stats,
            phase,
            safe_payload.get("window_idx"),
            elapsed_ms)
        self._attention_add_breakdown(
            self._attention_profiler_branch_stats,
            phase,
            safe_payload.get("branch"),
            elapsed_ms)

        timer_type = str(safe_payload.get("timer_type") or "unknown")
        self._attention_profiler_timer_type_counts[timer_type] = (
            self._attention_profiler_timer_type_counts.get(timer_type, 0) + 1)
        flash_key = self._attention_flash_key(
            safe_payload.get("used_flash_attention"))
        self._attention_profiler_flash_counts[flash_key] = (
            self._attention_profiler_flash_counts.get(flash_key, 0) + 1)
        if safe_payload.get("fallback_reason"):
            self._attention_profiler_fallback_count += 1

        self._attention_profiler_events.append(safe_payload)
        self._write_attention_profiler_jsonl(safe_payload)

    def _attention_profiler_phase_enabled(
        self,
        phase: str,
        *,
        branch: Optional[str] = None,
    ) -> bool:
        phase = str(phase)
        if phase == "attention_forward_total":
            return True
        if phase == "clean_cache_update_attention":
            return self.config.attention_profiler_record_clean_cache_update
        mapping = {
            "rope": self.config.attention_profiler_record_rope,
            "qkv_or_input_projection": (
                self.config.attention_profiler_record_qkv),
            "kv_cache_read_or_assembly": (
                self.config.attention_profiler_record_cache_assembly),
            "padding_or_mask_setup": (
                self.config.attention_profiler_record_padding),
            "attention_kernel": self.config.attention_profiler_record_kernel,
            "output_projection": (
                self.config.attention_profiler_record_output_proj),
            "kv_cache_write": (
                self.config.attention_profiler_record_cache_write),
        }
        return bool(mapping.get(phase, False))

    def _attention_profiler_layer_selected(
        self,
        layer_idx: Optional[int],
    ) -> bool:
        layers = self.config.attention_profiler_sample_layers
        if not layers:
            return False
        if layer_idx is None:
            return False
        return int(layer_idx) in set(int(item) for item in layers)

    def _attention_profiler_window_selected(
        self,
        window_idx: Optional[int],
    ) -> bool:
        interval = max(
            1, int(self.config.attention_profiler_sample_every_n_steps))
        if window_idx is None:
            return True
        return int(window_idx) % interval == 0

    def _attention_shape_text(self, value: Optional[Any]) -> str:
        if value is None:
            return "none"
        if isinstance(value, str):
            return value
        if hasattr(value, "shape"):
            try:
                return _shape_to_string(value.shape)
            except Exception:
                return str(tuple(value.shape))
        if isinstance(value, (list, tuple)):
            try:
                return _shape_to_string(value)
            except Exception:
                return "x".join(str(item) for item in value)
        return str(value)

    def _attention_first_not_none(self, *values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _attention_avg_ms(self, stats: Dict[str, Any]) -> float:
        count = int(stats.get("count", 0) or 0)
        elapsed_ms = float(stats.get("elapsed_ms", 0.0) or 0.0)
        return elapsed_ms / count if count else 0.0

    def _attention_percentile(
        self,
        values: Iterable[Any],
        quantile: float,
    ) -> float:
        numeric_values = sorted(float(value) for value in values)
        if not numeric_values:
            return 0.0
        if len(numeric_values) == 1:
            return numeric_values[0]
        position = (len(numeric_values) - 1) * float(quantile)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return numeric_values[lower]
        weight = position - lower
        return (
            numeric_values[lower] * (1.0 - weight) +
            numeric_values[upper] * weight)

    def _attention_add_breakdown(
        self,
        target: Dict[str, Dict[str, Dict[str, Any]]],
        phase: str,
        key: Any,
        elapsed_ms: float,
    ) -> None:
        key_text = "none" if key is None else str(key)
        phase_stats = target.setdefault(phase, {})
        stats = phase_stats.setdefault(
            key_text,
            {"elapsed_ms": 0.0, "count": 0, "values_ms": []})
        stats["elapsed_ms"] += float(elapsed_ms)
        stats["count"] += 1
        stats["values_ms"].append(float(elapsed_ms))

    def _attention_format_breakdown(
        self,
        source: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        formatted: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for phase, groups in sorted(source.items()):
            formatted[phase] = {}
            for key, stats in sorted(groups.items(), key=lambda item: item[0]):
                count = int(stats.get("count", 0) or 0)
                elapsed_ms = float(stats.get("elapsed_ms", 0.0) or 0.0)
                values_ms = stats.get("values_ms", [])
                formatted[phase][key] = {
                    "total_ms": elapsed_ms,
                    "count": count,
                    "avg_ms": elapsed_ms / count if count else 0.0,
                    "p50_ms": self._attention_percentile(values_ms, 0.50),
                    "p90_ms": self._attention_percentile(values_ms, 0.90),
                }
        return formatted

    def _attention_top_bottleneck_phases(
        self,
        phase_total_ms: Dict[str, float],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        rows = [
            {
                "phase": phase,
                "total_ms": elapsed_ms,
                "inclusive": phase in {
                    "attention_forward_total",
                    "clean_cache_update_attention",
                },
            }
            for phase, elapsed_ms in phase_total_ms.items()
        ]
        rows.sort(key=lambda item: item["total_ms"], reverse=True)
        return rows[:limit]

    def _attention_next_step_recommendation(
        self,
        phase_total_ms: Dict[str, float],
        phase_time_ratio: Dict[str, float],
    ) -> str:
        if not phase_total_ms:
            return (
                "Round 8.1 should not optimize yet: no attention samples were "
                "collected. Check sample_layers, sample_every_n_steps, and "
                "max_events.")

        subphase_items = [
            (phase, elapsed)
            for phase, elapsed in phase_total_ms.items()
            if phase not in {
                "attention_forward_total",
                "clean_cache_update_attention",
            }
        ]
        subphase_items.sort(key=lambda item: item[1], reverse=True)
        top_phase = subphase_items[0][0] if subphase_items else None
        flash_false = self._attention_profiler_flash_counts.get("false", 0)
        flash_true = self._attention_profiler_flash_counts.get("true", 0)
        if flash_false > flash_true:
            return (
                "Round 8.1 should first explain flash-attn fallback before "
                "optimizing kernels or cache assembly.")
        if top_phase == "attention_kernel":
            return (
                "Round 8.1 should focus on the attention kernel path, after "
                "confirming flash-attn status and quality invariance.")
        if top_phase == "kv_cache_read_or_assembly":
            return (
                "Round 8.1 should focus on cache assembly/read reshape cost, "
                "not KV compression or eviction changes.")
        if top_phase == "padding_or_mask_setup":
            return (
                "Round 8.1 should focus on padding, unpadding, mask, and "
                "cu_seqlens setup overhead.")
        clean_ms = phase_total_ms.get("clean_cache_update_attention", 0.0)
        total_ms = phase_total_ms.get("attention_forward_total", 0.0)
        if total_ms > 0 and clean_ms / total_ms >= 0.35:
            return (
                "Round 8.1 should isolate clean_cache_update attention/KV "
                "work before changing the denoise path.")
        vae_sec = float(
            self._profiler_phase_stats.get(
                "vae_decode_total", {}).get("elapsed_sec", 0.0) or 0.0)
        save_sec = float(
            self._profiler_phase_stats.get(
                "video_save_total", {}).get("elapsed_sec", 0.0) or 0.0)
        model_sec = float(
            self._profiler_phase_stats.get(
                "model_forward_total", {}).get("elapsed_sec", 0.0) or 0.0)
        if model_sec > 0 and vae_sec >= model_sec * 0.30:
            return (
                "Attention samples do not show a dominant subphase; consider "
                "VAE decode only after Round 8.0 evidence is reviewed.")
        if model_sec > 0 and save_sec >= model_sec * 0.20:
            return (
                "Attention samples do not show a dominant subphase; video save "
                "may be a later end-to-end target.")
        return (
            "Round 8.1 should pause optimization until the 21-frame and "
            "81-frame summaries agree on the top attention subphase.")

    def _attention_flash_key(self, value: Any) -> str:
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "unknown"
        return str(value).lower()

    def _write_attention_profiler_jsonl(
        self,
        payload: Dict[str, Any],
    ) -> None:
        if (
            not self.config.attention_profiler_enabled or
            not self.config.attention_profiler_output_path
        ):
            return
        try:
            output_path = self.config.attention_profiler_output_path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    self._profiler_json_safe(dict(payload)),
                    sort_keys=True) + "\n")
        except Exception as exc:
            if not self._attention_profiler_warned_jsonl:
                self._warning(
                    f"failed to write attention profiler jsonl: {exc}")
                self._attention_profiler_warned_jsonl = True

    def _write_attention_profiler_summary(
        self,
        payload: Dict[str, Any],
    ) -> None:
        if (
            not self.config.attention_profiler_enabled or
            not self.config.attention_profiler_summary_path
        ):
            return
        try:
            output_path = self.config.attention_profiler_summary_path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(
                    self._profiler_json_safe(payload),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)
                handle.write("\n")
        except Exception as exc:
            if not self._attention_profiler_warned_summary:
                self._warning(
                    f"failed to write attention profiler summary: {exc}")
                self._attention_profiler_warned_summary = True

    def _write_profiler_jsonl(self, payload: Dict[str, Any]) -> None:
        if (
            not self.config.profiler_enabled or
            not self.config.profiler_output_path or
            not self.config.profiler_log_events
        ):
            return

        try:
            payload = self._profiler_json_safe(dict(payload))
            payload.setdefault("flowcache_run_id", self._flowcache_run_id)
            payload.setdefault("mode", self.config.profiler_mode)
            payload.setdefault(
                "cuda_synchronize",
                self.config.profiler_cuda_synchronize)
            payload.setdefault("fine_enabled", self.config.profiler_fine_enabled)
            payload.setdefault("warning_count", self._warning_count)
            output_path = self.config.profiler_output_path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            if not self._profiler_warned_jsonl:
                self._warning(f"failed to write profiler jsonl: {exc}")
                self._profiler_warned_jsonl = True

    def _write_profiler_summary(self, payload: Dict[str, Any]) -> None:
        if not self.config.profiler_enabled or not self.config.profiler_summary_path:
            return

        try:
            output_path = self.config.profiler_summary_path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(
                    self._profiler_json_safe(payload),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)
                handle.write("\n")
        except Exception as exc:
            if not self._profiler_warned_summary:
                self._warning(f"failed to write profiler summary: {exc}")
                self._profiler_warned_summary = True

    def _profiler_json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(key): self._profiler_json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._profiler_json_safe(item) for item in value]
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return f"<tensor_like shape={tuple(value.shape)} dtype={value.dtype}>"
        return str(value)

    def _profiler_sync(self, device: Optional[Any] = None) -> None:
        if not self.config.profiler_cuda_synchronize:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device if device else None)
        except Exception as exc:
            self._warning(f"profiler cuda synchronize failed: {exc}")

    def _profiler_reset_peak_memory(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            self._warning(f"profiler reset peak memory failed: {exc}")

    def _profiler_peak_memory(self) -> tuple:
        try:
            import torch
            if torch.cuda.is_available():
                scale = 1024 ** 3
                return (
                    torch.cuda.max_memory_allocated() / scale,
                    torch.cuda.max_memory_reserved() / scale,
                )
        except Exception as exc:
            self._warning(f"profiler peak memory read failed: {exc}")
        return None, None

    def _profiler_layer_selected(self, layer_idx: Optional[int]) -> bool:
        layers = self.config.profiler_sample_layers
        if not layers:
            return False
        if layer_idx is None:
            return False
        return int(layer_idx) in set(int(item) for item in layers)

    def _profiler_step_selected(self, window_index: Optional[int]) -> bool:
        interval = max(1, int(self.config.profiler_sample_every_n_steps))
        if window_index is None:
            return True
        return int(window_index) % interval == 0

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        if not self.config.metadata_output_path:
            return

        try:
            payload = dict(payload)
            payload.setdefault("flowcache_run_id", self._flowcache_run_id)
            output_path = self.config.metadata_output_path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            if not self._warned_jsonl:
                self._warning(f"failed to write metadata jsonl: {exc}")
                self._warned_jsonl = True

    def _write_r0_jsonl(self, payload: Dict[str, Any]) -> None:
        output_path = self.config.r0_trace_output_path
        if not output_path:
            raise ValueError(
                "r0_trace_output_path is required when r0_trace_enabled=true")

        payload = dict(payload)
        payload.setdefault("flowcache_run_id", self._flowcache_run_id)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _warning(self, message: str) -> None:
        self._warning_count += 1
        if self.config.enabled or self.config.profiler_enabled:
            print(f"[FlowCache][warning] {message}")


class _FlowCacheProfilerScope:
    def __init__(
        self,
        manager: FlowCacheManager,
        phase: str,
        *,
        prompt_idx: Optional[int] = None,
        sample_idx: Optional[int] = None,
        window_index: Optional[int] = None,
        layer_idx: Optional[int] = None,
        count: int = 1,
        device: Optional[Any] = None,
        fallback_reason: Optional[str] = None,
    ):
        self.manager = manager
        self.phase = phase
        self.prompt_idx = prompt_idx
        self.sample_idx = sample_idx
        self.window_index = window_index
        self.layer_idx = layer_idx
        self.count = count
        self.device = device
        self.fallback_reason = fallback_reason
        self.token: Optional[Dict[str, Any]] = None

    def __enter__(self):
        self.token = self.manager.start_profiler_phase(
            self.phase,
            prompt_idx=self.prompt_idx,
            sample_idx=self.sample_idx,
            window_index=self.window_index,
            layer_idx=self.layer_idx,
            count=self.count,
            device=self.device,
            fallback_reason=self.fallback_reason,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        fallback_reason = self.fallback_reason
        if exc_type is not None and fallback_reason is None:
            fallback_reason = f"exception:{exc_type.__name__}"
        self.manager.end_profiler_phase(
            self.token,
            count=self.count,
            fallback_reason=fallback_reason,
        )
        return False


def _get_value(raw_config: Any, key: str, default: Any) -> Any:
    if raw_config is None:
        return default
    if isinstance(raw_config, dict):
        return raw_config.get(key, default)
    try:
        return getattr(raw_config, key)
    except (AttributeError, TypeError):
        return default


def _get_bool(raw_config: Any, key: str, default: bool) -> bool:
    value = _get_value(raw_config, key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", "none", "null", ""}:
            return False
    return bool(value)


def _get_float(raw_config: Any, key: str, default: float) -> float:
    return float(_get_value(raw_config, key, default))


def _rng_state_sha256(state: Any) -> str:
    """Hash a torch RNG state without changing it or writing raw bytes to disk."""
    if state is None:
        return "none"
    tensor = state.detach() if hasattr(state, "detach") else state
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    if hasattr(tensor, "numpy"):
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return hashlib.sha256(bytes(tensor)).hexdigest()


def _get_int(raw_config: Any, key: str, default: int) -> int:
    return int(_get_value(raw_config, key, default))


def _get_str(raw_config: Any, key: str, default: str) -> str:
    value = _get_value(raw_config, key, default)
    if value in (None, "none", "None", "null", "Null", ""):
        return default
    return str(value)


def _get_float_tuple(raw_config: Any, key: str, default: Iterable[float]) -> tuple:
    value = _get_value(raw_config, key, default)
    if value in (None, "none", "None", "null", "Null", ""):
        return tuple(float(item) for item in default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return tuple(float(item) for item in default)
        return tuple(float(part) for part in parts)
    if isinstance(value, (list, tuple, set)):
        return tuple(float(item) for item in value)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return tuple(float(item) for item in value)
    return (float(value),)


def _get_optional_str(raw_config: Any, key: str, default: Optional[str]) -> Optional[str]:
    value = _get_value(raw_config, key, default)
    if value in (None, "none", "None", "null", "Null", ""):
        return None
    return str(value)


def _get_optional_int_value(raw_config: Any, key: str, default: Optional[int]) -> Optional[int]:
    value = _get_value(raw_config, key, default)
    if value in (None, "none", "None", "null", "Null", ""):
        return None
    return int(value)


def _get_optional_int_filter(raw_config: Any, key: str, default: Optional[Any]) -> Optional[tuple]:
    value = _get_value(raw_config, key, default)
    if value in (None, "none", "None", "null", "Null", ""):
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return None
        return tuple(int(part) for part in parts)
    if isinstance(value, (list, tuple, set)):
        return tuple(int(item) for item in value)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _filter_contains(filter_values: Optional[Any], value: Optional[int]) -> bool:
    if filter_values is None:
        return True
    if value is None:
        return False
    return int(value) in set(filter_values)


def _counter_dict(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _format_count_dict(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return "|".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "none"
    return f"{float(value):.6e}"


def _format_threshold_key(value: Any) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _format_ratio_dict(values: Dict[str, float]) -> str:
    if not values:
        return "none"
    return "|".join(f"{key}:{values[key]:.6f}" for key in sorted(values))


def _shape_to_tuple(shape: Optional[Iterable[int]]) -> Optional[tuple]:
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _shape_to_string(shape: Optional[Iterable[int]]) -> str:
    if shape is None:
        return "none"
    return "x".join(str(int(item)) for item in shape)


def _to_list(value: Any) -> list:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _scalar_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return _scalar_to_int(value)


def _dict_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _count_unique(values: Iterable[Any]) -> int:
    return len(set(values))


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
