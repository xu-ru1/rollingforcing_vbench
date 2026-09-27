from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
# from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
# from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import time
import torch.distributed as dist

from utils.step_cache_sparse import block_token_indices, block_frame_indices, apply_reuse_residuals

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
# flex_attention = torch.compile(
#     flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def _flowcache_should_record_attention(flowcache_manager):
    return bool(
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_candidates", False)
    )


def _flowcache_start_profile(
    flowcache_manager,
    phase,
    *,
    window_index=None,
    layer_idx=None,
    device=None,
):
    if flowcache_manager is None:
        return None
    should_profile = getattr(flowcache_manager, "should_profile_phase", None)
    if should_profile is None:
        return None
    if not should_profile(
        phase,
        window_index=window_index,
        layer_idx=layer_idx,
    ):
        return None
    return flowcache_manager.start_profiler_phase(
        phase,
        window_index=window_index,
        layer_idx=layer_idx,
        device=device,
    )


def _flowcache_end_profile(flowcache_manager, token):
    if flowcache_manager is not None and token is not None:
        flowcache_manager.end_profiler_phase(token)


def _flowcache_start_attention_profile(
    flowcache_manager,
    phase,
    *,
    window_index=None,
    layer_idx=None,
    branch=None,
    device=None,
    visible_tokens=None,
    history_tokens=None,
    current_tokens=None,
    sink_tokens=None,
    q_shape=None,
    k_shape=None,
    v_shape=None,
    dtype=None,
    used_flash_attention=None,
    fallback_reason=None,
    warning=None,
    timing_scope="subphase",
):
    if flowcache_manager is None:
        return None
    if not getattr(flowcache_manager, "attention_profiler_enabled", False):
        return None
    start = getattr(flowcache_manager, "start_attention_profiler_phase", None)
    if start is None:
        return None
    return start(
        phase,
        window_idx=window_index,
        layer_idx=layer_idx,
        branch=branch,
        device=device,
        visible_tokens=visible_tokens,
        history_tokens=history_tokens,
        current_tokens=current_tokens,
        sink_tokens=sink_tokens,
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        dtype=dtype,
        used_flash_attention=used_flash_attention,
        fallback_reason=fallback_reason,
        warning=warning,
        timing_scope=timing_scope,
    )


def _flowcache_end_attention_profile(
    flowcache_manager,
    token,
    *,
    branch=None,
    device=None,
    visible_tokens=None,
    history_tokens=None,
    current_tokens=None,
    sink_tokens=None,
    q_shape=None,
    k_shape=None,
    v_shape=None,
    dtype=None,
    used_flash_attention=None,
    fallback_reason=None,
    warning=None,
):
    if flowcache_manager is None or token is None:
        return
    end = getattr(flowcache_manager, "end_attention_profiler_phase", None)
    if end is None:
        return
    end(
        token,
        branch=branch,
        device=device,
        visible_tokens=visible_tokens,
        history_tokens=history_tokens,
        current_tokens=current_tokens,
        sink_tokens=sink_tokens,
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        dtype=dtype,
        used_flash_attention=used_flash_attention,
        fallback_reason=fallback_reason,
        warning=warning,
    )


def _flowcache_profiled_attention(
    flowcache_manager,
    query,
    key,
    value,
    *,
    window_index=None,
    layer_idx=None,
    branch=None,
    visible_tokens=None,
    history_tokens=None,
    current_tokens=None,
    sink_tokens=None,
):
    token = _flowcache_start_profile(
        flowcache_manager,
        "attention_compute_sampled",
        window_index=window_index,
        layer_idx=layer_idx,
        device=query.device,
    )
    try:
        attention_profiler = (
            flowcache_manager
            if getattr(flowcache_manager, "attention_profiler_enabled", False)
            else None
        )
        metadata = None
        if attention_profiler is not None:
            metadata = {
                "branch": branch,
                "window_idx": window_index,
                "layer_idx": layer_idx,
                "visible_tokens": visible_tokens,
                "history_tokens": history_tokens,
                "current_tokens": current_tokens,
                "sink_tokens": sink_tokens,
                "q_shape": "x".join(str(int(item)) for item in query.shape),
                "k_shape": "x".join(str(int(item)) for item in key.shape),
                "v_shape": "x".join(str(int(item)) for item in value.shape),
                "dtype": str(value.dtype),
                "device": query.device,
            }
        return attention(
            query,
            key,
            value,
            flowcache_attention_profiler=attention_profiler,
            flowcache_attention_metadata=metadata,
            flop_tag="self_attention",
        )
    finally:
        _flowcache_end_profile(flowcache_manager, token)


def _flowcache_record_attention_parts(
    flowcache_manager,
    *,
    event,
    window_index,
    layer_idx,
    attention_branch,
    updating_cache,
    block_length,
    sink_tokens,
    anchor_tokens,
    anchor_value_tokens,
    working_tokens,
    working_value_tokens,
    current_tokens,
    current_value_tokens,
    input_tokens,
    input_value_tokens,
    current_start,
    cache_start,
    cache_end,
    global_end_index,
    local_end_index,
):
    if not _flowcache_should_record_attention(flowcache_manager):
        return

    flowcache_manager.record_attention_parts(
        event=event or attention_branch,
        window_index=window_index,
        layer_idx=layer_idx,
        attention_branch=attention_branch,
        updating_cache=updating_cache,
        block_length=block_length,
        sink_tokens=sink_tokens,
        anchor_tokens=anchor_tokens,
        anchor_value_tokens=anchor_value_tokens,
        working_tokens=working_tokens,
        working_value_tokens=working_value_tokens,
        current_tokens=current_tokens,
        current_value_tokens=current_value_tokens,
        input_tokens=input_tokens,
        input_value_tokens=input_value_tokens,
        current_start=current_start,
        cache_start=cache_start,
        cache_end=cache_end,
        global_end_index=global_end_index,
        local_end_index=local_end_index,
    )


def _flowcache_real_plan(
    flowcache_manager,
    *,
    window_index,
    layer_idx,
    branch,
    source_event,
    protected_sink_tokens,
    protected_current_tokens,
    original_history_tokens,
    original_total_kv_tokens,
    candidate_region_start,
    candidate_region_end,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_run_real_compression", False)
    ):
        return None

    try:
        return flowcache_manager.build_real_compression_plan(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            source_event=source_event or branch,
            protected_sink_tokens=protected_sink_tokens,
            protected_current_tokens=protected_current_tokens,
            original_history_tokens=original_history_tokens,
            original_total_kv_tokens=original_total_kv_tokens,
            candidate_region_start=candidate_region_start,
            candidate_region_end=candidate_region_end,
        )
    except Exception as exc:
        if hasattr(flowcache_manager, "_warning"):
            flowcache_manager._warning(
                "build real compression plan failed for "
                f"window_index={window_index} layer_idx={layer_idx}: {exc}")
        return None


def _flowcache_record_real_compression(flowcache_manager, plan, *, fallback=False, skipped_reason=None):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_run_real_compression", False) and
        plan is not None
    ):
        return

    payload = dict(plan)
    payload["fallback"] = fallback
    if skipped_reason is not None:
        payload["applied"] = False
        payload["skipped_reason"] = skipped_reason
        payload["compressed_history_tokens"] = payload["original_history_tokens"]
        payload["compressed_total_kv_tokens"] = payload["original_total_kv_tokens"]
        payload["saved_tokens"] = 0
        payload["saving_ratio"] = 0.0
        payload["keep_ratio"] = 1.0
    flowcache_manager.record_real_compression(**payload)


def _flowcache_cache_body_plan(
    flowcache_manager,
    *,
    window_index,
    layer_idx,
    branch,
    source_event,
    protected_sink_tokens,
    protected_current_tokens,
    original_history_tokens,
    original_visible_tokens,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_cache_body_compression", False)
    ):
        return None

    try:
        return flowcache_manager.build_cache_body_compression_plan(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            source_event=source_event or branch,
            protected_sink_tokens=protected_sink_tokens,
            protected_current_tokens=protected_current_tokens,
            original_history_tokens=original_history_tokens,
            original_visible_tokens=original_visible_tokens,
        )
    except Exception as exc:
        if hasattr(flowcache_manager, "_warning"):
            flowcache_manager._warning(
                "build cache-body compression plan failed for "
                f"window_index={window_index} layer_idx={layer_idx}: {exc}")
        return None


def _flowcache_record_cache_body_compression(
    flowcache_manager,
    plan,
    *,
    fallback=False,
    skipped_reason=None,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_cache_body_compression", False) and
        plan is not None
    ):
        return

    payload = dict(plan)
    payload["fallback"] = fallback
    if fallback or skipped_reason is not None:
        payload["applied"] = False
        payload["skipped_reason"] = skipped_reason or payload.get("skipped_reason")
        payload["compressed_history_tokens"] = payload["original_history_tokens"]
        payload["compressed_visible_tokens"] = payload["original_visible_tokens"]
        payload["saved_visible_tokens"] = 0
        payload["visible_saving_ratio"] = 0.0
        payload["keep_ratio"] = 1.0
    flowcache_manager.record_cache_body_compression(**payload)


def _flowcache_persistent_cache_plan(
    flowcache_manager,
    *,
    window_index,
    layer_idx,
    branch,
    source_event,
    protected_sink_tokens,
    protected_current_tokens,
    original_history_tokens,
    original_visible_tokens,
    source_start,
    source_end,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_persistent_cache_compression", False)
    ):
        return None

    try:
        return flowcache_manager.build_persistent_cache_compression_plan(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            source_event=source_event or branch,
            protected_sink_tokens=protected_sink_tokens,
            protected_current_tokens=protected_current_tokens,
            original_history_tokens=original_history_tokens,
            original_visible_tokens=original_visible_tokens,
            source_start=source_start,
            source_end=source_end,
        )
    except Exception as exc:
        if hasattr(flowcache_manager, "_warning"):
            flowcache_manager._warning(
                "build persistent-cache compression plan failed for "
                f"window_index={window_index} layer_idx={layer_idx}: {exc}")
        return None


def _flowcache_record_persistent_cache_compression(
    flowcache_manager,
    plan,
    *,
    fallback=False,
    skipped_reason=None,
    sidecar_created=False,
    sidecar_reused=False,
    sidecar_invalidated=False,
    sidecar_fallback=False,
    sidecar_valid=False,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_persistent_cache_compression", False) and
        plan is not None
    ):
        return

    payload = dict(plan)
    payload["fallback"] = fallback
    payload["sidecar_created"] = sidecar_created
    payload["sidecar_reused"] = sidecar_reused
    payload["sidecar_invalidated"] = sidecar_invalidated
    payload["sidecar_fallback"] = sidecar_fallback
    payload["sidecar_valid"] = sidecar_valid
    if fallback or skipped_reason is not None:
        payload["applied"] = False
        payload["skipped_reason"] = skipped_reason or payload.get("skipped_reason")
        payload["compressed_history_tokens"] = payload["original_history_tokens"]
        payload["compressed_visible_tokens"] = payload["original_visible_tokens"]
        payload["saved_visible_tokens"] = 0
        payload["visible_saving_ratio"] = 0.0
        payload["keep_ratio"] = 1.0
    flowcache_manager.record_persistent_cache_compression(**payload)


def _flowcache_compacted_kv_plan(
    flowcache_manager,
    *,
    window_index,
    layer_idx,
    branch,
    source_event,
    protected_sink_tokens,
    protected_current_tokens,
    original_history_tokens,
    original_visible_tokens,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_compacted_kv", False)
    ):
        return None

    try:
        return flowcache_manager.build_compacted_kv_plan(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            source_event=source_event or branch,
            protected_sink_tokens=protected_sink_tokens,
            protected_current_tokens=protected_current_tokens,
            original_history_tokens=original_history_tokens,
            original_visible_tokens=original_visible_tokens,
        )
    except Exception as exc:
        if hasattr(flowcache_manager, "_warning"):
            flowcache_manager._warning(
                "build compacted-kv plan failed for "
                f"window_index={window_index} layer_idx={layer_idx}: {exc}")
        return None


def _flowcache_record_compacted_kv(
    flowcache_manager,
    plan,
    *,
    fallback=False,
    fallback_reason=None,
    skipped_reason=None,
    dtype=None,
    device=None,
    elapsed_ms=None,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_compacted_kv", False) and
        plan is not None
    ):
        return

    payload = dict(plan)
    payload["fallback"] = fallback
    payload["fallback_reason"] = fallback_reason
    if dtype is not None:
        payload["dtype"] = str(dtype)
    if device is not None:
        payload["device"] = str(device)
    if elapsed_ms is not None:
        payload["elapsed_ms"] = float(elapsed_ms)
    if fallback or skipped_reason is not None:
        payload["applied"] = False
        payload["skipped_reason"] = skipped_reason or payload.get("skipped_reason")
        payload["compacted_history_tokens"] = payload["original_history_tokens"]
        payload["compacted_visible_tokens"] = payload["original_visible_tokens"]
        payload["saved_visible_tokens"] = 0
        payload["visible_saving_ratio"] = 0.0
        payload["keep_ratio"] = 1.0
    flowcache_manager.record_compacted_kv(**payload)


def _flowcache_persistent_sidecar_tokens(kv_cache):
    tensor = kv_cache.get("flowcache_compressed_history_key")
    if tensor is None:
        return 0
    return int(tensor.shape[1])


def _flowcache_persistent_sidecar_metadata(kv_cache):
    key_tensor = kv_cache.get("flowcache_compressed_history_key")
    value_tensor = kv_cache.get("flowcache_compressed_history_value")
    metadata = {
        "valid": bool(kv_cache.get("flowcache_compressed_history_valid", False)),
        "layer_idx": kv_cache.get("flowcache_compressed_history_layer_idx"),
        "source_start": kv_cache.get("flowcache_compressed_history_source_start"),
        "source_end": kv_cache.get("flowcache_compressed_history_source_end"),
        "window_index": kv_cache.get("flowcache_compressed_history_window_index"),
        "target_ratio": kv_cache.get("flowcache_compressed_history_target_ratio"),
        "strategy": kv_cache.get("flowcache_compressed_history_strategy"),
        "compressed_tokens": kv_cache.get("flowcache_compressed_history_tokens"),
        "global_end_index": kv_cache.get("flowcache_compressed_history_global_end_index"),
        "local_end_index": kv_cache.get("flowcache_compressed_history_local_end_index"),
    }
    if key_tensor is not None:
        metadata["device"] = str(key_tensor.device)
        metadata["dtype"] = str(key_tensor.dtype)
    if value_tensor is not None:
        metadata["value_device"] = str(value_tensor.device)
        metadata["value_dtype"] = str(value_tensor.dtype)
    return metadata


def _flowcache_record_persistent_sidecar(
    flowcache_manager,
    *,
    action,
    window_index,
    layer_idx,
    source_start,
    source_end,
    compressed_tokens,
    reason,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_persistent_cache_compression", False)
    ):
        return
    flowcache_manager.record_persistent_cache_sidecar(
        action=action,
        window_index=window_index,
        layer_idx=layer_idx,
        source_start=source_start,
        source_end=source_end,
        compressed_tokens=compressed_tokens,
        reason=reason,
    )


def _flowcache_invalidate_persistent_sidecar(
    flowcache_manager,
    kv_cache,
    *,
    window_index,
    layer_idx,
    reason,
    write_start=None,
    write_end=None,
    eviction_happened=False,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_track_persistent_cache_compression", False)
    ):
        return False
    if not kv_cache.get("flowcache_compressed_history_valid", False):
        return False

    source_start = kv_cache.get("flowcache_compressed_history_source_start")
    source_end = kv_cache.get("flowcache_compressed_history_source_end")
    compressed_tokens = _flowcache_persistent_sidecar_tokens(kv_cache)
    if hasattr(flowcache_manager, "should_invalidate_persistent_cache_sidecar_for_write"):
        should_invalidate, scoped_reason = (
            flowcache_manager.should_invalidate_persistent_cache_sidecar_for_write(
                _flowcache_persistent_sidecar_metadata(kv_cache),
                write_start=write_start,
                write_end=write_end,
                eviction_happened=eviction_happened,
            )
        )
    else:
        should_invalidate = True
        scoped_reason = reason

    if not should_invalidate:
        _flowcache_record_persistent_sidecar(
            flowcache_manager,
            action="keep_valid",
            window_index=window_index,
            layer_idx=layer_idx,
            source_start=source_start,
            source_end=source_end,
            compressed_tokens=compressed_tokens,
            reason=scoped_reason or "cache_write_did_not_affect_source",
        )
        return False

    kv_cache["flowcache_compressed_history_valid"] = False
    kv_cache["flowcache_compressed_history_key"] = None
    kv_cache["flowcache_compressed_history_value"] = None
    _flowcache_record_persistent_sidecar(
        flowcache_manager,
        action="invalidate",
        window_index=window_index,
        layer_idx=layer_idx,
        source_start=source_start,
        source_end=source_end,
        compressed_tokens=compressed_tokens,
        reason=scoped_reason or reason,
    )
    return True


def _flowcache_store_persistent_sidecar(
    flowcache_manager,
    kv_cache,
    compressed_history_key,
    compressed_history_value,
    plan,
    *,
    window_index,
    layer_idx,
    source_start,
    source_end,
    global_end_index,
    local_end_index,
    reason,
):
    keep_tokens = int(compressed_history_key.shape[1])
    kv_cache["flowcache_compressed_history_key"] = compressed_history_key
    kv_cache["flowcache_compressed_history_value"] = compressed_history_value
    kv_cache["flowcache_compressed_history_start"] = 0
    kv_cache["flowcache_compressed_history_end"] = keep_tokens
    kv_cache["flowcache_compressed_history_source_start"] = int(source_start)
    kv_cache["flowcache_compressed_history_source_end"] = int(source_end)
    kv_cache["flowcache_compressed_history_window_index"] = (
        None if window_index is None else int(window_index)
    )
    kv_cache["flowcache_compressed_history_layer_idx"] = (
        None if layer_idx is None else int(layer_idx)
    )
    kv_cache["flowcache_compressed_history_target_ratio"] = float(
        plan["target_ratio"])
    kv_cache["flowcache_compressed_history_strategy"] = str(plan["strategy"])
    kv_cache["flowcache_compressed_history_tokens"] = keep_tokens
    kv_cache["flowcache_compressed_history_global_end_index"] = int(
        global_end_index)
    kv_cache["flowcache_compressed_history_local_end_index"] = int(
        local_end_index)
    kv_cache["flowcache_compressed_history_valid"] = True

    _flowcache_record_persistent_sidecar(
        flowcache_manager,
        action="create",
        window_index=window_index,
        layer_idx=layer_idx,
        source_start=source_start,
        source_end=source_end,
        compressed_tokens=keep_tokens,
        reason=reason,
    )


def _flowcache_apply_persistent_history_sidecar(
    flowcache_manager,
    kv_cache,
    plan,
    history_key,
    history_value,
    *,
    window_index,
    layer_idx,
    source_start,
    source_end,
    global_end_index,
    local_end_index,
):
    flags = {
        "sidecar_created": False,
        "sidecar_reused": False,
        "sidecar_invalidated": False,
        "sidecar_fallback": False,
        "sidecar_valid": False,
        "fallback": False,
        "skipped_reason": None,
    }

    try:
        history_tokens = int(history_key.shape[1])
        keep_tokens = int(plan["compressed_history_tokens"])
        if history_key.shape[1] != history_value.shape[1]:
            raise ValueError(
                "persistent sidecar key/value token length mismatch: "
                f"k={history_key.shape[1]} v={history_value.shape[1]}")
        if keep_tokens <= 0 or keep_tokens > history_tokens:
            raise ValueError(
                f"invalid keep_tokens={keep_tokens} for history_tokens={history_tokens}")

        existing_key = kv_cache.get("flowcache_compressed_history_key")
        existing_value = kv_cache.get("flowcache_compressed_history_value")
        metadata = _flowcache_persistent_sidecar_metadata(kv_cache)

        if existing_key is not None or existing_value is not None:
            valid, reason = flowcache_manager.validate_persistent_cache_sidecar_metadata(
                metadata,
                layer_idx=layer_idx,
                source_start=source_start,
                source_end=source_end,
                target_ratio=plan["target_ratio"],
                strategy=plan["strategy"],
                original_history_tokens=history_tokens,
                compressed_tokens=keep_tokens,
                device=history_key.device,
                dtype=history_key.dtype,
            )
            if valid and existing_value is not None:
                if existing_key.shape[1] != existing_value.shape[1]:
                    valid = False
                    reason = "sidecar_key_value_length_mismatch"
                elif existing_key.device != history_key.device:
                    valid = False
                    reason = "key_device_mismatch"
                elif existing_value.device != history_value.device:
                    valid = False
                    reason = "value_device_mismatch"
                elif existing_key.dtype != history_key.dtype:
                    valid = False
                    reason = "key_dtype_mismatch"
                elif existing_value.dtype != history_value.dtype:
                    valid = False
                    reason = "value_dtype_mismatch"

            if valid and existing_key is not None and existing_value is not None:
                _flowcache_verify_finite_tensors(
                    flowcache_manager,
                    [
                        ("persistent_reuse_key", existing_key),
                        ("persistent_reuse_value", existing_value),
                    ],
                    feature="persistent_cache",
                )
                flags["sidecar_reused"] = True
                flags["sidecar_valid"] = True
                _flowcache_record_persistent_sidecar(
                    flowcache_manager,
                    action="reuse",
                    window_index=window_index,
                    layer_idx=layer_idx,
                    source_start=source_start,
                    source_end=source_end,
                    compressed_tokens=int(existing_key.shape[1]),
                    reason="metadata_valid",
                )
                return existing_key, existing_value, flags

            prefix_reason = None
            if (
                reason == "source_end_mismatch" and
                existing_key is not None and
                existing_value is not None and
                kv_cache.get("flowcache_compressed_history_valid", False)
            ):
                old_source_start = metadata.get("source_start")
                old_source_end = metadata.get("source_end")
                old_source_start_int = (
                    None if old_source_start is None else int(old_source_start)
                )
                old_source_end_int = (
                    None if old_source_end is None else int(old_source_end)
                )
                old_compressed_tokens = int(existing_key.shape[1])
                old_history_tokens = (
                    0 if old_source_start_int is None or old_source_end_int is None
                    else max(0, old_source_end_int - old_source_start_int)
                )
                prefix_valid, prefix_reason = (
                    flowcache_manager.validate_persistent_cache_sidecar_metadata(
                        metadata,
                        layer_idx=layer_idx,
                        source_start=old_source_start,
                        source_end=old_source_end,
                        target_ratio=plan["target_ratio"],
                        strategy=plan["strategy"],
                        original_history_tokens=old_history_tokens,
                        compressed_tokens=old_compressed_tokens,
                        device=history_key.device,
                        dtype=history_key.dtype,
                    )
                )

                if (
                    prefix_valid and
                    old_source_start_int == int(source_start) and
                    old_source_end_int is not None and
                    int(source_start) < old_source_end_int < int(source_end)
                ):
                    if existing_key.shape[1] != existing_value.shape[1]:
                        prefix_valid = False
                        prefix_reason = "prefix_key_value_length_mismatch"
                    elif existing_key.device != history_key.device:
                        prefix_valid = False
                        prefix_reason = "prefix_key_device_mismatch"
                    elif existing_value.device != history_value.device:
                        prefix_valid = False
                        prefix_reason = "prefix_value_device_mismatch"
                    elif existing_key.dtype != history_key.dtype:
                        prefix_valid = False
                        prefix_reason = "prefix_key_dtype_mismatch"
                    elif existing_value.dtype != history_value.dtype:
                        prefix_valid = False
                        prefix_reason = "prefix_value_dtype_mismatch"

                if (
                    prefix_valid and
                    old_source_start_int == int(source_start) and
                    old_source_end_int is not None and
                    int(source_start) < old_source_end_int < int(source_end)
                ):
                    delta_start = old_source_end_int - int(source_start)
                    delta_tokens = int(source_end) - old_source_end_int
                    delta_keep_tokens = keep_tokens - old_compressed_tokens
                    if delta_start < 0 or delta_start > history_tokens:
                        prefix_valid = False
                        prefix_reason = "prefix_delta_start_out_of_range"
                    elif delta_tokens <= 0:
                        prefix_valid = False
                        prefix_reason = "prefix_delta_empty"
                    elif delta_keep_tokens < 0:
                        prefix_valid = False
                        prefix_reason = "prefix_compressed_exceeds_target_keep"
                    elif delta_keep_tokens > delta_tokens:
                        prefix_valid = False
                        prefix_reason = "prefix_delta_keep_exceeds_delta"

                if (
                    prefix_valid and
                    old_source_start_int == int(source_start) and
                    old_source_end_int is not None and
                    int(source_start) < old_source_end_int < int(source_end)
                ):
                    delta_start = old_source_end_int - int(source_start)
                    delta_end = int(source_end) - int(source_start)
                    delta_keep_tokens = keep_tokens - old_compressed_tokens
                    if delta_keep_tokens > 0:
                        delta_key = history_key[:, delta_start:delta_end]
                        delta_value = history_value[:, delta_start:delta_end]
                        delta_indices = _flowcache_history_keep_indices(
                            delta_key.shape[1],
                            delta_keep_tokens,
                            history_key.device)
                        compressed_delta_key = delta_key.index_select(
                            1, delta_indices)
                        compressed_delta_value = delta_value.index_select(
                            1, delta_indices)
                        compressed_history_key = torch.cat(
                            [existing_key, compressed_delta_key], dim=1).contiguous()
                        compressed_history_value = torch.cat(
                            [existing_value, compressed_delta_value], dim=1).contiguous()
                    else:
                        compressed_history_key = existing_key[:, :keep_tokens].contiguous()
                        compressed_history_value = existing_value[:, :keep_tokens].contiguous()

                    if compressed_history_key.shape[1] != plan["compressed_history_tokens"]:
                        raise ValueError(
                            "persistent prefix-reuse key length mismatch: "
                            f"actual={compressed_history_key.shape[1]} "
                            f"planned={plan['compressed_history_tokens']}")
                    if compressed_history_value.shape[1] != plan["compressed_history_tokens"]:
                        raise ValueError(
                            "persistent prefix-reuse value length mismatch: "
                            f"actual={compressed_history_value.shape[1]} "
                            f"planned={plan['compressed_history_tokens']}")

                    _flowcache_verify_finite_tensors(
                        flowcache_manager,
                        [
                            ("persistent_prefix_key", existing_key),
                            ("persistent_prefix_value", existing_value),
                            ("persistent_prefix_extended_key", compressed_history_key),
                            ("persistent_prefix_extended_value", compressed_history_value),
                        ],
                        feature="persistent_cache",
                    )
                    flags["sidecar_reused"] = True
                    flags["sidecar_created"] = True
                    flags["sidecar_valid"] = True
                    _flowcache_record_persistent_sidecar(
                        flowcache_manager,
                        action="reuse",
                        window_index=window_index,
                        layer_idx=layer_idx,
                        source_start=old_source_start_int,
                        source_end=old_source_end_int,
                        compressed_tokens=old_compressed_tokens,
                        reason="prefix_source_reused",
                    )
                    _flowcache_store_persistent_sidecar(
                        flowcache_manager,
                        kv_cache,
                        compressed_history_key,
                        compressed_history_value,
                        plan,
                        window_index=window_index,
                        layer_idx=layer_idx,
                        source_start=source_start,
                        source_end=source_end,
                        global_end_index=global_end_index,
                        local_end_index=local_end_index,
                        reason="extended_from_prefix_reuse",
                    )
                    return compressed_history_key, compressed_history_value, flags

            if kv_cache.get("flowcache_compressed_history_valid", False):
                flags["sidecar_invalidated"] = True
                existing_compressed_tokens = _flowcache_persistent_sidecar_tokens(kv_cache)
                kv_cache["flowcache_compressed_history_valid"] = False
                kv_cache["flowcache_compressed_history_key"] = None
                kv_cache["flowcache_compressed_history_value"] = None
                _flowcache_record_persistent_sidecar(
                    flowcache_manager,
                    action="invalidate",
                    window_index=window_index,
                    layer_idx=layer_idx,
                    source_start=metadata.get("source_start"),
                    source_end=metadata.get("source_end"),
                    compressed_tokens=existing_compressed_tokens,
                    reason=prefix_reason or reason or "metadata_invalid",
                )

        keep_indices = _flowcache_history_keep_indices(
            history_tokens, keep_tokens, history_key.device)
        compressed_history_key = history_key.index_select(
            1, keep_indices).contiguous()
        compressed_history_value = history_value.index_select(
            1, keep_indices).contiguous()

        if compressed_history_key.shape[1] != plan["compressed_history_tokens"]:
            raise ValueError(
                "persistent sidecar compressed key length mismatch: "
                f"actual={compressed_history_key.shape[1]} "
                f"planned={plan['compressed_history_tokens']}")
        if compressed_history_value.shape[1] != plan["compressed_history_tokens"]:
            raise ValueError(
                "persistent sidecar compressed value length mismatch: "
                f"actual={compressed_history_value.shape[1]} "
                f"planned={plan['compressed_history_tokens']}")

        _flowcache_verify_finite_tensors(
            flowcache_manager,
            [
                ("persistent_create_key", compressed_history_key),
                ("persistent_create_value", compressed_history_value),
            ],
            feature="persistent_cache",
        )

        kv_cache["flowcache_compressed_history_key"] = compressed_history_key
        kv_cache["flowcache_compressed_history_value"] = compressed_history_value
        kv_cache["flowcache_compressed_history_start"] = 0
        kv_cache["flowcache_compressed_history_end"] = keep_tokens
        kv_cache["flowcache_compressed_history_source_start"] = int(source_start)
        kv_cache["flowcache_compressed_history_source_end"] = int(source_end)
        kv_cache["flowcache_compressed_history_window_index"] = (
            None if window_index is None else int(window_index)
        )
        kv_cache["flowcache_compressed_history_layer_idx"] = (
            None if layer_idx is None else int(layer_idx)
        )
        kv_cache["flowcache_compressed_history_target_ratio"] = float(
            plan["target_ratio"])
        kv_cache["flowcache_compressed_history_strategy"] = str(plan["strategy"])
        kv_cache["flowcache_compressed_history_tokens"] = int(keep_tokens)
        kv_cache["flowcache_compressed_history_global_end_index"] = int(
            global_end_index)
        kv_cache["flowcache_compressed_history_local_end_index"] = int(
            local_end_index)
        kv_cache["flowcache_compressed_history_valid"] = True

        flags["sidecar_created"] = True
        flags["sidecar_valid"] = True
        _flowcache_record_persistent_sidecar(
            flowcache_manager,
            action="create",
            window_index=window_index,
            layer_idx=layer_idx,
            source_start=source_start,
            source_end=source_end,
            compressed_tokens=keep_tokens,
            reason="created_from_history_range",
        )
        return compressed_history_key, compressed_history_value, flags
    except Exception as exc:
        flags["sidecar_fallback"] = True
        flags["fallback"] = True
        flags["skipped_reason"] = f"fallback_exception:{exc}"
        _flowcache_record_persistent_sidecar(
            flowcache_manager,
            action="fallback",
            window_index=window_index,
            layer_idx=layer_idx,
            source_start=source_start,
            source_end=source_end,
            compressed_tokens=0,
            reason=str(exc),
        )
        if hasattr(flowcache_manager, "_warning"):
            flowcache_manager._warning(
                "persistent-cache sidecar fallback for "
                f"window_index={window_index} layer_idx={layer_idx}: {exc}")
        return history_key, history_value, flags


def _flowcache_verify_finite_tensors(flowcache_manager, tensors, feature="cache_body"):
    config = getattr(flowcache_manager, "config", None)
    if feature == "persistent_cache":
        debug_verify = getattr(
            config, "persistent_cache_compress_debug_verify", False)
    elif feature == "compacted_kv":
        debug_verify = getattr(config, "compacted_kv_debug_verify", False)
    else:
        debug_verify = getattr(config, "cache_body_compress_debug_verify", False)
    if not (
        flowcache_manager is not None and
        debug_verify
    ):
        return

    for name, tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"non-finite tensor in {feature} {name}")


def _flowcache_history_keep_indices(num_history_tokens, keep_tokens, device):
    if keep_tokens <= 0:
        return None
    if keep_tokens == 1:
        return torch.zeros((1,), device=device, dtype=torch.long)
    return torch.div(
        torch.arange(keep_tokens, device=device, dtype=torch.long) *
        (num_history_tokens - 1),
        keep_tokens - 1,
        rounding_mode="floor",
    )


def _flowcache_record_attention_output_diff(
    flowcache_manager,
    *,
    window_index,
    layer_idx,
    branch,
    source_event,
    original_output,
    compressed_output,
):
    if not (
        flowcache_manager is not None and
        getattr(flowcache_manager, "should_run_shadow_compare", False)
    ):
        return

    try:
        original = original_output.detach().float()
        compressed = compressed_output.detach().float()
        diff = (original - compressed).abs()
        mean_abs_diff = diff.mean().item()
        max_abs_diff = diff.max().item()
        denom = original.abs().sum().clamp_min(1e-12)
        relative_l1 = (diff.sum() / denom).item()

        cosine_similarity = None
        if original.numel() > 0 and compressed.numel() > 0:
            original_flat = original.reshape(original.shape[0], -1)
            compressed_flat = compressed.reshape(compressed.shape[0], -1)
            cosine_similarity = torch.nn.functional.cosine_similarity(
                original_flat, compressed_flat, dim=1).mean().item()

        flowcache_manager.record_attention_output_diff(
            window_index=window_index,
            layer_idx=layer_idx,
            branch=branch,
            attention_event=source_event or branch,
            source_event=source_event or branch,
            mean_abs_diff=mean_abs_diff,
            max_abs_diff=max_abs_diff,
            relative_l1=relative_l1,
            cosine_similarity=cosine_similarity,
        )
    except Exception as exc:
        if hasattr(flowcache_manager, "_warning"):
            flowcache_manager._warning(
                "attention output diff failed for "
                f"window_index={window_index} layer_idx={layer_idx}: {exc}")


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=1,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.frame_length = 1560
        self.max_attention_size = 21 * self.frame_length
        self.block_length = 3 * self.frame_length

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        updating_cache=False,
        flowcache_manager=None,
        flowcache_window_index=None,
        flowcache_layer_idx=None,
        flowcache_event=None,
        step_cache_active_block_indices=None,
        step_cache_num_frame_per_block=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        full_window_tokens = s
        step_cache_sparse = step_cache_active_block_indices is not None
        step_cache_active_token_indices = None
        step_cache_active_frame_indices = None
        if step_cache_sparse:
            if kv_cache is None:
                raise ValueError("step-cache sparse attention requires kv_cache inference path")
            if b != 1:
                raise ValueError("R2 step-cache sparse attention currently requires batch=1")
            if step_cache_num_frame_per_block is None or step_cache_num_frame_per_block <= 0:
                raise ValueError("step-cache sparse attention requires num_frame_per_block")
            frames = int(grid_sizes[0, 0].item())
            if frames % int(step_cache_num_frame_per_block):
                raise ValueError("current frame count is not divisible by num_frame_per_block")
            frame_seqlen_sparse = int(math.prod(grid_sizes[0][1:]).item())
            num_blocks_sparse = frames // int(step_cache_num_frame_per_block)
            tokens_per_block_sparse = int(step_cache_num_frame_per_block) * frame_seqlen_sparse
            active_blocks_sparse = tuple(int(item) for item in step_cache_active_block_indices)
            if not active_blocks_sparse:
                raise ValueError("step-cache sparse attention requires at least one recompute block")
            step_cache_active_token_indices = block_token_indices(
                num_blocks=num_blocks_sparse,
                tokens_per_block=tokens_per_block_sparse,
                block_indices=active_blocks_sparse,
                device=x.device,
            )
            step_cache_active_frame_indices = block_frame_indices(
                num_blocks=num_blocks_sparse,
                frames_per_block=int(step_cache_num_frame_per_block),
                block_indices=active_blocks_sparse,
                device=x.device,
            )
        flowcache_initial_branch = (
            "clean_cache_update"
            if updating_cache or flowcache_event == "clean_cache_update_attention"
            else (flowcache_event or "denoise_attention")
        )
        flowcache_final_branch = flowcache_initial_branch
        flowcache_final_visible_tokens = s
        flowcache_final_history_tokens = 0
        flowcache_final_current_tokens = s
        flowcache_final_sink_tokens = 0
        attention_path_token = _flowcache_start_attention_profile(
            flowcache_manager,
            "attention_forward_total",
            window_index=flowcache_window_index,
            layer_idx=flowcache_layer_idx,
            branch=flowcache_initial_branch,
            device=x.device,
            visible_tokens=s,
            history_tokens=0,
            current_tokens=s,
            sink_tokens=0,
            q_shape=(b, s, n, d),
            k_shape=(b, s, n, d),
            v_shape=(b, s, n, d),
            dtype=x.dtype,
            timing_scope="inclusive",
        )
        clean_attention_path_token = None
        if flowcache_initial_branch == "clean_cache_update":
            clean_attention_path_token = _flowcache_start_attention_profile(
                flowcache_manager,
                "clean_cache_update_attention",
                window_index=flowcache_window_index,
                layer_idx=flowcache_layer_idx,
                branch="clean_cache_update",
                device=x.device,
                visible_tokens=s,
                history_tokens=0,
                current_tokens=s,
                sink_tokens=0,
                q_shape=(b, s, n, d),
                k_shape=(b, s, n, d),
                v_shape=(b, s, n, d),
                dtype=x.dtype,
                timing_scope="inclusive",
            )
        attention_profile_token = _flowcache_start_profile(
            flowcache_manager,
            "attention_forward_sampled",
            window_index=flowcache_window_index,
            layer_idx=flowcache_layer_idx,
            device=x.device,
        )
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q_input = x if step_cache_active_token_indices is None else x.index_select(1, step_cache_active_token_indices)
            q_len = q_input.shape[1]
            q = self.norm_q(self.q(q_input)).view(b, q_len, n, d)
            # K/V intentionally remain full-window in layer_residual_current_kv_v1.
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        qkv_profile_token = _flowcache_start_profile(
            flowcache_manager,
            "attention_qkv_sampled",
            window_index=flowcache_window_index,
            layer_idx=flowcache_layer_idx,
            device=x.device,
        )
        qkv_attention_token = _flowcache_start_attention_profile(
            flowcache_manager,
            "qkv_or_input_projection",
            window_index=flowcache_window_index,
            layer_idx=flowcache_layer_idx,
            branch=flowcache_initial_branch,
            device=x.device,
            visible_tokens=s,
            history_tokens=0,
            current_tokens=s,
            sink_tokens=0,
            q_shape=(b, (step_cache_active_token_indices.numel() if step_cache_active_token_indices is not None else s), n, d),
            k_shape=(b, s, n, d),
            v_shape=(b, s, n, d),
            dtype=x.dtype,
        )
        q, k, v = qkv_fn(x)
        _flowcache_end_attention_profile(
            flowcache_manager,
            qkv_attention_token,
            branch=flowcache_initial_branch,
            device=x.device,
            visible_tokens=s,
            history_tokens=0,
            current_tokens=s,
            sink_tokens=0,
            q_shape=q.shape,
            k_shape=k.shape,
            v_shape=v.shape,
            dtype=v.dtype,
        )
        _flowcache_end_profile(flowcache_manager, qkv_profile_token)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                rope_attention_token = _flowcache_start_attention_profile(
                    flowcache_manager,
                    "rope",
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    branch=flowcache_initial_branch,
                    device=q.device,
                    visible_tokens=s,
                    history_tokens=0,
                    current_tokens=s,
                    sink_tokens=0,
                    q_shape=q.shape,
                    k_shape=k.shape,
                    v_shape=v.shape,
                    dtype=v.dtype,
                )
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
                _flowcache_end_attention_profile(
                    flowcache_manager,
                    rope_attention_token,
                    branch=flowcache_initial_branch,
                    device=roped_query.device,
                    visible_tokens=roped_key.shape[1],
                    history_tokens=0,
                    current_tokens=roped_key.shape[1],
                    sink_tokens=0,
                    q_shape=roped_query.shape,
                    k_shape=roped_key.shape,
                    v_shape=v.shape,
                    dtype=v.dtype,
                )

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                rope_attention_token = _flowcache_start_attention_profile(
                    flowcache_manager,
                    "rope",
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    branch=flowcache_initial_branch,
                    device=q.device,
                    visible_tokens=s,
                    history_tokens=0,
                    current_tokens=s,
                    sink_tokens=0,
                    q_shape=q.shape,
                    k_shape=k.shape,
                    v_shape=v.shape,
                    dtype=v.dtype,
                )
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
                _flowcache_end_attention_profile(
                    flowcache_manager,
                    rope_attention_token,
                    branch=flowcache_initial_branch,
                    device=roped_query.device,
                    visible_tokens=roped_key.shape[1],
                    history_tokens=0,
                    current_tokens=roped_key.shape[1],
                    sink_tokens=0,
                    q_shape=roped_query.shape,
                    k_shape=roped_key.shape,
                    v_shape=v.shape,
                    dtype=v.dtype,
                )

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            rope_attention_token = _flowcache_start_attention_profile(
                flowcache_manager,
                "rope",
                window_index=flowcache_window_index,
                layer_idx=flowcache_layer_idx,
                branch=flowcache_initial_branch,
                device=q.device,
                visible_tokens=s,
                history_tokens=0,
                current_tokens=s,
                sink_tokens=0,
                q_shape=q.shape,
                k_shape=k.shape,
                v_shape=v.shape,
                dtype=v.dtype,
            )
            if step_cache_sparse:
                q_chunks = []
                q_offset = 0
                active_block_tokens = int(step_cache_num_frame_per_block) * frame_seqlen_sparse
                active_grid = grid_sizes.clone()
                active_grid[:, 0] = int(step_cache_num_frame_per_block)
                for local_block_index in active_blocks_sparse:
                    q_chunk = q[:, q_offset:q_offset + active_block_tokens]
                    q_chunks.append(
                        causal_rope_apply(
                            q_chunk,
                            active_grid,
                            freqs,
                            start_frame=current_start_frame + local_block_index * int(step_cache_num_frame_per_block),
                        ).type_as(v)
                    )
                    q_offset += active_block_tokens
                roped_query = torch.cat(q_chunks, dim=1)
            else:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)   # [B, L, 12, 128]
            _flowcache_end_attention_profile(
                flowcache_manager,
                rope_attention_token,
                branch=flowcache_initial_branch,
                device=roped_query.device,
                visible_tokens=roped_key.shape[1],
                history_tokens=0,
                current_tokens=roped_key.shape[1],
                sink_tokens=0,
                q_shape=roped_query.shape,
                k_shape=roped_key.shape,
                v_shape=v.shape,
                dtype=v.dtype,
            )
            
            grid_sizes_one_block = grid_sizes.clone()
            grid_sizes_one_block[:,0] = 3

            # only caching the first block
            cache_end = cache_start + self.block_length
            num_new_tokens = cache_end - kv_cache["global_end_index"].item()
            kv_cache_size = kv_cache["k"].shape[1]

            sink_tokens = 1 * self.block_length # we keep the first block in the cache
            flowcache_cache_write_reason = "cache_write"
            flowcache_eviction_happened = False

            kv_write_profile_token = _flowcache_start_profile(
                flowcache_manager,
                "kv_cache_write_sampled",
                window_index=flowcache_window_index,
                layer_idx=flowcache_layer_idx,
                device=roped_key.device,
            )
            kv_write_attention_token = _flowcache_start_attention_profile(
                flowcache_manager,
                "kv_cache_write",
                window_index=flowcache_window_index,
                layer_idx=flowcache_layer_idx,
                branch=flowcache_initial_branch,
                device=roped_key.device,
                visible_tokens=roped_key.shape[1],
                history_tokens=0,
                current_tokens=roped_key.shape[1],
                sink_tokens=sink_tokens,
                q_shape=roped_query.shape,
                k_shape=roped_key.shape,
                v_shape=v.shape,
                dtype=v.dtype,
            )
            try:
                if (num_new_tokens > 0) and (
                        num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                    flowcache_cache_write_reason = "cache_write_after_eviction"
                    flowcache_eviction_happened = True
                    kv_eviction_profile_token = _flowcache_start_profile(
                        flowcache_manager,
                        "kv_eviction_sampled",
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        device=roped_key.device,
                    )
                    kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    _flowcache_end_profile(flowcache_manager, kv_eviction_profile_token)
                    
                    local_end_index = kv_cache["local_end_index"].item() + cache_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    local_start_index = local_end_index - self.block_length
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :self.block_length]
                    kv_cache["v"][:, local_start_index:local_end_index] = v[:, :self.block_length]
                else:
                    local_end_index = kv_cache["local_end_index"].item() + cache_end - kv_cache["global_end_index"].item()
                    local_start_index = local_end_index - self.block_length
                    if local_start_index == 0: # first block is not roped in the cache
                        kv_cache["k"][:, local_start_index:local_end_index] = k[:, :self.block_length]
                    else:
                        kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :self.block_length]

                    kv_cache["v"][:, local_start_index:local_end_index] = v[:, :self.block_length]
            finally:
                _flowcache_end_attention_profile(
                    flowcache_manager,
                    kv_write_attention_token,
                    branch=flowcache_initial_branch,
                    device=roped_key.device,
                    visible_tokens=roped_key.shape[1],
                    history_tokens=0,
                    current_tokens=roped_key.shape[1],
                    sink_tokens=sink_tokens,
                    q_shape=roped_query.shape,
                    k_shape=roped_key.shape,
                    v_shape=v.shape,
                    dtype=v.dtype,
                )
                _flowcache_end_profile(flowcache_manager, kv_write_profile_token)

            if num_new_tokens > 0: # prevent updating when caching clean frame
                kv_cache["global_end_index"].fill_(cache_end)
                kv_cache["local_end_index"].fill_(local_end_index)

            _flowcache_invalidate_persistent_sidecar(
                flowcache_manager,
                kv_cache,
                window_index=flowcache_window_index,
                layer_idx=flowcache_layer_idx,
                reason=flowcache_cache_write_reason,
                write_start=local_start_index,
                write_end=local_end_index,
                eviction_happened=flowcache_eviction_happened,
            )

            if local_start_index == 0:
                # no kv attn with cache
                _flowcache_record_attention_parts(
                    flowcache_manager,
                    event=flowcache_event,
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    attention_branch="current_only",
                    updating_cache=updating_cache,
                    block_length=self.block_length,
                    sink_tokens=sink_tokens,
                    anchor_tokens=0,
                    anchor_value_tokens=0,
                    working_tokens=0,
                    working_value_tokens=0,
                    current_tokens=roped_key.shape[1],
                    current_value_tokens=v.shape[1],
                    input_tokens=roped_key.shape[1],
                    input_value_tokens=v.shape[1],
                    current_start=current_start,
                    cache_start=cache_start,
                    cache_end=cache_end,
                    global_end_index=kv_cache["global_end_index"].item(),
                    local_end_index=kv_cache["local_end_index"].item(),
                )
                real_plan = _flowcache_real_plan(
                    flowcache_manager,
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    branch="current_only",
                    source_event=flowcache_event,
                    protected_sink_tokens=0,
                    protected_current_tokens=roped_key.shape[1],
                    original_history_tokens=0,
                    original_total_kv_tokens=roped_key.shape[1],
                    candidate_region_start=0,
                    candidate_region_end=0,
                )
                _flowcache_record_real_compression(flowcache_manager, real_plan)
                cache_body_plan = _flowcache_cache_body_plan(
                    flowcache_manager,
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    branch="current_only",
                    source_event=flowcache_event,
                    protected_sink_tokens=0,
                    protected_current_tokens=roped_key.shape[1],
                    original_history_tokens=0,
                    original_visible_tokens=roped_key.shape[1],
                )
                _flowcache_record_cache_body_compression(
                    flowcache_manager, cache_body_plan)
                persistent_plan = _flowcache_persistent_cache_plan(
                    flowcache_manager,
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    branch="current_only",
                    source_event=flowcache_event,
                    protected_sink_tokens=0,
                    protected_current_tokens=roped_key.shape[1],
                    original_history_tokens=0,
                    original_visible_tokens=roped_key.shape[1],
                    source_start=0,
                    source_end=0,
                )
                _flowcache_record_persistent_cache_compression(
                    flowcache_manager, persistent_plan)
                compacted_kv_plan = _flowcache_compacted_kv_plan(
                    flowcache_manager,
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    branch="current_only",
                    source_event=flowcache_event,
                    protected_sink_tokens=0,
                    protected_current_tokens=roped_key.shape[1],
                    original_history_tokens=0,
                    original_visible_tokens=roped_key.shape[1],
                )
                _flowcache_record_compacted_kv(
                    flowcache_manager,
                    compacted_kv_plan,
                    dtype=roped_key.dtype,
                    device=roped_key.device,
                )
                flowcache_final_branch = "current_only"
                flowcache_final_visible_tokens = roped_key.shape[1]
                flowcache_final_history_tokens = 0
                flowcache_final_current_tokens = roped_key.shape[1]
                flowcache_final_sink_tokens = 0
                x = _flowcache_profiled_attention(
                    flowcache_manager,
                    roped_query,
                    roped_key,
                    v,
                    window_index=flowcache_window_index,
                    layer_idx=flowcache_layer_idx,
                    branch=flowcache_final_branch,
                    visible_tokens=flowcache_final_visible_tokens,
                    history_tokens=flowcache_final_history_tokens,
                    current_tokens=flowcache_final_current_tokens,
                    sink_tokens=flowcache_final_sink_tokens)
            else:
                if updating_cache: # updating working cache with clean frame
                    extract_cache_end = local_end_index
                    extract_cache_start = max(0, local_end_index-self.max_attention_size)
                    kv_read_profile_token = _flowcache_start_profile(
                        flowcache_manager,
                        "kv_cache_read_sampled",
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        device=roped_key.device,
                    )
                    kv_read_attention_token = _flowcache_start_attention_profile(
                        flowcache_manager,
                        "kv_cache_read_or_assembly",
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="clean_cache_update",
                        device=roped_key.device,
                        visible_tokens=local_end_index - extract_cache_start,
                        history_tokens=max(
                            0, local_end_index - extract_cache_start -
                            roped_key.shape[1]),
                        current_tokens=roped_key.shape[1],
                        sink_tokens=(
                            self.block_length if extract_cache_start == 0 else 0),
                        q_shape=roped_query.shape,
                        k_shape=kv_cache["k"][:, extract_cache_start:extract_cache_end].shape,
                        v_shape=kv_cache["v"][:, extract_cache_start:extract_cache_end].shape,
                        dtype=v.dtype,
                    )
                    try:
                        working_cache_key = kv_cache["k"][:, extract_cache_start:extract_cache_end].clone()
                        working_cache_v = kv_cache["v"][:, extract_cache_start:extract_cache_end]

                        if extract_cache_start == 0: # rope the global first block in working cache
                            working_cache_key[:,:self.block_length] = causal_rope_apply(
                                working_cache_key[:,:self.block_length], grid_sizes_one_block, freqs, start_frame=0).type_as(v)
                    finally:
                        _flowcache_end_attention_profile(
                            flowcache_manager,
                            kv_read_attention_token,
                            branch="clean_cache_update",
                            device=roped_key.device,
                            visible_tokens=local_end_index - extract_cache_start,
                            history_tokens=max(
                                0, local_end_index - extract_cache_start -
                                roped_key.shape[1]),
                            current_tokens=roped_key.shape[1],
                            sink_tokens=(
                                self.block_length
                                if extract_cache_start == 0 else 0),
                            q_shape=roped_query.shape,
                            k_shape=(
                                working_cache_key.shape
                                if "working_cache_key" in locals()
                                else None),
                            v_shape=(
                                working_cache_v.shape
                                if "working_cache_v" in locals()
                                else None),
                            dtype=v.dtype,
                        )
                        _flowcache_end_profile(flowcache_manager, kv_read_profile_token)

                    input_tokens = working_cache_key.shape[1]
                    input_value_tokens = working_cache_v.shape[1]
                    current_tokens = min(roped_key.shape[1], input_tokens)
                    current_value_tokens = min(v.shape[1], input_value_tokens)
                    anchor_tokens = self.block_length if extract_cache_start == 0 and input_tokens > current_tokens else 0
                    anchor_value_tokens = self.block_length if extract_cache_start == 0 and input_value_tokens > current_value_tokens else 0
                    working_tokens = max(0, input_tokens - anchor_tokens - current_tokens)
                    working_value_tokens = max(0, input_value_tokens - anchor_value_tokens - current_value_tokens)
                    _flowcache_record_attention_parts(
                        flowcache_manager,
                        event=flowcache_event,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        attention_branch="clean_cache_update",
                        updating_cache=updating_cache,
                        block_length=self.block_length,
                        sink_tokens=sink_tokens,
                        anchor_tokens=anchor_tokens,
                        anchor_value_tokens=anchor_value_tokens,
                        working_tokens=working_tokens,
                        working_value_tokens=working_value_tokens,
                        current_tokens=current_tokens,
                        current_value_tokens=current_value_tokens,
                        input_tokens=input_tokens,
                        input_value_tokens=input_value_tokens,
                        current_start=current_start,
                        cache_start=cache_start,
                        cache_end=cache_end,
                        global_end_index=kv_cache["global_end_index"].item(),
                        local_end_index=kv_cache["local_end_index"].item(),
                    )
                    real_plan = _flowcache_real_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="clean_cache_update",
                        source_event=flowcache_event,
                        protected_sink_tokens=anchor_tokens,
                        protected_current_tokens=current_tokens,
                        original_history_tokens=working_tokens,
                        original_total_kv_tokens=input_tokens,
                        candidate_region_start=anchor_tokens,
                        candidate_region_end=anchor_tokens + working_tokens,
                    )
                    _flowcache_record_real_compression(flowcache_manager, real_plan)
                    cache_body_plan = _flowcache_cache_body_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="clean_cache_update",
                        source_event=flowcache_event,
                        protected_sink_tokens=anchor_tokens,
                        protected_current_tokens=current_tokens,
                        original_history_tokens=working_tokens,
                        original_visible_tokens=input_tokens,
                    )
                    cache_body_recorded = False
                    if cache_body_plan is not None and cache_body_plan.get("applied", False):
                        try:
                            history_tokens = working_tokens
                            keep_tokens = int(cache_body_plan["compressed_history_tokens"])
                            if working_tokens != working_value_tokens:
                                raise ValueError(
                                    "clean cache key/value history length mismatch: "
                                    f"k={working_tokens} v={working_value_tokens}")
                            if keep_tokens <= 0 or keep_tokens > history_tokens:
                                raise ValueError(
                                    f"invalid keep_tokens={keep_tokens} for history_tokens={history_tokens}")

                            history_start = anchor_tokens
                            history_end = anchor_tokens + history_tokens
                            keep_indices = _flowcache_history_keep_indices(
                                history_tokens, keep_tokens, working_cache_key.device)
                            compressed_history_key = working_cache_key[
                                :, history_start:history_end].index_select(1, keep_indices)
                            compressed_history_v = working_cache_v[
                                :, history_start:history_end].index_select(1, keep_indices)
                            working_cache_key = torch.cat([
                                working_cache_key[:, :history_start],
                                compressed_history_key,
                                working_cache_key[:, history_end:],
                            ], dim=1)
                            working_cache_v = torch.cat([
                                working_cache_v[:, :history_start],
                                compressed_history_v,
                                working_cache_v[:, history_end:],
                            ], dim=1)

                            if working_cache_key.shape[1] != cache_body_plan["compressed_visible_tokens"]:
                                raise ValueError(
                                    "compressed clean key visible length mismatch: "
                                    f"actual={working_cache_key.shape[1]} "
                                    f"planned={cache_body_plan['compressed_visible_tokens']}")
                            if working_cache_v.shape[1] != cache_body_plan["compressed_visible_tokens"]:
                                raise ValueError(
                                    "compressed clean value visible length mismatch: "
                                    f"actual={working_cache_v.shape[1]} "
                                    f"planned={cache_body_plan['compressed_visible_tokens']}")

                            _flowcache_verify_finite_tensors(
                                flowcache_manager,
                                [
                                    ("clean_compressed_key", working_cache_key),
                                    ("clean_compressed_value", working_cache_v),
                                ],
                            )
                            _flowcache_record_cache_body_compression(
                                flowcache_manager, cache_body_plan, fallback=False)
                            cache_body_recorded = True
                        except Exception as exc:
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "cache-body clean fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            _flowcache_record_cache_body_compression(
                                flowcache_manager,
                                cache_body_plan,
                                fallback=True,
                                skipped_reason=f"fallback_exception:{exc}",
                            )
                            cache_body_recorded = True

                    if cache_body_plan is not None and not cache_body_recorded:
                        _flowcache_record_cache_body_compression(
                            flowcache_manager, cache_body_plan)

                    persistent_history_start = anchor_tokens
                    persistent_history_end = anchor_tokens + working_tokens
                    persistent_source_start = extract_cache_start + persistent_history_start
                    persistent_source_end = persistent_source_start + working_tokens
                    persistent_plan = _flowcache_persistent_cache_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="clean_cache_update",
                        source_event=flowcache_event,
                        protected_sink_tokens=anchor_tokens,
                        protected_current_tokens=current_tokens,
                        original_history_tokens=working_tokens,
                        original_visible_tokens=input_tokens,
                        source_start=persistent_source_start,
                        source_end=persistent_source_end,
                    )
                    persistent_recorded = False
                    if persistent_plan is not None and persistent_plan.get("applied", False):
                        original_working_cache_key = working_cache_key
                        original_working_cache_v = working_cache_v
                        try:
                            compressed_history_key, compressed_history_v, persistent_flags = (
                                _flowcache_apply_persistent_history_sidecar(
                                    flowcache_manager,
                                    kv_cache,
                                    persistent_plan,
                                    working_cache_key[:, persistent_history_start:persistent_history_end],
                                    working_cache_v[:, persistent_history_start:persistent_history_end],
                                    window_index=flowcache_window_index,
                                    layer_idx=flowcache_layer_idx,
                                    source_start=persistent_source_start,
                                    source_end=persistent_source_end,
                                    global_end_index=kv_cache["global_end_index"].item(),
                                    local_end_index=kv_cache["local_end_index"].item(),
                                )
                            )
                            if persistent_flags.get("fallback"):
                                _flowcache_record_persistent_cache_compression(
                                    flowcache_manager,
                                    persistent_plan,
                                    fallback=True,
                                    skipped_reason=persistent_flags.get("skipped_reason"),
                                    sidecar_fallback=True,
                                    sidecar_valid=False,
                                )
                                persistent_recorded = True
                            else:
                                working_cache_key = torch.cat([
                                    working_cache_key[:, :persistent_history_start],
                                    compressed_history_key,
                                    working_cache_key[:, persistent_history_end:],
                                ], dim=1)
                                working_cache_v = torch.cat([
                                    working_cache_v[:, :persistent_history_start],
                                    compressed_history_v,
                                    working_cache_v[:, persistent_history_end:],
                                ], dim=1)
                                if working_cache_key.shape[1] != persistent_plan["compressed_visible_tokens"]:
                                    raise ValueError(
                                        "persistent clean visible key length mismatch: "
                                        f"actual={working_cache_key.shape[1]} "
                                        f"planned={persistent_plan['compressed_visible_tokens']}")
                                if working_cache_v.shape[1] != persistent_plan["compressed_visible_tokens"]:
                                    raise ValueError(
                                        "persistent clean visible value length mismatch: "
                                        f"actual={working_cache_v.shape[1]} "
                                        f"planned={persistent_plan['compressed_visible_tokens']}")
                                _flowcache_verify_finite_tensors(
                                    flowcache_manager,
                                    [
                                        ("persistent_clean_key", working_cache_key),
                                        ("persistent_clean_value", working_cache_v),
                                    ],
                                    feature="persistent_cache",
                                )
                                _flowcache_record_persistent_cache_compression(
                                    flowcache_manager,
                                    persistent_plan,
                                    fallback=False,
                                    sidecar_created=persistent_flags.get("sidecar_created", False),
                                    sidecar_reused=persistent_flags.get("sidecar_reused", False),
                                    sidecar_invalidated=persistent_flags.get("sidecar_invalidated", False),
                                    sidecar_fallback=persistent_flags.get("sidecar_fallback", False),
                                    sidecar_valid=persistent_flags.get("sidecar_valid", False),
                                )
                                persistent_recorded = True
                        except Exception as exc:
                            working_cache_key = original_working_cache_key
                            working_cache_v = original_working_cache_v
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "persistent-cache clean fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            _flowcache_record_persistent_cache_compression(
                                flowcache_manager,
                                persistent_plan,
                                fallback=True,
                                skipped_reason=f"fallback_exception:{exc}",
                                sidecar_fallback=True,
                                sidecar_valid=False,
                            )
                            persistent_recorded = True

                    if persistent_plan is not None and not persistent_recorded:
                        _flowcache_record_persistent_cache_compression(
                            flowcache_manager, persistent_plan)

                    compacted_anchor_tokens = anchor_tokens
                    compacted_current_tokens = min(
                        current_tokens, working_cache_key.shape[1])
                    compacted_history_start = compacted_anchor_tokens
                    compacted_history_end = max(
                        compacted_history_start,
                        working_cache_key.shape[1] - compacted_current_tokens)
                    compacted_history_tokens = max(
                        0, compacted_history_end - compacted_history_start)
                    compacted_kv_plan = _flowcache_compacted_kv_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="clean_cache_update",
                        source_event=flowcache_event,
                        protected_sink_tokens=compacted_anchor_tokens,
                        protected_current_tokens=compacted_current_tokens,
                        original_history_tokens=compacted_history_tokens,
                        original_visible_tokens=working_cache_key.shape[1],
                    )
                    compacted_kv_recorded = False
                    if compacted_kv_plan is not None and compacted_kv_plan.get("applied", False):
                        original_working_cache_key = working_cache_key
                        original_working_cache_v = working_cache_v
                        compact_start_time = None
                        try:
                            compact_start_time = torch.cuda.Event(enable_timing=True) if working_cache_key.is_cuda else None
                            compact_end_time = torch.cuda.Event(enable_timing=True) if working_cache_key.is_cuda else None
                            if compact_start_time is not None:
                                compact_start_time.record()
                            history_tokens = compacted_history_tokens
                            keep_tokens = int(compacted_kv_plan["compacted_history_tokens"])
                            if working_cache_key.shape[1] != working_cache_v.shape[1]:
                                raise ValueError(
                                    "compacted clean key/value token length mismatch: "
                                    f"k={working_cache_key.shape[1]} v={working_cache_v.shape[1]}")
                            if keep_tokens <= 0 or keep_tokens > history_tokens:
                                raise ValueError(
                                    f"invalid compacted keep_tokens={keep_tokens} "
                                    f"for history_tokens={history_tokens}")

                            keep_indices = _flowcache_history_keep_indices(
                                history_tokens, keep_tokens, working_cache_key.device)
                            compacted_history_key = working_cache_key[
                                :, compacted_history_start:compacted_history_end].index_select(
                                    1, keep_indices)
                            compacted_history_v = working_cache_v[
                                :, compacted_history_start:compacted_history_end].index_select(
                                    1, keep_indices)
                            working_cache_key = torch.cat([
                                working_cache_key[:, :compacted_history_start],
                                compacted_history_key,
                                working_cache_key[:, compacted_history_end:],
                            ], dim=1).contiguous()
                            working_cache_v = torch.cat([
                                working_cache_v[:, :compacted_history_start],
                                compacted_history_v,
                                working_cache_v[:, compacted_history_end:],
                            ], dim=1).contiguous()
                            if compact_end_time is not None:
                                compact_end_time.record()
                                torch.cuda.synchronize(working_cache_key.device)
                                elapsed_ms = compact_start_time.elapsed_time(compact_end_time)
                            else:
                                elapsed_ms = None

                            if working_cache_key.shape[1] != compacted_kv_plan["compacted_visible_tokens"]:
                                raise ValueError(
                                    "compacted clean visible key length mismatch: "
                                    f"actual={working_cache_key.shape[1]} "
                                    f"planned={compacted_kv_plan['compacted_visible_tokens']}")
                            if working_cache_v.shape[1] != compacted_kv_plan["compacted_visible_tokens"]:
                                raise ValueError(
                                    "compacted clean visible value length mismatch: "
                                    f"actual={working_cache_v.shape[1]} "
                                    f"planned={compacted_kv_plan['compacted_visible_tokens']}")
                            _flowcache_verify_finite_tensors(
                                flowcache_manager,
                                [
                                    ("compacted_clean_key", working_cache_key),
                                    ("compacted_clean_value", working_cache_v),
                                ],
                                feature="compacted_kv",
                            )
                            _flowcache_record_compacted_kv(
                                flowcache_manager,
                                compacted_kv_plan,
                                fallback=False,
                                dtype=working_cache_key.dtype,
                                device=working_cache_key.device,
                                elapsed_ms=elapsed_ms,
                            )
                            compacted_kv_recorded = True
                        except Exception as exc:
                            working_cache_key = original_working_cache_key
                            working_cache_v = original_working_cache_v
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "compacted-kv clean fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            _flowcache_record_compacted_kv(
                                flowcache_manager,
                                compacted_kv_plan,
                                fallback=True,
                                fallback_reason=f"fallback_exception:{exc}",
                                skipped_reason=f"fallback_exception:{exc}",
                                dtype=working_cache_key.dtype,
                                device=working_cache_key.device,
                            )
                            compacted_kv_recorded = True

                    if compacted_kv_plan is not None and not compacted_kv_recorded:
                        _flowcache_record_compacted_kv(
                            flowcache_manager,
                            compacted_kv_plan,
                            dtype=working_cache_key.dtype,
                            device=working_cache_key.device,
                        )

                    flowcache_final_branch = "clean_cache_update"
                    flowcache_final_visible_tokens = working_cache_key.shape[1]
                    flowcache_final_history_tokens = max(
                        0, working_cache_key.shape[1] - current_tokens -
                        anchor_tokens)
                    flowcache_final_current_tokens = current_tokens
                    flowcache_final_sink_tokens = anchor_tokens
                    x = _flowcache_profiled_attention(
                        flowcache_manager,
                        roped_query,
                        working_cache_key,
                        working_cache_v,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch=flowcache_final_branch,
                        visible_tokens=flowcache_final_visible_tokens,
                        history_tokens=flowcache_final_history_tokens,
                        current_tokens=flowcache_final_current_tokens,
                        sink_tokens=flowcache_final_sink_tokens,
                    )

                else:
                    # 1. extract working cache
                    # calculate the length of working cache
                    # History visibility must be based on the original full window, not sparse Q length.
                    query_length = full_window_tokens
                    working_cache_max_length = self.max_attention_size - query_length - self.block_length

                    extract_cache_end = local_start_index
                    extract_cache_start = max(self.block_length, local_start_index - working_cache_max_length) # working cache does not include the first anchor block
                    kv_read_profile_token = _flowcache_start_profile(
                        flowcache_manager,
                        "kv_cache_read_sampled",
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        device=roped_key.device,
                    )
                    kv_read_attention_token = _flowcache_start_attention_profile(
                        flowcache_manager,
                        "kv_cache_read_or_assembly",
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="anchor_working_current",
                        device=roped_key.device,
                        visible_tokens=(
                            self.block_length +
                            extract_cache_end - extract_cache_start +
                            roped_key.shape[1]),
                        history_tokens=extract_cache_end - extract_cache_start,
                        current_tokens=roped_key.shape[1],
                        sink_tokens=self.block_length,
                        q_shape=roped_query.shape,
                        k_shape=kv_cache["k"][:, extract_cache_start:extract_cache_end].shape,
                        v_shape=kv_cache["v"][:, extract_cache_start:extract_cache_end].shape,
                        dtype=v.dtype,
                    )
                    try:
                        working_cache_key = kv_cache["k"][:, extract_cache_start:extract_cache_end]
                        working_cache_v = kv_cache["v"][:, extract_cache_start:extract_cache_end]

                        # 2. extract anchor cache, roped as the past frame
                        working_cache_frame_length = working_cache_key.shape[1] // self.frame_length
                        rope_start_frame = current_start_frame - working_cache_frame_length - 3

                        anchor_cache_key = causal_rope_apply(
                            kv_cache["k"][:, :self.block_length], grid_sizes_one_block, freqs, start_frame=rope_start_frame).type_as(v)
                        anchor_cache_v = kv_cache["v"][:, :self.block_length]
                    finally:
                        _flowcache_end_attention_profile(
                            flowcache_manager,
                            kv_read_attention_token,
                            branch="anchor_working_current",
                            device=roped_key.device,
                            visible_tokens=(
                                self.block_length +
                                extract_cache_end - extract_cache_start +
                                roped_key.shape[1]),
                            history_tokens=(
                                extract_cache_end - extract_cache_start),
                            current_tokens=roped_key.shape[1],
                            sink_tokens=self.block_length,
                            q_shape=roped_query.shape,
                            k_shape=(
                                working_cache_key.shape
                                if "working_cache_key" in locals()
                                else None),
                            v_shape=(
                                working_cache_v.shape
                                if "working_cache_v" in locals()
                                else None),
                            dtype=v.dtype,
                        )
                        _flowcache_end_profile(flowcache_manager, kv_read_profile_token)

                    original_working_tokens = working_cache_key.shape[1]
                    original_visible_tokens = (
                        anchor_cache_key.shape[1] +
                        original_working_tokens +
                        roped_key.shape[1]
                    )
                    cache_body_plan = _flowcache_cache_body_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="anchor_working_current",
                        source_event=flowcache_event,
                        protected_sink_tokens=anchor_cache_key.shape[1],
                        protected_current_tokens=roped_key.shape[1],
                        original_history_tokens=original_working_tokens,
                        original_visible_tokens=original_visible_tokens,
                    )
                    cache_body_recorded = False
                    if cache_body_plan is not None and cache_body_plan.get("applied", False):
                        original_working_cache_key = working_cache_key
                        original_working_cache_v = working_cache_v
                        try:
                            history_tokens = original_working_tokens
                            keep_tokens = int(cache_body_plan["compressed_history_tokens"])
                            if working_cache_key.shape[1] != working_cache_v.shape[1]:
                                raise ValueError(
                                    "working key/value token length mismatch: "
                                    f"k={working_cache_key.shape[1]} v={working_cache_v.shape[1]}")
                            if keep_tokens <= 0 or keep_tokens > history_tokens:
                                raise ValueError(
                                    f"invalid keep_tokens={keep_tokens} for history_tokens={history_tokens}")

                            keep_indices = _flowcache_history_keep_indices(
                                history_tokens, keep_tokens, working_cache_key.device)
                            compressed_working_cache_key = working_cache_key.index_select(
                                1, keep_indices)
                            compressed_working_cache_v = working_cache_v.index_select(
                                1, keep_indices)

                            compressed_visible_tokens = (
                                anchor_cache_key.shape[1] +
                                compressed_working_cache_key.shape[1] +
                                roped_key.shape[1]
                            )
                            if compressed_visible_tokens != cache_body_plan["compressed_visible_tokens"]:
                                raise ValueError(
                                    "compressed cache-body visible length mismatch: "
                                    f"actual={compressed_visible_tokens} "
                                    f"planned={cache_body_plan['compressed_visible_tokens']}")

                            _flowcache_verify_finite_tensors(
                                flowcache_manager,
                                [
                                    ("anchor_key", anchor_cache_key),
                                    ("anchor_value", anchor_cache_v),
                                    ("compressed_working_key", compressed_working_cache_key),
                                    ("compressed_working_value", compressed_working_cache_v),
                                    ("current_key", roped_key),
                                    ("current_value", v),
                                ],
                            )
                            working_cache_key = compressed_working_cache_key
                            working_cache_v = compressed_working_cache_v
                            _flowcache_record_cache_body_compression(
                                flowcache_manager, cache_body_plan, fallback=False)
                            cache_body_recorded = True
                        except Exception as exc:
                            working_cache_key = original_working_cache_key
                            working_cache_v = original_working_cache_v
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "cache-body compression fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            _flowcache_record_cache_body_compression(
                                flowcache_manager,
                                cache_body_plan,
                                fallback=True,
                                skipped_reason=f"fallback_exception:{exc}",
                            )
                            cache_body_recorded = True

                    if cache_body_plan is not None and not cache_body_recorded:
                        _flowcache_record_cache_body_compression(
                            flowcache_manager, cache_body_plan)

                    persistent_history_tokens = working_cache_key.shape[1]
                    persistent_visible_tokens = (
                        anchor_cache_key.shape[1] +
                        persistent_history_tokens +
                        roped_key.shape[1]
                    )
                    persistent_plan = _flowcache_persistent_cache_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="anchor_working_current",
                        source_event=flowcache_event,
                        protected_sink_tokens=anchor_cache_key.shape[1],
                        protected_current_tokens=roped_key.shape[1],
                        original_history_tokens=persistent_history_tokens,
                        original_visible_tokens=persistent_visible_tokens,
                        source_start=extract_cache_start,
                        source_end=extract_cache_end,
                    )
                    persistent_recorded = False
                    if persistent_plan is not None and persistent_plan.get("applied", False):
                        original_working_cache_key = working_cache_key
                        original_working_cache_v = working_cache_v
                        try:
                            compressed_working_cache_key, compressed_working_cache_v, persistent_flags = (
                                _flowcache_apply_persistent_history_sidecar(
                                    flowcache_manager,
                                    kv_cache,
                                    persistent_plan,
                                    working_cache_key,
                                    working_cache_v,
                                    window_index=flowcache_window_index,
                                    layer_idx=flowcache_layer_idx,
                                    source_start=extract_cache_start,
                                    source_end=extract_cache_end,
                                    global_end_index=kv_cache["global_end_index"].item(),
                                    local_end_index=kv_cache["local_end_index"].item(),
                                )
                            )
                            if persistent_flags.get("fallback"):
                                working_cache_key = original_working_cache_key
                                working_cache_v = original_working_cache_v
                                _flowcache_record_persistent_cache_compression(
                                    flowcache_manager,
                                    persistent_plan,
                                    fallback=True,
                                    skipped_reason=persistent_flags.get("skipped_reason"),
                                    sidecar_fallback=True,
                                    sidecar_valid=False,
                                )
                                persistent_recorded = True
                            else:
                                compressed_visible_tokens = (
                                    anchor_cache_key.shape[1] +
                                    compressed_working_cache_key.shape[1] +
                                    roped_key.shape[1]
                                )
                                if compressed_visible_tokens != persistent_plan["compressed_visible_tokens"]:
                                    raise ValueError(
                                        "persistent visible length mismatch: "
                                        f"actual={compressed_visible_tokens} "
                                        f"planned={persistent_plan['compressed_visible_tokens']}")
                                _flowcache_verify_finite_tensors(
                                    flowcache_manager,
                                    [
                                        ("persistent_anchor_key", anchor_cache_key),
                                        ("persistent_anchor_value", anchor_cache_v),
                                        ("persistent_working_key", compressed_working_cache_key),
                                        ("persistent_working_value", compressed_working_cache_v),
                                        ("persistent_current_key", roped_key),
                                        ("persistent_current_value", v),
                                    ],
                                    feature="persistent_cache",
                                )
                                working_cache_key = compressed_working_cache_key
                                working_cache_v = compressed_working_cache_v
                                _flowcache_record_persistent_cache_compression(
                                    flowcache_manager,
                                    persistent_plan,
                                    fallback=False,
                                    sidecar_created=persistent_flags.get("sidecar_created", False),
                                    sidecar_reused=persistent_flags.get("sidecar_reused", False),
                                    sidecar_invalidated=persistent_flags.get("sidecar_invalidated", False),
                                    sidecar_fallback=persistent_flags.get("sidecar_fallback", False),
                                    sidecar_valid=persistent_flags.get("sidecar_valid", False),
                                )
                                persistent_recorded = True
                        except Exception as exc:
                            working_cache_key = original_working_cache_key
                            working_cache_v = original_working_cache_v
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "persistent-cache compression fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            _flowcache_record_persistent_cache_compression(
                                flowcache_manager,
                                persistent_plan,
                                fallback=True,
                                skipped_reason=f"fallback_exception:{exc}",
                                sidecar_fallback=True,
                                sidecar_valid=False,
                            )
                            persistent_recorded = True

                    if persistent_plan is not None and not persistent_recorded:
                        _flowcache_record_persistent_cache_compression(
                            flowcache_manager, persistent_plan)

                    compacted_history_tokens = working_cache_key.shape[1]
                    compacted_original_visible_tokens = (
                        anchor_cache_key.shape[1] +
                        compacted_history_tokens +
                        roped_key.shape[1]
                    )
                    compacted_kv_plan = _flowcache_compacted_kv_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="anchor_working_current",
                        source_event=flowcache_event,
                        protected_sink_tokens=anchor_cache_key.shape[1],
                        protected_current_tokens=roped_key.shape[1],
                        original_history_tokens=compacted_history_tokens,
                        original_visible_tokens=compacted_original_visible_tokens,
                    )
                    compacted_kv_recorded = False
                    if compacted_kv_plan is not None and compacted_kv_plan.get("applied", False):
                        original_working_cache_key = working_cache_key
                        original_working_cache_v = working_cache_v
                        start_time = time.perf_counter()
                        try:
                            history_tokens = working_cache_key.shape[1]
                            keep_tokens = int(compacted_kv_plan["compacted_history_tokens"])
                            if working_cache_key.shape[1] != working_cache_v.shape[1]:
                                raise ValueError(
                                    "compacted key/value token length mismatch: "
                                    f"k={working_cache_key.shape[1]} v={working_cache_v.shape[1]}")
                            if keep_tokens <= 0 or keep_tokens > history_tokens:
                                raise ValueError(
                                    f"invalid compacted keep_tokens={keep_tokens} "
                                    f"for history_tokens={history_tokens}")

                            keep_indices = _flowcache_history_keep_indices(
                                history_tokens, keep_tokens, working_cache_key.device)
                            compacted_working_cache_key = working_cache_key.index_select(
                                1, keep_indices).contiguous()
                            compacted_working_cache_v = working_cache_v.index_select(
                                1, keep_indices).contiguous()

                            compacted_visible_tokens = (
                                anchor_cache_key.shape[1] +
                                compacted_working_cache_key.shape[1] +
                                roped_key.shape[1]
                            )
                            if compacted_visible_tokens != compacted_kv_plan["compacted_visible_tokens"]:
                                raise ValueError(
                                    "compacted visible length mismatch: "
                                    f"actual={compacted_visible_tokens} "
                                    f"planned={compacted_kv_plan['compacted_visible_tokens']}")

                            _flowcache_verify_finite_tensors(
                                flowcache_manager,
                                [
                                    ("compacted_anchor_key", anchor_cache_key),
                                    ("compacted_anchor_value", anchor_cache_v),
                                    ("compacted_working_key", compacted_working_cache_key),
                                    ("compacted_working_value", compacted_working_cache_v),
                                    ("compacted_current_key", roped_key),
                                    ("compacted_current_value", v),
                                ],
                                feature="compacted_kv",
                            )
                            working_cache_key = compacted_working_cache_key
                            working_cache_v = compacted_working_cache_v
                            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                            _flowcache_record_compacted_kv(
                                flowcache_manager,
                                compacted_kv_plan,
                                fallback=False,
                                dtype=working_cache_key.dtype,
                                device=working_cache_key.device,
                                elapsed_ms=elapsed_ms,
                            )
                            compacted_kv_recorded = True
                        except Exception as exc:
                            working_cache_key = original_working_cache_key
                            working_cache_v = original_working_cache_v
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "compacted-kv fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                            _flowcache_record_compacted_kv(
                                flowcache_manager,
                                compacted_kv_plan,
                                fallback=True,
                                fallback_reason=f"fallback_exception:{exc}",
                                skipped_reason=f"fallback_exception:{exc}",
                                dtype=working_cache_key.dtype,
                                device=working_cache_key.device,
                                elapsed_ms=elapsed_ms,
                            )
                            compacted_kv_recorded = True

                    if compacted_kv_plan is not None and not compacted_kv_recorded:
                        _flowcache_record_compacted_kv(
                            flowcache_manager,
                            compacted_kv_plan,
                            dtype=working_cache_key.dtype,
                            device=working_cache_key.device,
                        )

                    # 3. attention with working cache and anchor cache
                    kv_assembly_attention_token = _flowcache_start_attention_profile(
                        flowcache_manager,
                        "kv_cache_read_or_assembly",
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="anchor_working_current",
                        device=roped_key.device,
                        visible_tokens=(
                            anchor_cache_key.shape[1] +
                            working_cache_key.shape[1] +
                            roped_key.shape[1]),
                        history_tokens=working_cache_key.shape[1],
                        current_tokens=roped_key.shape[1],
                        sink_tokens=anchor_cache_key.shape[1],
                        q_shape=roped_query.shape,
                        k_shape=working_cache_key.shape,
                        v_shape=working_cache_v.shape,
                        dtype=v.dtype,
                    )
                    try:
                        input_key = torch.cat([
                            anchor_cache_key,
                            working_cache_key,
                            roped_key
                        ], dim=1)

                        input_v = torch.cat([
                            anchor_cache_v,
                            working_cache_v,
                            v
                        ], dim=1)
                    finally:
                        _flowcache_end_attention_profile(
                            flowcache_manager,
                            kv_assembly_attention_token,
                            branch="anchor_working_current",
                            device=roped_key.device,
                            visible_tokens=(
                                anchor_cache_key.shape[1] +
                                working_cache_key.shape[1] +
                                roped_key.shape[1]),
                            history_tokens=working_cache_key.shape[1],
                            current_tokens=roped_key.shape[1],
                            sink_tokens=anchor_cache_key.shape[1],
                            q_shape=roped_query.shape,
                            k_shape=(
                                input_key.shape
                                if "input_key" in locals() else None),
                            v_shape=(
                                input_v.shape
                                if "input_v" in locals() else None),
                            dtype=v.dtype,
                        )

                    _flowcache_record_attention_parts(
                        flowcache_manager,
                        event=flowcache_event,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        attention_branch="anchor_working_current",
                        updating_cache=updating_cache,
                        block_length=self.block_length,
                        sink_tokens=sink_tokens,
                        anchor_tokens=anchor_cache_key.shape[1],
                        anchor_value_tokens=anchor_cache_v.shape[1],
                        working_tokens=working_cache_key.shape[1],
                        working_value_tokens=working_cache_v.shape[1],
                        current_tokens=roped_key.shape[1],
                        current_value_tokens=v.shape[1],
                        input_tokens=input_key.shape[1],
                        input_value_tokens=input_v.shape[1],
                        current_start=current_start,
                        cache_start=cache_start,
                        cache_end=cache_end,
                        global_end_index=kv_cache["global_end_index"].item(),
                        local_end_index=kv_cache["local_end_index"].item(),
                    )
                    real_plan = _flowcache_real_plan(
                        flowcache_manager,
                        window_index=flowcache_window_index,
                        layer_idx=flowcache_layer_idx,
                        branch="anchor_working_current",
                        source_event=flowcache_event,
                        protected_sink_tokens=anchor_cache_key.shape[1],
                        protected_current_tokens=roped_key.shape[1],
                        original_history_tokens=working_cache_key.shape[1],
                        original_total_kv_tokens=input_key.shape[1],
                        candidate_region_start=anchor_cache_key.shape[1],
                        candidate_region_end=anchor_cache_key.shape[1] + working_cache_key.shape[1],
                    )

                    attention_key = input_key
                    attention_v = input_v
                    compression_applied = False
                    compression_recorded = False

                    if real_plan is not None and real_plan.get("applied", False):
                        try:
                            history_tokens = working_cache_key.shape[1]
                            keep_tokens = int(real_plan["compressed_history_tokens"])
                            if working_cache_key.shape[1] != working_cache_v.shape[1]:
                                raise ValueError(
                                    "working key/value token length mismatch: "
                                    f"k={working_cache_key.shape[1]} v={working_cache_v.shape[1]}")
                            if keep_tokens <= 0 or keep_tokens > history_tokens:
                                raise ValueError(
                                    f"invalid keep_tokens={keep_tokens} for history_tokens={history_tokens}")

                            keep_indices = _flowcache_history_keep_indices(
                                history_tokens, keep_tokens, working_cache_key.device)
                            compressed_working_cache_key = working_cache_key.index_select(
                                1, keep_indices)
                            compressed_working_cache_v = working_cache_v.index_select(
                                1, keep_indices)

                            attention_key = torch.cat([
                                anchor_cache_key,
                                compressed_working_cache_key,
                                roped_key
                            ], dim=1)
                            attention_v = torch.cat([
                                anchor_cache_v,
                                compressed_working_cache_v,
                                v
                            ], dim=1)

                            if attention_key.shape[1] != real_plan["compressed_total_kv_tokens"]:
                                raise ValueError(
                                    "compressed key token length mismatch: "
                                    f"actual={attention_key.shape[1]} "
                                    f"planned={real_plan['compressed_total_kv_tokens']}")
                            if attention_v.shape[1] != real_plan["compressed_total_kv_tokens"]:
                                raise ValueError(
                                    "compressed value token length mismatch: "
                                    f"actual={attention_v.shape[1]} "
                                    f"planned={real_plan['compressed_total_kv_tokens']}")

                            compression_applied = True
                        except Exception as exc:
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "real compression fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            _flowcache_record_real_compression(
                                flowcache_manager,
                                real_plan,
                                fallback=True,
                                skipped_reason=f"fallback_exception:{exc}",
                            )
                            compression_recorded = True
                            attention_key = input_key
                            attention_v = input_v
                            compression_applied = False

                    if real_plan is not None and not compression_recorded and not compression_applied:
                        _flowcache_record_real_compression(
                            flowcache_manager,
                            real_plan,
                            fallback=False,
                        )

                    flowcache_final_branch = "anchor_working_current"
                    flowcache_final_visible_tokens = attention_key.shape[1]
                    flowcache_final_history_tokens = max(
                        0,
                        attention_key.shape[1] - anchor_cache_key.shape[1] -
                        roped_key.shape[1])
                    flowcache_final_current_tokens = roped_key.shape[1]
                    flowcache_final_sink_tokens = anchor_cache_key.shape[1]

                    if compression_applied:
                        try:
                            if (
                                flowcache_manager is not None and
                                getattr(flowcache_manager, "should_run_shadow_compare", False)
                            ):
                                original_x = _flowcache_profiled_attention(
                                    flowcache_manager,
                                    roped_query,
                                    input_key,
                                    input_v,
                                    window_index=flowcache_window_index,
                                    layer_idx=flowcache_layer_idx,
                                    branch="anchor_working_current",
                                    visible_tokens=input_key.shape[1],
                                    history_tokens=working_cache_key.shape[1],
                                    current_tokens=roped_key.shape[1],
                                    sink_tokens=anchor_cache_key.shape[1],
                                )
                                compressed_x = _flowcache_profiled_attention(
                                    flowcache_manager,
                                    roped_query,
                                    attention_key,
                                    attention_v,
                                    window_index=flowcache_window_index,
                                    layer_idx=flowcache_layer_idx,
                                    branch="anchor_working_current",
                                    visible_tokens=attention_key.shape[1],
                                    history_tokens=max(
                                        0,
                                        attention_key.shape[1] -
                                        anchor_cache_key.shape[1] -
                                        roped_key.shape[1]),
                                    current_tokens=roped_key.shape[1],
                                    sink_tokens=anchor_cache_key.shape[1],
                                )
                                _flowcache_record_attention_output_diff(
                                    flowcache_manager,
                                    window_index=flowcache_window_index,
                                    layer_idx=flowcache_layer_idx,
                                    branch="anchor_working_current",
                                    source_event=flowcache_event,
                                    original_output=original_x,
                                    compressed_output=compressed_x,
                                )
                                x = compressed_x
                            else:
                                x = _flowcache_profiled_attention(
                                    flowcache_manager,
                                    roped_query,
                                    attention_key,
                                    attention_v,
                                    window_index=flowcache_window_index,
                                    layer_idx=flowcache_layer_idx,
                                    branch="anchor_working_current",
                                    visible_tokens=attention_key.shape[1],
                                    history_tokens=max(
                                        0,
                                        attention_key.shape[1] -
                                        anchor_cache_key.shape[1] -
                                        roped_key.shape[1]),
                                    current_tokens=roped_key.shape[1],
                                    sink_tokens=anchor_cache_key.shape[1],
                                )
                            _flowcache_record_real_compression(
                                flowcache_manager,
                                real_plan,
                                fallback=False,
                            )
                        except Exception as exc:
                            if hasattr(flowcache_manager, "_warning"):
                                flowcache_manager._warning(
                                    "compressed attention fallback for "
                                    f"window_index={flowcache_window_index} "
                                    f"layer_idx={flowcache_layer_idx}: {exc}")
                            _flowcache_record_real_compression(
                                flowcache_manager,
                                real_plan,
                                fallback=True,
                                skipped_reason=f"fallback_attention_exception:{exc}",
                            )
                            x = _flowcache_profiled_attention(
                                flowcache_manager,
                                roped_query,
                                input_key,
                                input_v,
                                window_index=flowcache_window_index,
                                layer_idx=flowcache_layer_idx,
                                branch="anchor_working_current",
                                visible_tokens=input_key.shape[1],
                                history_tokens=working_cache_key.shape[1],
                                current_tokens=roped_key.shape[1],
                                sink_tokens=anchor_cache_key.shape[1],
                            )
                    else:
                        x = _flowcache_profiled_attention(
                            flowcache_manager,
                            roped_query,
                            input_key,
                            input_v,
                            window_index=flowcache_window_index,
                            layer_idx=flowcache_layer_idx,
                            branch="anchor_working_current",
                            visible_tokens=input_key.shape[1],
                            history_tokens=working_cache_key.shape[1],
                            current_tokens=roped_key.shape[1],
                            sink_tokens=anchor_cache_key.shape[1],
                        )
                 

        # output
        attention_output_profile_token = _flowcache_start_profile(
            flowcache_manager,
            "attention_output_sampled",
            window_index=flowcache_window_index,
            layer_idx=flowcache_layer_idx,
            device=x.device,
        )
        output_projection_attention_token = _flowcache_start_attention_profile(
            flowcache_manager,
            "output_projection",
            window_index=flowcache_window_index,
            layer_idx=flowcache_layer_idx,
            branch=flowcache_final_branch,
            device=x.device,
            visible_tokens=flowcache_final_visible_tokens,
            history_tokens=flowcache_final_history_tokens,
            current_tokens=flowcache_final_current_tokens,
            sink_tokens=flowcache_final_sink_tokens,
            q_shape=x.shape,
            k_shape=None,
            v_shape=None,
            dtype=x.dtype,
        )
        x = x.flatten(2)
        x = self.o(x)
        _flowcache_end_attention_profile(
            flowcache_manager,
            output_projection_attention_token,
            branch=flowcache_final_branch,
            device=x.device,
            visible_tokens=flowcache_final_visible_tokens,
            history_tokens=flowcache_final_history_tokens,
            current_tokens=flowcache_final_current_tokens,
            sink_tokens=flowcache_final_sink_tokens,
            q_shape=x.shape,
            k_shape=None,
            v_shape=None,
            dtype=x.dtype,
        )
        _flowcache_end_profile(flowcache_manager, attention_output_profile_token)
        _flowcache_end_profile(flowcache_manager, attention_profile_token)
        _flowcache_end_attention_profile(
            flowcache_manager,
            clean_attention_path_token,
            branch="clean_cache_update",
            device=x.device,
            visible_tokens=flowcache_final_visible_tokens,
            history_tokens=flowcache_final_history_tokens,
            current_tokens=flowcache_final_current_tokens,
            sink_tokens=flowcache_final_sink_tokens,
            q_shape=(b, s, n, d),
            k_shape=None,
            v_shape=None,
            dtype=x.dtype,
        )
        _flowcache_end_attention_profile(
            flowcache_manager,
            attention_path_token,
            branch=flowcache_final_branch,
            device=x.device,
            visible_tokens=flowcache_final_visible_tokens,
            history_tokens=flowcache_final_history_tokens,
            current_tokens=flowcache_final_current_tokens,
            sink_tokens=flowcache_final_sink_tokens,
            q_shape=(b, s, n, d),
            k_shape=None,
            v_shape=None,
            dtype=x.dtype,
        )
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        updating_cache=False,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        flowcache_manager=None,
        flowcache_window_index=None,
        flowcache_layer_idx=None,
        flowcache_event=None,
        step_cache_metric_recorder=None,
        step_cache_runtime=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        block_profile_token = _flowcache_start_profile(
            flowcache_manager,
            "block_forward_sampled",
            window_index=flowcache_window_index,
            layer_idx=flowcache_layer_idx,
            device=x.device,
        )
        layer_input = x
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None, num_frames_override=None):
            local_num_frames = num_frames if num_frames_override is None else int(num_frames_override)
            cross_profile_token = _flowcache_start_profile(
                flowcache_manager,
                "cross_attention_sampled",
                window_index=flowcache_window_index,
                layer_idx=flowcache_layer_idx,
                device=x.device,
            )
            cross_attn_output = self.cross_attn(
                self.norm3(x), context,
                context_lens, crossattn_cache=crossattn_cache)
            _flowcache_end_profile(flowcache_manager, cross_profile_token)
            x = x + cross_attn_output
            ffn_profile_token = _flowcache_start_profile(
                flowcache_manager,
                "mlp_forward_sampled",
                window_index=flowcache_window_index,
                layer_idx=flowcache_layer_idx,
                device=x.device,
            )
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(local_num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            _flowcache_end_profile(flowcache_manager, ffn_profile_token)
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(local_num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x


        # self-attention
        first_layer_modulated_input = (
            self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1 + e[1])
            + e[0]
        ).flatten(1, 2)
        if (
            step_cache_runtime is not None
            and getattr(step_cache_runtime, "needs_first_layer_metric", False)
            and flowcache_layer_idx == 0
        ):
            step_cache_runtime.prepare_decisions_from_first_layer(
                first_layer_modulated_input,
                grid_sizes,
            )
        if (
            step_cache_metric_recorder is not None
            and getattr(step_cache_metric_recorder, "enabled", False)
            and flowcache_layer_idx == 0
        ):
            step_cache_metric_recorder.record_first_layer_modulated_input(
                first_layer_modulated_input,
                grid_sizes,
            )
        step_cache_active = bool(
            step_cache_runtime is not None
            and getattr(step_cache_runtime, "active", False)
            and not updating_cache
            and flowcache_layer_idx is not None
        )
        if step_cache_active and step_cache_runtime.implementation == "sparse":
            recompute_blocks = step_cache_runtime.recompute_local_block_indices(flowcache_layer_idx)
            reuse_blocks = step_cache_runtime.reuse_local_block_indices(flowcache_layer_idx)
            if reuse_blocks:
                num_blocks = step_cache_runtime.current_block_count
                tokens_per_block = step_cache_runtime.tokens_per_block
                num_frame_per_block = step_cache_runtime.num_frame_per_block
                active_token_indices = block_token_indices(
                    num_blocks=num_blocks,
                    tokens_per_block=tokens_per_block,
                    block_indices=recompute_blocks,
                    device=x.device,
                )
                active_frame_indices = block_frame_indices(
                    num_blocks=num_blocks,
                    frames_per_block=num_frame_per_block,
                    block_indices=recompute_blocks,
                    device=x.device,
                )
                e_active = tuple(part.index_select(1, active_frame_indices) for part in e)
                y_active = self.self_attn(
                    first_layer_modulated_input,
                    seq_lens, grid_sizes,
                    freqs, block_mask, kv_cache, current_start, cache_start,
                    updating_cache=updating_cache,
                    flowcache_manager=flowcache_manager,
                    flowcache_window_index=flowcache_window_index,
                    flowcache_layer_idx=flowcache_layer_idx,
                    flowcache_event=flowcache_event,
                    step_cache_active_block_indices=recompute_blocks,
                    step_cache_num_frame_per_block=num_frame_per_block,
                )
                active_num_frames = int(active_frame_indices.numel())
                x_active_input = layer_input.index_select(1, active_token_indices)
                x_active = x_active_input + (
                    y_active.unflatten(dim=1, sizes=(active_num_frames, frame_seqlen)) * e_active[2]
                ).flatten(1, 2)
                x_active = cross_attn_ffn(
                    x_active, context, context_lens, e_active, crossattn_cache,
                    num_frames_override=active_num_frames,
                )
                # Refresh residuals only for truly recomputed blocks.
                for active_position, local_block_index in enumerate(recompute_blocks):
                    start_token = active_position * tokens_per_block
                    end_token = start_token + tokens_per_block
                    residual = (
                        x_active[:, start_token:end_token]
                        - x_active_input[:, start_token:end_token]
                    )
                    step_cache_runtime.update_residual(
                        local_block_index, flowcache_layer_idx, residual
                    )
                cached_residuals = [
                    step_cache_runtime.get_residual(local_block_index, flowcache_layer_idx)
                    for local_block_index in reuse_blocks
                ]
                x = apply_reuse_residuals(
                    layer_input,
                    x_active,
                    recompute_token_indices=active_token_indices,
                    reuse_blocks=reuse_blocks,
                    tokens_per_block=tokens_per_block,
                    residuals=cached_residuals,
                )
                step_cache_runtime.record_layer_execution(
                    layer_index=flowcache_layer_idx,
                    full_blocks=num_blocks,
                    recompute_blocks=recompute_blocks,
                    reuse_blocks=reuse_blocks,
                )
                _flowcache_end_profile(flowcache_manager, block_profile_token)
                return x

        y = self.self_attn(
            first_layer_modulated_input,
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            updating_cache=updating_cache,
            flowcache_manager=flowcache_manager,
            flowcache_window_index=flowcache_window_index,
            flowcache_layer_idx=flowcache_layer_idx,
            flowcache_event=flowcache_event)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        if step_cache_active:
            recompute_blocks = step_cache_runtime.recompute_local_block_indices(flowcache_layer_idx)
            reuse_blocks = step_cache_runtime.reuse_local_block_indices(flowcache_layer_idx)
            num_blocks = step_cache_runtime.current_block_count
            tokens_per_block = step_cache_runtime.tokens_per_block
            # Full path is used either when every block recomputes or as dense oracle.
            for local_block_index in recompute_blocks:
                start_token = local_block_index * tokens_per_block
                end_token = start_token + tokens_per_block
                step_cache_runtime.update_residual(
                    local_block_index,
                    flowcache_layer_idx,
                    x[:, start_token:end_token] - layer_input[:, start_token:end_token],
                )
            if reuse_blocks:
                if step_cache_runtime.implementation != "dense_reference":
                    raise RuntimeError("sparse step-cache reuse unexpectedly reached dense block path")
                recompute_token_indices = block_token_indices(
                    num_blocks=num_blocks,
                    tokens_per_block=tokens_per_block,
                    block_indices=recompute_blocks,
                    device=x.device,
                )
                recompute_output = x.index_select(1, recompute_token_indices)
                cached_residuals = [
                    step_cache_runtime.get_residual(local_block_index, flowcache_layer_idx)
                    for local_block_index in reuse_blocks
                ]
                x = apply_reuse_residuals(
                    layer_input,
                    recompute_output,
                    recompute_token_indices=recompute_token_indices,
                    reuse_blocks=reuse_blocks,
                    tokens_per_block=tokens_per_block,
                    residuals=cached_residuals,
                )
            step_cache_runtime.record_layer_execution(
                layer_index=flowcache_layer_idx,
                full_blocks=num_blocks,
                recompute_blocks=recompute_blocks,
                reuse_blocks=reuse_blocks,
            )
        _flowcache_end_profile(flowcache_manager, block_profile_token)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        updating_cache=False,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        flowcache_manager=None,
        flowcache_window_index=None,
        flowcache_event=None,
        step_cache_metric_recorder=None,
        step_cache_runtime=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            updating_cache=updating_cache,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "flowcache_manager": flowcache_manager,
                        "flowcache_window_index": flowcache_window_index,
                        "flowcache_layer_idx": block_index,
                        "flowcache_event": flowcache_event,
                        "step_cache_metric_recorder": step_cache_metric_recorder,
                        "step_cache_runtime": step_cache_runtime,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "flowcache_manager": flowcache_manager,
                        "flowcache_window_index": flowcache_window_index,
                        "flowcache_layer_idx": block_index,
                        "flowcache_event": flowcache_event,
                        "step_cache_metric_recorder": step_cache_metric_recorder,
                        "step_cache_runtime": step_cache_runtime,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
