# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

from utils.step_cache_flops import record_active_attention

__all__ = [
    'flash_attention',
    'attention',
]


def _flowcache_shape_text(tensor):
    if tensor is None or not hasattr(tensor, "shape"):
        return "none"
    return "x".join(str(int(item)) for item in tensor.shape)


def _flowcache_attention_meta(metadata, q=None, k=None, v=None):
    payload = dict(metadata or {})
    if q is not None:
        payload.setdefault("q_shape", _flowcache_shape_text(q))
        payload.setdefault("dtype", str(q.dtype))
        payload.setdefault("device", q.device)
    if k is not None:
        payload.setdefault("k_shape", _flowcache_shape_text(k))
    if v is not None:
        payload.setdefault("v_shape", _flowcache_shape_text(v))
        payload.setdefault("dtype", str(v.dtype))
    if "visible_tokens" not in payload and k is not None:
        payload["visible_tokens"] = int(k.shape[1])
    return payload


def _flowcache_attention_start(profiler, phase, metadata, device=None):
    if profiler is None:
        return None
    start = getattr(profiler, "start_attention_profiler_phase", None)
    if start is None:
        return None
    payload = dict(metadata or {})
    payload_device = payload.pop("device", None)
    if device is None:
        device = payload_device
    return start(phase, device=device, **payload)


def _flowcache_attention_end(profiler, token, metadata):
    if profiler is None or token is None:
        return
    end = getattr(profiler, "end_attention_profiler_phase", None)
    if end is not None:
        payload = dict(metadata or {})
        allowed_keys = {
            "branch",
            "device",
            "visible_tokens",
            "history_tokens",
            "current_tokens",
            "sink_tokens",
            "q_shape",
            "k_shape",
            "v_shape",
            "dtype",
            "used_flash_attention",
            "fallback_reason",
            "warning",
        }
        end(token, **{
            key: value for key, value in payload.items()
            if key in allowed_keys
        })


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    flowcache_attention_profiler=None,
    flowcache_attention_metadata=None,
    flop_tag="unknown_attention",
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    record_active_attention(q, k, tag=str(flop_tag))
    if flowcache_attention_profiler is not None:
        base_metadata = _flowcache_attention_meta(
            flowcache_attention_metadata, q=q, k=k, v=v)
        base_metadata["used_flash_attention"] = True
    else:
        base_metadata = None

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    setup_token = _flowcache_attention_start(
        flowcache_attention_profiler,
        "padding_or_mask_setup",
        base_metadata,
        device=q.device)
    try:
        # preprocess query
        if q_lens is None:
            q = half(q.flatten(0, 1))
            q_lens = torch.tensor(
                [lq] * b, dtype=torch.int32).to(
                    device=q.device, non_blocking=True)
        else:
            q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

        # preprocess key, value
        if k_lens is None:
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
            k_lens = torch.tensor(
                [lk] * b, dtype=torch.int32).to(
                    device=k.device, non_blocking=True)
        else:
            k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)
    finally:
        if setup_token is not None:
            _flowcache_attention_end(
                flowcache_attention_profiler,
                setup_token,
                _flowcache_attention_meta(base_metadata, q=q, k=k, v=v))

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    kernel_token = _flowcache_attention_start(
        flowcache_attention_profiler,
        "attention_kernel",
        _flowcache_attention_meta(base_metadata, q=q, k=k, v=v)
        if base_metadata is not None else None,
        device=q.device)
    try:
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            # Note: dropout_p, window_size are not supported in FA3 now.
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)[0].unflatten(0, (b, lq))
        else:
            assert FLASH_ATTN_2_AVAILABLE
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))
    finally:
        if kernel_token is not None:
            _flowcache_attention_end(
                flowcache_attention_profiler,
                kernel_token,
                _flowcache_attention_meta(base_metadata, q=q, k=k, v=v))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    flowcache_attention_profiler=None,
    flowcache_attention_metadata=None,
    flop_tag="unknown_attention",
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
            flowcache_attention_profiler=flowcache_attention_profiler,
            flowcache_attention_metadata=flowcache_attention_metadata,
            flop_tag=flop_tag,
        )
    else:
        record_active_attention(q, k, tag=str(flop_tag))
        if flowcache_attention_profiler is not None:
            base_metadata = _flowcache_attention_meta(
                flowcache_attention_metadata, q=q, k=k, v=v)
            base_metadata["used_flash_attention"] = False
            base_metadata.setdefault(
                "fallback_reason", "flash_attention_unavailable")
        else:
            base_metadata = None
        setup_token = _flowcache_attention_start(
            flowcache_attention_profiler,
            "padding_or_mask_setup",
            base_metadata,
            device=q.device)
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        try:
            attn_mask = None

            q = q.transpose(1, 2).to(dtype)
            k = k.transpose(1, 2).to(dtype)
            v = v.transpose(1, 2).to(dtype)
        finally:
            if setup_token is not None:
                _flowcache_attention_end(
                    flowcache_attention_profiler,
                    setup_token,
                    _flowcache_attention_meta(base_metadata, q=q, k=k, v=v))

        kernel_token = _flowcache_attention_start(
            flowcache_attention_profiler,
            "attention_kernel",
            _flowcache_attention_meta(base_metadata, q=q, k=k, v=v)
            if base_metadata is not None else None,
            device=q.device)
        try:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
        finally:
            if kernel_token is not None:
                _flowcache_attention_end(
                    flowcache_attention_profiler,
                    kernel_token,
                    _flowcache_attention_meta(base_metadata, q=q, k=k, v=v))

        out = out.transpose(1, 2).contiguous()
        return out
