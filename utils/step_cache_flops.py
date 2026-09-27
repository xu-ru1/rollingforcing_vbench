"""Executed DiT FLOP accounting for the RollingForcing step-cache experiments.

The recorder aggregates actual Linear/Conv3d operations through forward hooks and
actual attention tensor products at the common attention entry point.  It records
both main denoise and clean-cache-update forwards, so clean work cannot disappear
from a cache-method cost report.  Counts use the conventional two FLOPs per
multiply-accumulate; elementwise, normalization, RoPE, softmax, scheduler, VAE,
and video encoding are deliberately reported as out of scope.
"""

from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
import json
import os
from typing import Any, Callable

import torch
import torch.nn as nn


_ACTIVE_RECORDER: ContextVar["DiTFLOPRecorder | None"] = ContextVar(
    "step_cache_active_flop_recorder", default=None)


def record_active_attention(q: torch.Tensor, k: torch.Tensor, *, tag: str) -> None:
    """Record QK and AV products for one attention kernel invocation."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is not None:
        recorder.record_attention(q, k, tag=tag)


class DiTFLOPRecorder:
    """Forward-hook based executed FLOP recorder with explicit branch accounting."""

    def __init__(self, output_path: str | None = None) -> None:
        self.output_path = output_path
        self._handles: list[Any] = []
        self._active_token = None
        self._current_branch: str | None = None
        self._current_sample: int | None = None
        self._model_metadata: dict[str, Any] = {}
        self._totals: dict[str, int] = defaultdict(int)
        self._by_branch: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        self._by_sample: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))
        self._forward_counts: dict[str, int] = defaultdict(int)
        self._sample_forward_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

    @property
    def enabled(self) -> bool:
        return bool(self.output_path)

    @classmethod
    def from_args(cls, args: Any) -> "DiTFLOPRecorder":
        raw = args.get("step_cache") if isinstance(args, dict) else getattr(args, "step_cache", None)
        if raw is None:
            return cls()
        get = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
        value = get("flops_output_path", None)
        if value in (None, "", "null", "None"):
            value = None
        return cls(str(value) if value is not None else None)

    def attach(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        if self._handles:
            raise RuntimeError("FLOP recorder is already attached")
        self._model_metadata = {
            "dim": int(getattr(model, "dim", 0)),
            "ffn_dim": int(getattr(model, "ffn_dim", 0)),
            "num_heads": int(getattr(model, "num_heads", 0)),
            "num_layers": int(getattr(model, "num_layers", len(getattr(model, "blocks", [])))),
            "text_len": int(getattr(model, "text_len", 0)),
            "patch_size": list(getattr(model, "patch_size", ())),
        }
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._linear_hook(name)))
            elif isinstance(module, nn.Conv3d):
                self._handles.append(module.register_forward_hook(self._conv3d_hook(name)))

    def begin_sample(self, sample_id: int) -> None:
        if not self.enabled:
            return
        if self._current_sample is not None:
            raise RuntimeError("previous FLOP sample was not closed")
        self._current_sample = int(sample_id)

    def end_sample(self, sample_id: int) -> None:
        if not self.enabled:
            return
        if self._current_branch is not None:
            raise RuntimeError("FLOP sample closed while a model forward is active")
        if self._current_sample != int(sample_id):
            raise RuntimeError("FLOP sample id mismatch")
        self._current_sample = None

    def begin_forward(self, branch: str) -> None:
        if not self.enabled:
            return
        if self._current_sample is None or self._current_branch is not None:
            raise RuntimeError("invalid FLOP forward lifetime")
        self._current_branch = str(branch)
        self._forward_counts[self._current_branch] += 1
        self._sample_forward_counts[str(self._current_sample)][self._current_branch] += 1
        self._active_token = _ACTIVE_RECORDER.set(self)

    def end_forward(self) -> None:
        if not self.enabled:
            return
        if self._active_token is None:
            raise RuntimeError("FLOP forward was not active")
        _ACTIVE_RECORDER.reset(self._active_token)
        self._active_token = None
        self._current_branch = None

    def record_attention(self, q: torch.Tensor, k: torch.Tensor, *, tag: str) -> None:
        if not self.enabled or self._current_branch is None:
            return
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(f"expected [B,L,H,D] attention tensors, got {q.shape=} {k.shape=}")
        batch, query_tokens, heads, head_dim = (int(item) for item in q.shape)
        key_batch, key_tokens, key_heads, key_dim = (int(item) for item in k.shape)
        if (batch, heads, head_dim) != (key_batch, key_heads, key_dim):
            raise ValueError(f"incompatible Q/K shapes: {q.shape=} {k.shape=}")
        # QK^T plus AV: each matrix product costs two FLOPs per multiply-accumulate.
        flops = 4 * batch * query_tokens * key_tokens * heads * head_dim
        self._add(f"{tag}_attention", flops)

    def summary(self) -> dict[str, Any]:
        total = sum(self._totals.values())
        by_branch = {
            branch: {**dict(values), "total_flops": sum(values.values())}
            for branch, values in self._by_branch.items()
        }
        by_sample = {
            sample: {
                branch: {**dict(values), "total_flops": sum(values.values())}
                for branch, values in branches.items()
            }
            for sample, branches in self._by_sample.items()
        }
        return {
            "status": "ok",
            "counting_convention": {
                "multiply_accumulate_flops": 2,
                "included": ["Linear", "Conv3d", "self_attention_QK_and_AV", "cross_attention_QK_and_AV"],
                "excluded": ["normalization", "elementwise", "RoPE", "softmax", "scheduler", "VAE", "video_encoding"],
                "clean_cache_forward_included": True,
            },
            "model": self._model_metadata,
            "forward_counts": dict(self._forward_counts),
            "sample_forward_counts": {key: dict(value) for key, value in self._sample_forward_counts.items()},
            "by_branch": by_branch,
            "by_sample": by_sample,
            "totals": {**dict(self._totals), "total_flops": total, "total_pflops": total / 1.0e15},
        }

    def write_summary(self) -> None:
        if not self.enabled:
            return
        directory = os.path.dirname(os.path.abspath(self.output_path))
        os.makedirs(directory, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _linear_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(module: nn.Linear, inputs: tuple[Any, ...], output: Any) -> None:
            if self._current_branch is None or not isinstance(output, torch.Tensor):
                return
            self._add(f"linear:{name}", 2 * int(output.numel()) * int(module.in_features))
        return hook

    def _conv3d_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(module: nn.Conv3d, inputs: tuple[Any, ...], output: Any) -> None:
            if self._current_branch is None or not isinstance(output, torch.Tensor):
                return
            kernel_volume = 1
            for size in module.kernel_size:
                kernel_volume *= int(size)
            flops_per_output = 2 * (int(module.in_channels) // int(module.groups)) * kernel_volume
            self._add(f"conv3d:{name}", int(output.numel()) * flops_per_output)
        return hook

    def _add(self, operation: str, flops: int) -> None:
        if self._current_branch is None or self._current_sample is None:
            raise RuntimeError("FLOP operation recorded outside a model forward")
        branch = self._current_branch
        value = int(flops)
        self._totals[operation] += value
        self._by_branch[branch][operation] += value
        self._by_sample[str(self._current_sample)][branch][operation] += value
