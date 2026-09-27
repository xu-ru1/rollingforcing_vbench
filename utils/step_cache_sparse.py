"""Small tensor helpers used by the R2 sparse layer-residual adapter."""
from __future__ import annotations

from typing import Iterable, Sequence
import torch


def block_token_indices(
    *, num_blocks: int, tokens_per_block: int, block_indices: Sequence[int], device
) -> torch.Tensor:
    if num_blocks <= 0 or tokens_per_block <= 0:
        raise ValueError("num_blocks and tokens_per_block must be positive")
    normalized = [int(item) for item in block_indices]
    if len(set(normalized)) != len(normalized):
        raise ValueError("block indices must be unique")
    if any(item < 0 or item >= num_blocks for item in normalized):
        raise ValueError("block index out of range")
    if not normalized:
        return torch.empty((0,), dtype=torch.long, device=device)
    pieces = [
        torch.arange(
            item * tokens_per_block,
            (item + 1) * tokens_per_block,
            dtype=torch.long,
            device=device,
        )
        for item in normalized
    ]
    return torch.cat(pieces)


def block_frame_indices(
    *, num_blocks: int, frames_per_block: int, block_indices: Sequence[int], device
) -> torch.Tensor:
    return block_token_indices(
        num_blocks=num_blocks,
        tokens_per_block=frames_per_block,
        block_indices=block_indices,
        device=device,
    )


def apply_reuse_residuals(
    layer_input: torch.Tensor,
    recompute_output: torch.Tensor,
    *,
    recompute_token_indices: torch.Tensor,
    reuse_blocks: Sequence[int],
    tokens_per_block: int,
    residuals: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Assemble complete H_out from active outputs and cached block residuals."""
    if layer_input.ndim != 3:
        raise ValueError("layer_input must be [B, L, C]")
    if recompute_output.ndim != 3:
        raise ValueError("recompute_output must be [B, L_active, C]")
    if recompute_output.shape[1] != recompute_token_indices.numel():
        raise ValueError("recompute output/token-index length mismatch")
    if len(reuse_blocks) != len(residuals):
        raise ValueError("reuse_blocks/residuals length mismatch")
    output = layer_input.clone()
    if recompute_token_indices.numel():
        output.index_copy_(1, recompute_token_indices, recompute_output)
    for block_index, residual in zip(reuse_blocks, residuals):
        start = int(block_index) * tokens_per_block
        end = start + tokens_per_block
        current = layer_input[:, start:end]
        if residual.shape != current.shape:
            raise ValueError(
                f"cached residual shape mismatch: residual={tuple(residual.shape)} current={tuple(current.shape)}")
        if residual.device != current.device or residual.dtype != current.dtype:
            raise ValueError("cached residual device/dtype mismatch")
        output[:, start:end] = current + residual
    return output
