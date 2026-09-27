#!/usr/bin/env python3
"""GPU micro-oracle for R2 active-Q attention.

Compares the active block rows from a normal full-Q current-only attention call
against a call that projects/ropes only that active block while keeping full K/V.
No model checkpoint is needed.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Allow direct execution via `python scripts/validate_step_cache_r2_attention.py`.
# In that mode Python otherwise places only `scripts/` on sys.path, so the
# repository-level `wan` package is not importable. Prefer RF_ROOT when set,
# and fall back to the parent of this scripts directory.
_REPO_ROOT = Path(os.environ.get("RF_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from wan.modules.causal_model import CausalWanSelfAttention
from wan.modules.model import rope_params


def make_freqs(head_dim, device):
    d = head_dim
    return torch.cat([
        rope_params(64, d - 4 * (d // 6)),
        rope_params(64, 2 * (d // 6)),
        rope_params(64, 2 * (d // 6)),
    ], dim=1).to(device)


def make_cache(batch, capacity, heads, head_dim, dtype, device):
    return {
        "k": torch.zeros(batch, capacity, heads, head_dim, dtype=dtype, device=device),
        "v": torch.zeros(batch, capacity, heads, head_dim, dtype=dtype, device=device),
        "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
        "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the R2 attention oracle")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    dim, heads = 12, 2
    head_dim = dim // heads
    h = w = 2
    frames_per_block = 3
    num_blocks = 2
    frames = frames_per_block * num_blocks
    frame_tokens = h * w
    block_tokens = frames_per_block * frame_tokens
    seq_tokens = frames * frame_tokens

    attn = CausalWanSelfAttention(dim, heads, local_attn_size=-1, sink_size=0).to(device=device, dtype=dtype).eval()
    attn.frame_length = frame_tokens
    attn.block_length = block_tokens
    attn.max_attention_size = 7 * block_tokens
    x = torch.randn(1, seq_tokens, dim, device=device, dtype=dtype)
    grid = torch.tensor([[frames, h, w]], dtype=torch.long)
    seq_lens = torch.tensor([seq_tokens], dtype=torch.long)
    freqs = make_freqs(head_dim, device)
    capacity = 8 * block_tokens
    cache_full = make_cache(1, capacity, heads, head_dim, dtype, device)
    cache_sparse = make_cache(1, capacity, heads, head_dim, dtype, device)

    with torch.no_grad():
        full = attn(
            x, seq_lens, grid, freqs, None,
            kv_cache=cache_full, current_start=0, cache_start=0, updating_cache=False,
        )
        sparse = attn(
            x, seq_lens, grid, freqs, None,
            kv_cache=cache_sparse, current_start=0, cache_start=0, updating_cache=False,
            step_cache_active_block_indices=(1,),
            step_cache_num_frame_per_block=frames_per_block,
        )
    reference = full[:, block_tokens:2*block_tokens]
    diff = (reference.float() - sparse.float()).abs()
    denom = reference.float().abs().mean().clamp_min(1e-8)
    report = {
        "status": "ok",
        "dtype": str(dtype),
        "full_output_shape": list(full.shape),
        "sparse_output_shape": list(sparse.shape),
        "active_reference_shape": list(reference.shape),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "relative_l1": float(diff.mean().div(denom).item()),
        "kv_cache_equal": bool(torch.equal(cache_full["k"], cache_sparse["k"]) and torch.equal(cache_full["v"], cache_sparse["v"])),
    }
    # This is a structural micro-oracle. Record the exact error; fail only on material drift.
    if not report["kv_cache_equal"] or report["relative_l1"] > 0.02:
        report["status"] = "error"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
