# R3/M0 server assessment (2026-09-19)

Source result: `debug-GPT/step_cache_r3_m0_result_r3_m0_20260919T061043Z.tar.gz`.

Status: `SERVER_VALIDATED PASS` for the seven-case M0 numerical seal.

- Server CPU tests: 35/35 PASS; all five standalone configs passed preflight.
- All seven initial-noise tensors have the same BF16 SHA256.
- `disabled_a == disabled_b == all_recompute` at final latent level, bitwise exact.
- mixed sparse and dense-reference have the same 35-row mask, nine reuse decisions,
  and bitwise-exact final latents (`max_abs=relative_l1=0`).
- fixed quiet and fixed observed have the same mask, nine reuse decisions, and
  bitwise-exact final latents. The observer produced 35 scalar records.
- K/V tokens remain full. Sparse Q/output/cross/MLP tokens are reduced from
  4,914,000 to 3,650,400, with 1,263,600 reuse tokens and exact conservation.
- Every runtime summary ends with zero active blocks, residual tensors, and bytes.
- All seven MP4s decode as 81 frames at 832x480 and 16 fps.

The independently encoded MP4 decoded-frame hashes are not identical even when the
saved final latent tensors are bitwise identical. M0 therefore uses the saved latent
tensor as the numerical oracle and MP4 decoding only as an output-validity gate.

The remaining R3 matrix is reduced to two process launches and three videos:

1. Front, 81 latent, one prompt: long lifecycle, native KV capacity/eviction,
   online schedule wiring, terminal residual cleanup, and peak memory.
2. U-shape, 21 latent, two prompts in one process: U schedule wiring, KV reset,
   sample-id isolation, state cleanup, and two valid videos.

The 3/6/12 latent runs are removed because the accepted 21-latent trace already
executes growing and shrinking windows of one through five blocks; they would repeat
the same tensor/window shapes without closing a remaining risk.
