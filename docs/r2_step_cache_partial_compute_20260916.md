# R2 partial-compute prototype (2026-09-16)

Prerequisite: R0 SERVER_VALIDATED, R1 SERVER_VALIDATED.

Adapter: `layer_residual_current_kv_v1`.

R2 preserves full current-window K/V, native KV write/eviction/history visibility, head,
scheduler, and clean refresh. For injected reuse rows it reuses the latest residual
`H_out-H_in`; for recompute rows it projects active Q only and executes attention output,
cross-attention, and MLP only for active rows. `dense_reference` remains available for
same-mask oracle work and is not a performance method.

R2 server matrix uses 21 latent frames (7 video blocks):

- all_recompute: 0 reuse decisions;
- few: 2 registered reuse decisions `(block,stage)=(2,3),(4,1)`;
- more: 9 registered decisions spanning stages 1/2/3 and non-contiguous active queries.

Before loading the checkpoint, `validate_step_cache_r2_attention.py` runs a CUDA
micro-oracle: full-Q active rows must match sparse-Q rows within the fixed structural gate,
and the full K/V caches must be exactly equal.

Acceptance evidence:

- CPU step-cache tests pass;
- attention micro-oracle passes;
- 3 videos decode to 81 frames;
- decisions: 35 per video, 7 per stage;
- layer execution records: 330 per video (11 windows x 30 layers);
- all-recompute has Q tokens == K tokens;
- sparse cases keep K/V full but reduce Q/output/cross/MLP tokens;
- actual layer reuse events equal decision reuse count x 30;
- more-injection has lower Q-token work than few-injection.

This round does not claim a quality or speed advantage and does not select deployment
thresholds. Those remain R4-R6 tasks.
