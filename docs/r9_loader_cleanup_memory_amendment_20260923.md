# R9 loader memory amendment

The formal one-process memory audit found a 42.45 GiB process RSS peak while
loading the EMA checkpoint and a stable 23--24 GiB RSS during generation.  The
R8-frozen `inference.py` kept both `state_dict` and `state_dict_to_load` alive at
module scope after `load_state_dict`, so the selected CPU checkpoint tensors
remained resident for the complete multi-prompt run.

This amendment deletes only those two Python references immediately after the
weights have been copied into `pipeline.generator`, then runs `gc.collect()`.
It does not change configs, model weights, cache decisions, RNG operations,
latent length, crop rules, prompts, or scoring.

Before this source amendment may be used for additional formal shards,
`scripts/run_step_cache_r9_loader_cleanup_gate.sh` must pass on the server.  The
gate reruns the first prompt of the already completed `rf_vanilla/8_32` shard
with seed 0 and 126 latent frames.  It requires exact equality for initial-noise,
final-latent, CPU RNG, and CUDA RNG SHA256 values, and writes a separate
`source_amendment.json`.  The original immutable R8 manifest is retained.
