# R3 final server assessment (2026-09-19)

Source result: `debug-GPT/step_cache_r3_final_result_r3_final_20260919T070812Z.tar.gz`.

Status: `R3 SERVER_VALIDATED PASS`.

- Front 81-latent: 135 decisions, 27 per stage, 31 main forwards, 26 reuse
  decisions and 780 layer reuse events.
- Native KV lifecycle reached global token index 126,360 while local storage stayed
  capped at 37,440 tokens, so eviction was exercised and observed.
- Front operator conservation holds: Q 15,303,600 + reuse 3,650,400 = full
  18,954,000 tokens; K/V both remain full.
- U-shape two-prompt process: 35 decisions per sample, 12 reuse decisions and 360
  layer reuse events. Both prompt starts observed global/local KV indices at zero.
- Front thresholds were 0.13/0.26/0.338 and U-shape thresholds were
  0.13/0.364/0.182 for stages 1/2/3. No protected stage or window-first block reused.
- Front peak residual storage remained 2,156,544,000 bytes. Peak CUDA allocated
  was 24.73 GiB and reserved was 32.59 GiB in the metadata-enabled 81-latent run.
- The final assessor reported no errors and all three MP4 files decoded correctly.

R3 is closed. Later experiments do not repeat disabled/all-recompute, sparse/dense,
observer equivalence, short boundary shapes, KV eviction, or prompt reset unless model
code changes invalidate this source manifest.
