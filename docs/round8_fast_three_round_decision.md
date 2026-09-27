# Round 8 Fast Three-Round Decision Plan

## Goal

Accelerate the next stage so that we can reach a useful conclusion within at most three rounds:

1. whether compacted-token reduction can produce real end-to-end speedup;
2. whether the result survives 3-prompt validation;
3. whether to continue this FlowCache direction or pivot to attention-kernel/backend or VAE/save optimization.

This plan intentionally uses larger steps. It does not continue small KV-ratio ablations.

## Round 8.1: Timing Baseline

Run one 81-frame single-prompt baseline with checkpoint confirmation and no FlowCache intervention:

```bash
cd /mnt/42_store/zxz2/xr/RollingForcing-main
conda activate /data/zxz2/condaenv/rollingforcing
GPU=0 CKPT=checkpoints/rolling_forcing_dmd.pt EMA_FLAG=--use_ema bash scripts/run_round8_fast_three_rounds.sh round81
```

Pass criteria:

- log contains checkpoint loading confirmation;
- no Traceback/OOM/RuntimeError;
- `[EvalMetrics]` exists;
- video is visually normal.

## Round 8.2: Aggressive Upper-Bound Test

Run two 81-frame single-prompt candidates in the same round:

- `ratio025_denoise`: compact only denoise `anchor_working_current` history to 25%;
- `ratio025_clean`: compact denoise plus clean-cache-update history to 25%.

```bash
GPU=0 CKPT=checkpoints/rolling_forcing_dmd.pt EMA_FLAG=--use_ema bash scripts/run_round8_fast_three_rounds.sh round82
```

Decision gate:

- If neither candidate reaches `MIN_SPEEDUP=0.05` against Round 8.1 baseline, stop this token-reduction direction and pivot.
- If a candidate reaches at least 5% speedup but visual quality is bad, stop or mark it as quality-failed.
- If a candidate reaches at least 5% speedup and video is acceptable, continue to Round 8.3.

You can change the gate, for example:

```bash
MIN_SPEEDUP=0.03 GPU=0 bash scripts/run_round8_fast_three_rounds.sh round82
```

## Round 8.3: 3-Prompt Reproduction

This stage automatically selects the best Round 8.2 candidate from the report. If no candidate passed the speed gate, the script stops.

```bash
GPU_3PROMPT=1 CKPT=checkpoints/rolling_forcing_dmd.pt EMA_FLAG=--use_ema bash scripts/run_round8_fast_three_rounds.sh round83
```

Final decision:

- If 3-prompt speedup is still at least 5% and visual quality is acceptable, continue refining that candidate.
- If 3-prompt speedup disappears or video quality fails, conclude that the compacted-token direction is not a robust speed path.
- If Round 8.2 already fails, do not run Round 8.3; use the report as the negative conclusion.

## Report Command

At any time:

```bash
bash scripts/run_round8_fast_three_rounds.sh summary
```

Main report files:

- `logs/round8_fast_three_rounds_report.txt`
- `logs/round8_fast_three_rounds_report.json`

## Files To Return

After Round 8.1:

- `logs/round8_1_81_baseline.log`
- `logs/round8_1_81_baseline_ls.txt`
- `logs/round8_fast_three_rounds_report.txt`
- `logs/round8_fast_three_rounds_report.json`

After Round 8.2:

- `logs/round8_2_81_ratio025_denoise.log`
- `logs/round8_2_81_ratio025_denoise_ls.txt`
- `logs/round8_2_81_ratio025_clean.log`
- `logs/round8_2_81_ratio025_clean_ls.txt`
- `logs/round8_fast_three_rounds_report.txt`
- `logs/round8_fast_three_rounds_report.json`

After Round 8.3:

- `logs/round8_3_81_3prompt_baseline.log`
- `logs/round8_3_81_3prompt_baseline_ls.txt`
- the selected candidate's `round8_3_81_3prompt_ratio025_*.log`
- the selected candidate's `round8_3_81_3prompt_ratio025_*_ls.txt`
- `logs/round8_fast_three_rounds_report.txt`
- `logs/round8_fast_three_rounds_report.json`

For visual QA, keep the corresponding video folders under `videos/`.

## Expected Conclusion Shape

Within three rounds, the result should be one of:

- Continue: `ratio025_clean` or `ratio025_denoise` gives reproducible speedup and acceptable visual quality.
- Stop token reduction: aggressive 25% history retention still does not beat baseline by the speed gate.
- Stop for quality: speedup exists but visible quality regression is unacceptable.
- Pivot: focus next on attention-kernel/backend work or VAE/video-save, not more KV compression ratios.
