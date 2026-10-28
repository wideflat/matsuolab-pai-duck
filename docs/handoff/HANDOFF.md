# HANDOFF

Read this first in a new session. Update it before stopping (see the `handoff` skill).

## Current state
- Done: E001 (published policy replay), E002 (smoke), E003 (baseline 50M, 0.056 m/s), E004 (own change A:
  tracking_lin_vel x2, success 0.119 m/s, but drifts left ~3 deg/s). See docs/experiments.md.
- Checkpoints (gitignored): `checkpoints/baseline_e003/`, `checkpoints/e004/`. Per-checkpoint eval CSVs in `docs/data/`.
- Scripts: `duck_rollout.py` (shared control loop, `phase_rate`), `replay_headless.py`, `eval_checkpoints.py`,
  `render_compare.py` (2x2 fixed-camera video), `render_reference.py` (reference motion vs policy),
  `train_variant.py` (config overrides on AWS), `estimate_bpm.py` (tempo, per band).
- Videos (gitignored): `videos/e003_compare_v2.mp4`, `videos/reference_vs_e003.mp4`.
- Dance: no training needed (command choreography). Probes: command switches every 1-8 beats, head servo
  override, and phase_rate 0.8-1.3 all without falls (E003 policy). Music in `videos/music/` (Kevin MacLeod,
  CC BY 4.0, credit required): Fluffing a Duck 122 BPM, Monkeys Spinning Monkeys 144 BPM.
- Scope is simulation only. Deadline 2026-11-02 10:00 (report PDF, public GitHub, 1-3 min YouTube MP4).

## Next steps
1. Dance pipeline: choreography file in configs/ (moves per beat), sim with phase_rate = BPM*0.54/60,
   render, mux music with ffmpeg. Start with Fluffing a Duck, ~30 s. Choose policy (E004 faster but drifts left).
2. Optional own-change B (tracking_sigma) or fixing E004's yaw drift; each run ~25 min, ~$0.4, needs approval.
3. Report and the YouTube video; make the GitHub repo public.
4. Check E001 gap: replay speed 0.09 vs command 0.15 m/s.

## Open questions
- Whether to use E003 or E004 policy for the dance (E004 drifts left on long straight walks).

## Running machines / cost
- None. E004 instance i-0148caba8dc964cd6 terminated 2026-10-28 11:41 -0400; no volumes left.
  Total spent about $1.2 of $20.
