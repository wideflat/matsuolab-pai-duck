# HANDOFF

Read this first in a new session. Update it before stopping (see the `handoff` skill).

## Current state
- Experiments (docs/experiments.md): E001 published policy, E002 smoke, E003 baseline 50M (0.056 m/s),
  E004 tracking_lin_vel x2 50M (0.119 m/s, drifts left), E005 E004 settings 150M (partial success:
  backward 0.71 m and turns >180 deg learned; sideways unstable across checkpoints).
- Checkpoints (gitignored): `checkpoints/{baseline_e003,e004,e005}/`. Eval CSVs in `docs/data/`
  (`*_checkpoints.csv` forward speed, `*_commands.csv` per-command tracking).
- Dance (no training): `scripts/dance.py` + `configs/dance_*.json`. Final videos (gitignored):
  `videos/dance_fluffing_v2.mp4` (67.9 s, 122 BPM) and `videos/dance_monkeys.mp4` (60.8 s, 144 BPM), both with
  E005 129.8M (sha cbbee2e3), no falls. Music: Kevin MacLeod, CC BY 4.0, credit in the YouTube description.
- Other videos: `e003_compare_v2.mp4`, `reference_vs_e003.mp4`, `e003_vs_e004.mp4`, `e003_vs_e004_traj30s.png`.
- E006 navigator (`scripts/navigator.py`, no training): E005 129.8M reaches 4/4 target points, no falls.
- E007 duck soccer (`scripts/soccer_scene.py`, `striker.py`, `eval_soccer.py`): 24 cm / 50 g ball pushed by the
  trunk without falls, but 0/5 goals after 3 striker versions. Blocker: turning in place shifts the base 5-8 cm
  (pivot at the feet), so the striker dithers between line-up and push. Paused, waiting on the user. Next idea:
  track the feet midpoint instead of the base, and enter "push" by a cone behind the ball.
- E008 jump test (`scripts/jump_test.py`): max 0.15 cm sole clearance, 0.02 s airtime -> no jump/backflip.
- Scope is simulation only. Deadline 2026-11-02 10:00 (report PDF, public GitHub, 1-3 min YouTube MP4).

## Next steps
1. Report: E003 -> E004 -> E005 story (reward weight, training length, checkpoint choice), command sensitivity
   analysis (notes.md), dance pipeline. Collect figures from docs/data and videos/.
2. YouTube video (1-3 min): learning progress + E003/E004 comparison + dance clips; add music credits.
3. Make the GitHub repo public (needs user's go-ahead); README update.
4. Optional: E006 turn-only tracking sigma (needs code in scripts/, AWS approval).

## Open questions
- None blocking.

## Running machines / cost
- None. E005 instance i-09c5c564ed455814c terminated 2026-10-28 13:48 -0400; no volumes left.
  Total spent about $2.1 of $20.
