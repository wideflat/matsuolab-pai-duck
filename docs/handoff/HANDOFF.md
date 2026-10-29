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
- E007 duck soccer (`scripts/soccer_scene.py`, `striker.py`, `eval_soccer.py`, `render_soccer.py`): partial
  success with striker v4 (feet-midpoint position, no slip check while turning): 4/5 tuned goals + 1 fall
  (ball jammed on a post), 3/3 held-out goals. Videos `videos/soccer_*.mp4`.
- E008 jump test (`scripts/jump_test.py`): max 0.15 cm sole clearance, 0.02 s airtime -> no jump/backflip.
- E009 skip-like gait: reference regenerated in Docker (placo 0.6.3; `scripts/refgen_sweep.py`, reproduces the
  official pkl 239/240, the other one is flipped in the official pkl). 8 cm foot height + imitation x2, 100M:
  failed, sole lift 0.43 cm (E005 0.8 cm). Policy ignores reference foot height. Stopped as pre-registered.
- AWS: E009 instance terminated 22:50; nothing left. Total about $3.0 of $20.
- E010 ONNX parity (`scripts/check_onnx_parity.py`): E005 129.8M, max diff 5.1e-07 (tol 1e-4), negative control fails. Pass.
- Records: results in docs/experiments.md (one row per experiment), events in docs/notes.md. Spec 001 progress table and
  spec 002 are up to date through E010.
- Final video: `scripts/assemble_video.py` -> `videos/final/open_duck_mini_final.mp4` (157 s; clips in `videos/final/src/`,
  cards by `scripts/video_cards.py`; title card follows the course slide; no experiment IDs in any caption). Gitignored.
  The end card has no GitHub URL yet (add after the repo is public, then re-run assemble_video.py).
- Report: `docs/report/report.html` -> `docs/report/report.pdf` (PDF is gitignored; A4, 5 pages; `scripts/make_report_figs.py`,
  `scripts/build_report.sh`). Needs the omnicampus account name for the final file name
  `説明資料_PAI最終課題_<account>.pdf`. README updated (results, sources, reproduction steps).
- Scope is simulation only. Deadline 2026-11-02 10:00 (report PDF, public GitHub, 1-3 min YouTube MP4).

## Next steps
1. User: watch the final video and read the report PDF; fix anything wrong.
2. Make the GitHub repo public (needs the user's go-ahead; remote is origin, nothing pushed without being asked), then add the
   URL to the end card and rebuild the video.
3. Upload to YouTube (title `PAI最終課題_<omnicampus account>`, description with the full CC BY 4.0 music credits); submit report
   PDF, GitHub URL, video URL in omnicampus.
4. Optional: jump footage for the failures section (needs a render script); E006-style turn-only tuning (AWS approval needed).

## Open questions
- omnicampus account name (for the PDF and video titles).

## Running machines / cost
- None. E005 instance i-09c5c564ed455814c terminated 2026-10-28 13:48 -0400; no volumes left.
  Total spent about $3.0 of $20 (E009 instance also terminated; see above).
