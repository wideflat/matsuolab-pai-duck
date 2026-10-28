# HANDOFF

Read this first in a new session. Update it before stopping (see the `handoff` skill).

## Current state
- Phases 0, 2, 3 of spec 001 done on the Mac (see docs/notes.md, docs/experiments.md E001).
- Mac env: `.venv` (uv, CPU jax). Reference repos and the published ONNX policy are in `external/` (gitignored).
- Scripts: `scripts/inspect_mjcf.py`, `scripts/replay_headless.py`.
- Scope is simulation only. GPU budget: $20 total.

## Next steps
1. GPU environment decided: AWS us-east-1 g6.xlarge, $20 cap (see docs/notes.md, gpu-cost.md). Not launched yet.
2. Phase 4 done: bundled pkl is usable; Placo regeneration needs Linux x86_64 and is optional (needs user approval for cost). Phase 5 smoke run done (E002).
3. Phase 5 baseline: decide step count (1.5e8 default ~1.6 h/$1.3, 3e8 ~3.1 h/$2.5) and run on AWS with the pinned env in configs/requirements-aws-train.txt (steps in docs/notes.md). Needs user approval to launch. Key pair and SG already exist (project=matsuolab-pai-duck).
4. Check E001 gap: replay speed 0.09 vs command 0.15 m/s (loop equivalence with the official viewer script).

## Open questions
- Submission deadline unknown.

## Running machines / cost
- None. Instance i-076e1704e8f8776fc terminated 2026-10-27 21:48 -0400; no volumes left. Total spent so far about $0.3 of $20.
