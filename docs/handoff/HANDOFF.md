# HANDOFF

Read this first in a new session. Update it before stopping (see the `handoff` skill).

## Current state
- Phases 0, 2, 3 of spec 001 done on the Mac (see docs/notes.md, docs/experiments.md E001).
- Mac env: `.venv` (uv, CPU jax). Reference repos and the published ONNX policy are in `external/` (gitignored).
- Scripts: `scripts/inspect_mjcf.py`, `scripts/replay_headless.py`.
- Scope is simulation only. GPU budget: $20 total.

## Next steps
1. GPU environment decided: AWS us-east-1 g6.xlarge, $20 cap (see docs/notes.md, gpu-cost.md). Not launched yet.
2. Phase 4/5: polynomial_coefficients.pkl is already bundled for open_duck_mini_v2; check whether Placo generation is needed.
3. Smoke-train on a GPU to measure steps/s before deciding the step count.
4. Check E001 gap: replay speed 0.09 vs command 0.15 m/s (loop equivalence with the official viewer script).

## Open questions
- Submission deadline unknown.

## Running machines / cost
- None. AWS us-east-1 had no instances or volumes on 2026-10-27.
