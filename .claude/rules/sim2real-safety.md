# Sim-to-Real Safety (real robot)

Applies to anything that sends commands to physical hardware (Open Duck Mini / Microduck).

## Before the first run of a policy on hardware

- **The user is present and has said to go.** Never run a policy on the robot on your own.
- Hardware limits are set in the runtime, not only in the policy: joint position limits,
  velocity limit, torque/current limit. The first run uses reduced limits.
- Robot is suspended (feet off the ground) for the first run; check joint order, signs and
  zero offsets against the MJCF before it touches the floor.
- An emergency stop (cut power / disable torque) is within reach.
- Observation contract verified (see `testing.md` #4); ONNX parity test passes.
- Control rate and measured latency match what training assumed (domain randomization range
  must cover the real latency).

## During and after

- Start with zero velocity command, then ramp. Stop at the first unexpected motion; do not
  "see if it settles".
- Record each hardware run (policy ckpt hash, runtime commit, limits, outcome, video) in
  `docs/experiments.md`, including falls and failures.
- A policy that worked in sim but failed on hardware is a result: record the symptom before
  changing anything.
