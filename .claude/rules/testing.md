# Testing Requirements

This is RL experiment code, not a service. There is no coverage gate and TDD is not mandatory.
What must hold is that the pieces that silently break training or deployment are checked.

## Required checks

1. **Environment shapes**: observation and action dimensions match the config and the policy
   network input/output. Assert on shape and dtype, not just values.
2. **Smoke training**: a run of a few steps with tiny `num_envs` completes and logs finite
   rewards (no NaN/Inf). Run it before every long run.
3. **ONNX parity**: the exported ONNX policy and the JAX policy give the same action on the
   same observation, within an explicit tolerance (e.g. `atol=1e-4`). Never compare floats
   with equality.
4. **Observation contract (sim vs runtime)**: the observation vector the runtime builds has the
   same order, scaling and history as the one used in training. Write the order down once and
   test against it.

## Rules

- Seed anything stochastic; a test that passes only sometimes is a failing test.
- Tests never hit live services (wandb, cloud). Use small fixtures.
- When something fails, use the `superpowers:systematic-debugging` skill: reproduce before
  fixing. Fix the implementation, not the test, unless the test is wrong.
- Never claim a run "works" without having run the command and read its output.
