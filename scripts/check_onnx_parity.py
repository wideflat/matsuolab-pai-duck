"""ONNX parity check: the exported ONNX policy must give the same action as the JAX policy (testing.md, item 3).

JAX side: restores the orbax checkpoint (the training-time parameters: observation mean/std and the policy MLP)
and re-implements the forward pass in plain JAX, independent of the TF/tf2onnx export path:
  x -> (x - mean) / std -> 3 x [Dense + swish] -> Dense(28) -> first 14 outputs -> tanh   (brax's deterministic action).
ONNX side: Open_Duck_Playground's OnnxInfer, the same object the control loop uses.
Observations: (a) real ones recorded while the ONNX policy walks under several commands, (b) seeded random ones
drawn around the training mean/std (to cover regions walking does not visit).

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground:scripts .venv/bin/python scripts/check_onnx_parity.py \
      --ckpt checkpoints/e005/2026_10_28_173132_129761280 \
      --onnx checkpoints/e005/2026_10_28_173132_129761280.onnx --atol 1e-4

Guarantees: deterministic (fixed seed, no noise); exits non-zero if any action differs by more than --atol, if
shapes differ, or if any value is non-finite. Both files are named, trusted sources (our own AWS run).
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from duck_rollout import DuckRollout
from eval_checkpoints import sha8

COMMANDS = [(0.15, 0.0, 0.0), (-0.15, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, -0.2, 0.0), (0.0, 0.0, 1.0), (0.1, 0.0, 0.5)]
OBS_DIM, ACT_DIM = 101, 14


def load_jax_policy(ckpt):
    ck = ocp.PyTreeCheckpointer()
    path = os.path.abspath(ckpt)
    meta = ck.metadata(path)
    meta = getattr(meta, "item_metadata", meta)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    tree = ck.restore(path, restore_args=jax.tree_util.tree_map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), meta))
    mean, std = tree[0]["mean"]["state"], tree[0]["std"]["state"]
    layers = [tree[1]["params"][f"hidden_{i}"] for i in range(4)]
    if mean.shape != (OBS_DIM,) or layers[0]["kernel"].shape != (OBS_DIM, 512) or layers[3]["kernel"].shape != (128, 2 * ACT_DIM):
        raise ValueError("checkpoint does not have the expected 101 -> 512 -> 256 -> 128 -> 28 policy")

    @jax.jit
    def act(obs):
        h = (obs - mean) / std
        for lyr in layers[:3]:
            h = jax.nn.swish(h @ lyr["kernel"] + lyr["bias"])
        out = h @ layers[3]["kernel"] + layers[3]["bias"]
        return jnp.tanh(out[..., :ACT_DIM])

    return act, np.asarray(mean), np.asarray(std)


def record_observations(onnx, per_command_s):
    obs = []
    for cmd in COMMANDS:
        r = DuckRollout(onnx, cmd)
        infer = r.inf.policy.infer

        def recording(o, _infer=infer):
            obs.append(np.asarray(o, dtype=np.float32).reshape(-1).copy())
            return _infer(o)

        r.inf.policy.infer = recording
        while r.t < per_command_s:
            r.step()
    return np.stack(obs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--atol", type=float, required=True)
    a = p.parse_args()

    act, mean, std = load_jax_policy(a.ckpt)
    real = record_observations(a.onnx, per_command_s=30.0)
    rng = np.random.default_rng(0)
    rand = (mean + std * rng.standard_normal((1000, OBS_DIM))).astype(np.float32)

    onnx_policy = DuckRollout(a.onnx, (0.0, 0.0, 0.0)).inf.policy
    worst = 0.0
    for name, batch in (("walking", real), ("random", rand)):
        j = np.asarray(act(jnp.asarray(batch)))
        o = np.stack([np.asarray(onnx_policy.infer(x)).reshape(-1) for x in batch])
        if j.shape != o.shape or j.shape != (len(batch), ACT_DIM):
            raise SystemExit(f"{name}: shape mismatch jax {j.shape} vs onnx {o.shape}")
        if not (np.isfinite(j).all() and np.isfinite(o).all()):
            raise SystemExit(f"{name}: non-finite values")
        diff = np.abs(j - o)
        worst = max(worst, float(diff.max()))
        print(f"{name:8s} n={len(batch):5d}  max|jax-onnx|={diff.max():.2e}  mean={diff.mean():.2e}  "
              f"action range [{o.min():+.2f}, {o.max():+.2f}]")
    print(f"onnx sha8={sha8(a.onnx)}  atol={a.atol:g}  worst={worst:.2e}  ->", "PASS" if worst <= a.atol else "FAIL")
    raise SystemExit(0 if worst <= a.atol else 1)


if __name__ == "__main__":
    main()
