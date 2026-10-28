"""Roll out checkpoints under each joystick command (forward, back, left, right, turn left/right) and write a CSV.

One row per checkpoint: displacement along the commanded axis (x for vx, y for vy) or the yaw turned in
degrees (for wz), each over --seconds, plus the first fall time across all commands. Used for the E005
decision rule (docs/experiments.md).

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/eval_commands.py \
      --run checkpoints/e004 --seconds 10 --out docs/data/e004_commands.csv

Guarantees: deterministic (no noise); x/y are world-frame displacement from the start; yaw is wrapped
to (-180, 180] per sample and unwrapped over time, so turns past 180 degrees are counted in full.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from duck_rollout import DuckRollout
from eval_checkpoints import list_checkpoints, sha8

COMMANDS = [("fwd_x_m", (0.15, 0.0, 0.0)), ("back_x_m", (-0.15, 0.0, 0.0)), ("left_y_m", (0.0, 0.2, 0.0)),
            ("right_y_m", (0.0, -0.2, 0.0)), ("turn_l_deg", (0.0, 0.0, 1.0)), ("turn_r_deg", (0.0, 0.0, -1.0))]


def yaw(d):
    q = d.qpos[3:7]
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def run(onnx, cmd, seconds):
    r = DuckRollout(onnx, cmd)
    y0 = float(r.data.qpos[1])
    yaws = [yaw(r.data)]
    for _ in range(int(seconds / r.model.opt.timestep)):
        r.step()
        if r.i % r.inf.decimation == 0:
            yaws.append(yaw(r.data))
    turned = np.degrees(np.unwrap(yaws)[-1] - yaws[0])
    return r.dx, float(r.data.qpos[1]) - y0, turned, r.fell_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows = []
    for steps, onnx in list_checkpoints(a.run):
        row, falls = {"steps": steps}, []
        for key, cmd in COMMANDS:
            dx, dy, turned, fell = run(str(onnx), cmd, a.seconds)
            row[key] = f"{turned:.0f}" if key.startswith("turn") else f"{dx if key.endswith('x_m') else dy:.3f}"
            falls += [f"{key}@{fell:.1f}"] if fell is not None else []
        row["falls"] = " ".join(falls)
        row["onnx_sha8"] = sha8(onnx)
        rows.append(row)
        print(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
