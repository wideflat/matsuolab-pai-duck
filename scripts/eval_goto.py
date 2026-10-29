"""E006: check that the navigator (scripts/navigator.py) steers a policy to target points, and write a CSV + plot.

Each target is a fresh rollout from the origin facing +x. Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/eval_goto.py \
      --onnx checkpoints/e005/2026_10_28_173132_129761280.onnx --seconds 60 \
      --out docs/data/e006_goto.csv --png videos/e006_goto.png

Guarantees: deterministic (no noise); one CSV row per target with reach time, final distance, fall time,
and the largest distance from the straight start-to-target line along the way.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from duck_rollout import DuckRollout
from eval_checkpoints import sha8
from navigator import go_to

TARGETS = [(1.0, 0.0), (0.5, 0.5), (0.5, -0.5), (-0.5, 0.0)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def off_line(path, target):
    """Largest perpendicular distance (m) of the path from the line origin -> target."""
    t = np.asarray(target) / np.linalg.norm(target)
    return float(np.max(np.abs(path[:, 0] * t[1] - path[:, 1] * t[0]))) if len(path) else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png", required=True)
    a = p.parse_args()

    rows, paths = [], []
    for target in TARGETS:
        r = DuckRollout(a.onnx, (0.0, 0.0, 0.0))
        res = go_to(r, target, a.seconds)
        row = {"target_x": target[0], "target_y": target[1], "reached": res["reached"],
               "t_s": f"{res['t']:.2f}", "d_final_m": f"{res['d_final']:.3f}",
               "max_off_line_m": f"{off_line(res['path'], target):.3f}",
               "fell_at": "" if res["fell_at"] is None else f"{res['fell_at']:.2f}", "onnx_sha8": sha8(a.onnx)}
        rows.append(row)
        paths.append(res["path"])
        print(row)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for (tx, ty), path, c, row in zip(TARGETS, paths, COLORS, rows):
        ax.plot(path[:, 0], path[:, 1], color=c, lw=2, label=f"({tx:+.1f}, {ty:+.1f})  {row['t_s']} s")
        ax.add_patch(plt.Circle((tx, ty), 0.05, fill=False, color=c, lw=1.5))
    ax.plot(0, 0, "o", color="#0b0b0b", ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(color="#e0e0dc", lw=0.8)
    ax.legend(title="target  (time)", frameon=False, loc="lower left")
    ax.set_title("E006 navigator: go to point (E005 129.8M)")
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, bbox_inches="tight")
    print("wrote", a.png)


if __name__ == "__main__":
    main()
