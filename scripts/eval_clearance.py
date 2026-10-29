"""E009: how high does each checkpoint lift its feet while walking forward? Writes a CSV.

For every ONNX in --run (and any extra --onnx), walks at vx=0.15 for --seconds and, from 2 s on, records
the sole height (lowest collision-mesh vertex of each foot, scripts/jump_test.sole_z) above the floor.
Reports the max per foot, the mean of each step's peak (median of per-step maxima), mean forward speed,
and the first fall time.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground:scripts .venv/bin/python scripts/eval_clearance.py \
      --run checkpoints/e009 --onnx checkpoints/e005/2026_10_28_173132_129761280.onnx \
      --seconds 8 --out docs/data/e009_clearance.csv

Guarantees: deterministic (no noise); the gait period of the rollout's reference is the trained one (27).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from duck_rollout import DuckRollout
from eval_checkpoints import list_checkpoints, sha8
from jump_test import sole_z

WARMUP = 2.0  # s


def measure(onnx, seconds):
    r = DuckRollout(str(onnx), (0.15, 0.0, 0.0))
    feet = [r.model.body("foot_assembly").id, r.model.body("foot_assembly_2").id]
    zs = []
    x_warm = None
    while r.t < seconds:
        r.step()
        if r.t >= WARMUP and r.i % 5 == 0:
            if x_warm is None:
                x_warm = float(r.data.qpos[0])
            zs.append([sole_z(r.model, r.data, f) for f in feet])
    zs = np.array(zs)
    # per-step peaks: split each foot's trace where it touches down (z < 2 mm) and take each swing's max
    peaks = []
    for k in range(2):
        z, cur = zs[:, k], []
        for v in z:
            if v > 0.002:
                cur.append(v)
            elif cur:
                peaks.append(max(cur))
                cur = []
    vx = (float(r.data.qpos[0]) - x_warm) / (seconds - WARMUP)
    return {"max_L_cm": zs[:, 0].max() * 100, "max_R_cm": zs[:, 1].max() * 100,
            "median_step_peak_cm": (np.median(peaks) * 100) if peaks else 0.0, "n_steps": len(peaks),
            "vx_mps": vx, "fell_at": r.fell_at}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--onnx", nargs="*", default=[])
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    items = [(s, o) for s, o in list_checkpoints(a.run)] + [(-1, Path(o)) for o in a.onnx]
    rows = []
    for steps, onnx in items:
        m = measure(onnx, a.seconds)
        row = {"steps": steps, "onnx": onnx.name, "max_L_cm": f"{m['max_L_cm']:.2f}", "max_R_cm": f"{m['max_R_cm']:.2f}",
               "median_step_peak_cm": f"{m['median_step_peak_cm']:.2f}", "n_steps": m["n_steps"],
               "vx_mps": f"{m['vx_mps']:.3f}", "fell_at": "" if m["fell_at"] is None else f"{m['fell_at']:.2f}",
               "onnx_sha8": sha8(onnx)}
        rows.append(row)
        print(row, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
