"""E007: duck soccer trials (navigator + striker, no training); writes a CSV and a small-multiples plot.

Each trial is a fresh rollout: duck at the origin facing +x, ball at one start position. A trial ends on a
goal (scripts/soccer_scene.is_goal), a fall, or after --seconds. Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground:scripts .venv/bin/python scripts/eval_soccer.py \
      --onnx checkpoints/e005/2026_10_28_173132_129761280.onnx --seconds 90 \
      --out docs/data/e007_soccer.csv --png videos/e007_soccer.png

Guarantees: deterministic (no noise). run_trial() is also used by the video renderer, so the video and the
CSV come from the same control loop.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import soccer_scene as S
from duck_rollout import DuckRollout
from eval_checkpoints import sha8
from navigator import command_to
from striker import target

BALL_STARTS = [(0.5, 0.0), (0.5, 0.35), (0.6, -0.35), (1.0, 0.5), (-0.4, 0.2)]


def run_trial(onnx, model, ball0, seconds, on_step=None):
    """Play one trial. on_step(rollout, info) is called after every physics step (for rendering)."""
    r = S.attach(DuckRollout(onnx, (0.0, 0.0, 0.0)), model, ball0)
    mode, goal_at, switches = "line_up", None, 0
    duck_path, ball_path, tgt = [], [], np.zeros(2)
    while r.t < seconds and r.fell_at is None and goal_at is None:
        if r.i % r.inf.decimation == 0:
            duck = r.data.qpos[0:2].copy()
            ball = S.ball_xy(r.data)
            tgt, new_mode = target(duck, ball, mode)
            switches += new_mode != mode
            mode = new_mode
            vx, vy, wz, _, _ = command_to(r.data, tgt)
            r.inf.commands[0:3] = [vx, vy, wz]
            duck_path.append(duck)
            ball_path.append(ball)
        r.step()
        if S.is_goal(r.data):
            goal_at = r.t
        if on_step is not None:
            on_step(r, {"mode": mode, "target": tgt, "goal_at": goal_at})
    return {"goal_at": goal_at, "fell_at": r.fell_at, "t": r.t, "mode_switches": switches,
            "ball_end": S.ball_xy(r.data), "duck_path": np.array(duck_path), "ball_path": np.array(ball_path)}


def draw_field(ax):
    gx, hw, dp = S.GOAL_X, S.GOAL_HALF_W, S.GOAL_DEPTH
    ax.plot([gx, gx + dp, gx + dp, gx], [hw, hw, -hw, -hw], color="#52514e", lw=1.5)
    ax.plot([gx, gx], [-hw, hw], color="#d62020", lw=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png", required=True)
    a = p.parse_args()

    model = S.build_model()
    rows, results = [], []
    for ball0 in BALL_STARTS:
        res = run_trial(a.onnx, model, ball0, a.seconds)
        row = {"ball_x0": ball0[0], "ball_y0": ball0[1], "goal": res["goal_at"] is not None,
               "goal_at_s": "" if res["goal_at"] is None else f"{res['goal_at']:.2f}",
               "fell_at": "" if res["fell_at"] is None else f"{res['fell_at']:.2f}",
               "t_end_s": f"{res['t']:.2f}", "ball_end_x": f"{res['ball_end'][0]:.3f}",
               "ball_end_y": f"{res['ball_end'][1]:.3f}", "mode_switches": res["mode_switches"],
               "onnx_sha8": sha8(a.onnx)}
        rows.append(row)
        results.append(res)
        print(row, flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)

    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.4), dpi=150, sharey=True)
    for ax, res, row in zip(axes, results, rows):
        draw_field(ax)
        ax.plot(res["duck_path"][:, 0], res["duck_path"][:, 1], color="#2a78d6", lw=1.2, label="duck")
        ax.plot(res["ball_path"][:, 0], res["ball_path"][:, 1], color="#eb6834", lw=2, label="ball")
        ax.add_patch(plt.Circle((row["ball_x0"], row["ball_y0"]), S.BALL_R, fill=False, color="#eb6834", ls="--"))
        ax.plot(0, 0, "o", color="#0b0b0b", ms=4)
        verdict = f"GOAL {row['goal_at_s']} s" if row["goal"] else ("FELL" if row["fell_at"] else "no goal")
        ax.set_title(f"ball ({row['ball_x0']:+.1f}, {row['ball_y0']:+.2f})\n{verdict}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.9, 2.0)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(color="#e0e0dc", lw=0.6)
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("E007 duck soccer (E005 129.8M + navigator + striker)", fontsize=10)
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, bbox_inches="tight")
    print("wrote", a.png)


if __name__ == "__main__":
    main()
