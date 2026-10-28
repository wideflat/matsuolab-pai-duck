"""Render a cross-run comparison MP4: rows = two training runs, columns = two checkpoint steps.

Same fixed side camera and floor marks as render_compare.py. The bottom strip has two charts:
forward speed vs training steps for both runs (from eval_checkpoints.py CSVs), and a top-down
trajectory of each run's last-column checkpoint, drawn up to the current time. Optionally also writes
a static top-down trajectory PNG over a longer horizon (--traj_seconds) for the report.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/render_runs.py \
      --run "E003 ベースライン" checkpoints/baseline_e003 docs/data/e003_checkpoints.csv \
      --run "E004 前進報酬×2" checkpoints/e004 docs/data/e004_checkpoints.csv \
      --steps 28835840 50462720 --vx 0.15 --seconds 12 \
      --out videos/e003_vs_e004.mp4 --poster videos/e003_vs_e004.png \
      --traj_seconds 30 --traj_png videos/e003_vs_e004_traj30s.png

Guarantees: deterministic (no noise); exactly two runs and two steps; every step must exist in each
run directory and CSV, else it raises. Writes only --out, --poster and --traj_png.
"""
import argparse
import csv
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duck_rollout import DuckRollout
from eval_checkpoints import list_checkpoints
from render_compare import (CAM_AZ, CAM_EL, COLORS, FONT, FPS, HEADER, INK, INK2, PANEL_H, PANEL_W, STRIP, SURFACE,
                            H, W, add_floor_marks, project)

RUN_COLORS = COLORS[:2]
# Wider than render_compare.py: E004 covers ~1.45 m in 12 s.
CAM_LOOKAT, CAM_DIST = (0.75, 0.0, 0.17), 1.3
MARK_X = np.round(np.arange(-0.1, 1.81, 0.1), 2)


def simulate_xy(onnx, vx, seconds):
    """Base (x, y) relative to start, sampled once per video frame."""
    r = DuckRollout(onnx, (vx, 0.0, 0.0))
    xy = []
    for n in range(int(seconds * FPS)):
        r.advance_to((n + 1) / FPS)
        xy.append((r.dx, float(r.data.qpos[1])))
    return np.array(xy), r.fell_at


def strip_axes(runs, steps, trajs):
    """Static strip: speed chart (left) and empty trajectory axes (right); returns image and a data->pixel map."""
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W / 100, STRIP / 100), dpi=100, facecolor=SURFACE)
    for (name, _, rows), c in zip(runs, RUN_COLORS):
        s = np.array([int(r["steps"]) for r in rows]) / 1e6
        v = np.array([float(r["mean_vx"]) for r in rows])
        ax1.plot(s, v, color=c, lw=2, label=name)
        for st in steps:
            k = [int(r["steps"]) for r in rows].index(st)
            ax1.plot(s[k], v[k], "o", ms=8, color=c, mec=SURFACE, mew=2, zorder=3)
            ax1.annotate(f"{v[k]:.3f}", (s[k], v[k]), xytext=(-4, 6), textcoords="offset points",
                         ha="right", fontsize=8, color=INK)
    ax1.set_title("前進速度 m/s(指令 0.15、10 秒再生)", fontsize=10, color=INK, loc="left")
    ax1.set_xlabel("学習ステップ数(百万)", fontsize=8, color=INK2)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    allxy = np.concatenate(trajs)
    x_hi = max(allxy[:, 0].max(), 0.2) * 1.08
    y_lo, y_hi = min(allxy[:, 1].min(), 0) - 0.05, max(allxy[:, 1].max(), 0) + 0.05
    ax2.set_xlim(-0.05, x_hi)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_title(f"上から見た軌跡({steps[-1] / 1e6:.1f}M、+y が左、縦は拡大)", fontsize=10, color=INK, loc="left")
    ax2.set_xlabel("前方 x [m]", fontsize=8, color=INK2)
    ax2.set_ylabel("横 y [m]", fontsize=8, color=INK2)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=8, colors=INK2)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    height = img.height
    to_px = lambda pts: [(u, height - v) for u, v in ax2.transData.transform(pts)]
    plt.close(fig)
    return img, to_px


def trajectory_png(runs, onnx_last, vx, seconds, path):
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (name, _, _), onnx, c in zip(runs, onnx_last, RUN_COLORS):
        xy, fell = simulate_xy(onnx, vx, seconds)
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=2, label=f"{name}({'転倒' if fell else '転倒なし'})")
        ax.plot(*xy[-1], "o", ms=8, color=c, mec=SURFACE, mew=2)
        ax.annotate(f"{seconds:.0f} 秒後 ({xy[-1, 0]:.2f}, {xy[-1, 1]:.2f})", xy[-1], xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.plot(0, 0, "s", color=INK2, ms=6)
    ax.annotate("スタート", (0, 0), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("前方 x [m]")
    ax.set_ylabel("横 y [m](+ が左)")
    ax.set_title(f"同じ指令(前へ {vx} m/s)で {seconds:.0f} 秒歩いた軌跡(上から見た図)", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs=3, action="append", required=True, metavar=("NAME", "DIR", "CSV"))
    p.add_argument("--steps", type=int, nargs=2, required=True)
    p.add_argument("--vx", type=float, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--poster", required=True)
    p.add_argument("--traj_seconds", type=float)
    p.add_argument("--traj_png")
    a = p.parse_args()
    if len(a.run) != 2:
        raise ValueError("exactly two --run entries are required")

    runs = [(name, dict(list_checkpoints(d)), list(csv.DictReader(open(c)))) for name, d, c in a.run]
    onnx = [[str(ck[s]) for s in a.steps] for _, ck, _ in runs]
    trajs = [simulate_xy(o[-1], a.vx, a.seconds)[0] for o in onnx]
    strip, to_px = strip_axes(runs, a.steps, trajs)

    rollouts = [[DuckRollout(path, (a.vx, 0.0, 0.0)) for path in row] for row in onnx]
    model = rollouts[0][0].model
    renderer = mujoco.Renderer(model, PANEL_H, PANEL_W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = CAM_LOOKAT, CAM_DIST, CAM_AZ, CAM_EL
    x0 = rollouts[0][0].x0
    ticks = [(x, project(cam, model.vis.global_.fovy, (x0 + x, -0.12, 0.0), PANEL_W, PANEL_H)) for x in MARK_X]
    big, mid, small = (ImageFont.truetype(FONT, s) for s in (22, 18, 13))

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", a.out], stdin=subprocess.PIPE)
    frame = None
    for n in range(int(a.seconds * FPS)):
        for row in rollouts:
            for r in row:
                r.advance_to((n + 1) / FPS)
        frame = Image.new("RGB", (W, H), SURFACE)
        frame.paste(strip, (0, H - STRIP))
        draw = ImageDraw.Draw(frame)
        draw.text((16, 8), f"E003 と E004 の比較(同じ指令: 前へ {a.vx} m/s)", font=big, fill=INK)
        draw.text((W - 16, 8), f"t = {rollouts[0][0].t:4.1f} s", font=big, fill=INK, anchor="ra")
        for k, traj in enumerate(trajs):
            pts = to_px(traj[: n + 1])
            ox, oy = 0, H - STRIP
            pts = [(u + ox, v + oy) for u, v in pts]
            if len(pts) > 1:
                draw.line(pts, fill=RUN_COLORS[k], width=3)
            u, v = pts[-1]
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=RUN_COLORS[k], outline=SURFACE, width=2)
        for i, ((name, _, _), row, c) in enumerate(zip(runs, rollouts, RUN_COLORS)):
            for j, (r, s) in enumerate(zip(row, a.steps)):
                renderer.update_scene(r.data, cam)
                add_floor_marks(renderer.scene, r.x0, MARK_X)
                panel = Image.fromarray(renderer.render())
                pd = ImageDraw.Draw(panel)
                for x, (u, v) in ticks:
                    if x * 10 % 5 == 0:
                        pd.text((u, v + 4), f"{x:.1f} m", font=small, fill="white", anchor="ma")
                pd.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=c, width=4)
                pd.rectangle([4, 4, 380, 34], fill=c)
                pd.text((12, 8), f"{name}  {s / 1e6:.1f}M steps", font=mid, fill="white")
                state = "転倒" if r.fell_at is not None else f"{r.dx:+.2f} m"
                pd.text((PANEL_W - 12, 42), state, font=big, fill="white", anchor="ra",
                        stroke_width=2, stroke_fill="black")
                frame.paste(panel, (j * PANEL_W, HEADER + i * PANEL_H))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    frame.save(a.poster)
    for (name, _, _), row in zip(runs, rollouts):
        for r, s in zip(row, a.steps):
            print(f"{name} steps={s} dx={r.dx:.3f} y={r.data.qpos[1]:.3f} fell_at={r.fell_at}")
    if a.traj_png:
        trajectory_png(runs, [o[-1] for o in onnx], a.vx, a.traj_seconds, a.traj_png)
        print("wrote", a.traj_png)
    print("wrote", a.out, a.poster)


if __name__ == "__main__":
    main()
