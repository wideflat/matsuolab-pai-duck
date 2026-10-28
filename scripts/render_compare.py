"""Render a 2x2 side-by-side MP4 of several checkpoints walking, for the report and the video.

Each panel uses the same fixed side camera (it does not follow the robot), with floor marks every
0.1 m, so forward progress is visible. Each panel shows its own distance and fall state. A bottom
strip plots eval reward and forward speed for every checkpoint, with the shown ones highlighted.
The speed data comes from the CSV that eval_checkpoints.py writes.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/render_compare.py \
      --csv docs/data/e003_checkpoints.csv --run checkpoints/baseline_e003 \
      --steps 0 25231360 39649280 50462720 --names 学習開始 報酬は頭打ち 歩き始め 最終 \
      --vx 0.15 --seconds 12 --out videos/e003_compare_v2.mp4 --poster videos/e003_compare_v2.png

Guarantees: deterministic (no noise); exactly four checkpoints; each requested step must exist in
both the CSV and the run directory, else it raises. Writes only --out and --poster.
"""
import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duck_rollout import DuckRollout
from eval_checkpoints import list_checkpoints

W, H = 1280, 720
HEADER, PANEL_W, PANEL_H, STRIP = 40, 640, 260, 160
FPS = 30
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz reference palette, slots 1-4 (light)
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# Fixed side camera: looks along +y, so +x (walking direction) is to the right of the image.
CAM_LOOKAT, CAM_DIST, CAM_AZ, CAM_EL = (0.36, 0.0, 0.17), 0.9, 90.0, -18.0
MARK_X = np.round(np.arange(-0.1, 0.91, 0.1), 2)


def add_floor_marks(scene, x0, marks=MARK_X):
    """Draw thin boxes on the floor every 0.1 m from the start position; start line in red."""
    for x in marks:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("scene.maxgeom too small for floor marks")
        major = x * 10 % 5 == 0
        rgba = [0.95, 0.15, 0.15, 1] if x == 0 else ([1, 0.85, 0.2, 1] if major else [1, 1, 1, 0.9])
        width = 0.006 if major else 0.003
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([width, 0.12, 0.0005]), np.array([x0 + x, 0.0, 0.0006]),
                            np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scene.ngeom += 1


def project(cam, fovy_deg, point, w, h):
    """World point -> pixel (u, v) for a free MjvCamera."""
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = np.array(cam.lookat) - cam.distance * fwd
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = np.asarray(point) - pos
    f = (h / 2) / np.tan(np.radians(fovy_deg) / 2)
    z = rel @ fwd
    return w / 2 + f * (rel @ right) / z, h / 2 - f * (rel @ up) / z


def strip_image(rows, shown):
    """Two small charts (no dual axis): eval reward and 10 s forward speed vs training steps."""
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    steps = np.array([int(r["steps"]) for r in rows]) / 1e6
    series = [("評価報酬(学習中の指標)", [float(r["eval_reward"]) for r in rows], "{:.0f}"),
              ("前進速度 m/s(指令 0.15, 10 秒再生)", [float(r["mean_vx"]) for r in rows], "{:.3f}")]
    fig, axes = plt.subplots(1, 2, figsize=(W / 100, STRIP / 100), dpi=100, facecolor=SURFACE)
    for ax, (title, ys, fmt) in zip(axes, series):
        ax.set_facecolor(SURFACE)
        ax.plot(steps, ys, color=INK2, lw=2)
        for c, s in zip(COLORS, shown):
            k = [int(r["steps"]) for r in rows].index(s)
            ax.plot(steps[k], ys[k], "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(fmt.format(ys[k]), (steps[k], ys[k]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("学習ステップ数(百万)", fontsize=8, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        ax.grid(alpha=0.25)
        ax.set_ylim(min(0, min(ys)), max(ys) * 1.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int, nargs=4, required=True)
    p.add_argument("--names", nargs=4, required=True)
    p.add_argument("--vx", type=float, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--poster", required=True)
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    onnx_by_step = dict(list_checkpoints(a.run))
    rollouts = [DuckRollout(str(onnx_by_step[s]), (a.vx, 0.0, 0.0)) for s in a.steps]
    model = rollouts[0].model
    renderer = mujoco.Renderer(model, PANEL_H, PANEL_W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = CAM_LOOKAT, CAM_DIST, CAM_AZ, CAM_EL
    fovy = model.vis.global_.fovy
    big, mid, small = (ImageFont.truetype(FONT, s) for s in (22, 18, 13))
    strip = strip_image(rows, a.steps)
    ticks = [(x, project(cam, fovy, (rollouts[0].x0 + x, -0.12, 0.0), PANEL_W, PANEL_H)) for x in MARK_X]

    out = Path(a.out)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", str(out)], stdin=subprocess.PIPE)
    n_frames = int(a.seconds * FPS)
    frame = None
    for n in range(n_frames):
        for r in rollouts:
            r.advance_to((n + 1) / FPS)
        frame = Image.new("RGB", (W, H), SURFACE)
        frame.paste(strip, (0, H - STRIP))
        draw = ImageDraw.Draw(frame)
        draw.text((16, 8), f"Open Duck Mini v2  学習の進み方(同じ指令: 前へ {a.vx} m/s)", font=big, fill=INK)
        draw.text((W - 16, 8), f"t = {rollouts[0].t:4.1f} s", font=big, fill=INK, anchor="ra")
        for k, (r, s, name, c) in enumerate(zip(rollouts, a.steps, a.names, COLORS)):
            ox, oy = (k % 2) * PANEL_W, HEADER + (k // 2) * PANEL_H
            renderer.update_scene(r.data, cam)
            add_floor_marks(renderer.scene, r.x0)
            panel = Image.fromarray(renderer.render())
            pd = ImageDraw.Draw(panel)
            for x, (u, v) in ticks:
                if x * 10 % 5 == 0:
                    pd.text((u, v + 4), f"{x:.1f} m", font=small, fill="white", anchor="ma")
            pd.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=c, width=4)
            pd.rectangle([4, 4, 300, 34], fill=c)
            pd.text((12, 8), f"{s / 1e6:.1f}M steps  {name}", font=mid, fill="white")
            state = "転倒" if r.fell_at is not None else f"{r.dx:+.2f} m"
            pd.text((PANEL_W - 12, 10), state, font=big, fill="white", anchor="ra",
                    stroke_width=2, stroke_fill="black")
            frame.paste(panel, (ox, oy))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    frame.save(a.poster)
    print("wrote", out, n_frames, "frames;", a.poster)
    for r, s in zip(rollouts, a.steps):
        print(f"steps={s} dx={r.dx:.3f} m fell_at={r.fell_at}")


if __name__ == "__main__":
    main()
