"""Render the navigator (scripts/navigator.py) walking E005 through a list of waypoints: follow camera + top-down map.

Left: MuJoCo view following the robot. Right: map with the waypoints (numbered, the active one in blue), the
robot's trail, its heading and the line to the active waypoint; below it the numbers the navigator uses each
control step (distance d to the waypoint, heading error alpha). Video time = sim time / --speed.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground:scripts .venv/bin/python scripts/render_nav.py \
      --onnx checkpoints/e005/2026_10_28_173132_129761280.onnx --speed 4 --out videos/final/clip_navigator.mp4

Guarantees: deterministic (no noise); same control law as scripts/eval_goto.py (navigator.command_to, base
position, ARRIVE radius); stops 2 s after the last waypoint, or after 120 s of sim time, and prints which.
"""
import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from duck_rollout import DuckRollout
from navigator import ARRIVE, command_to, yaw
from render_compare import COLORS, FPS, INK, INK2, SURFACE
from video_cards import font, open_writer

W, H = 1280, 720
VW = 800                      # MuJoCo view width; the map column is W - VW
MAP = (VW + 40, 110, W - 40, 110 + (W - VW - 80))   # square map box
EXTENT = (-0.3, 1.2)          # world range shown on both map axes, m
WAYPOINTS = [(0.9, 0.0), (0.9, 0.9), (0.0, 0.9), (0.0, 0.0)]
CAM_DIST, CAM_AZ, CAM_EL, FOLLOW = 1.8, 135.0, -30.0, 0.04
MAX_SIM_S, TAIL_S = 120.0, 2.0


def to_map(x, y):
    x0, y0, x1, y1 = MAP
    s = (x1 - x0) / (EXTENT[1] - EXTENT[0])
    return x0 + (x - EXTENT[0]) * s, y1 - (y - EXTENT[0]) * s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--speed", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    r = DuckRollout(a.onnx, (0.0, 0.0, 0.0))
    r.model.vis.global_.offwidth, r.model.vis.global_.offheight = VW, H  # rendering only
    renderer = mujoco.Renderer(r.model, H, VW)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(r.model, cam)
    cam.distance, cam.azimuth, cam.elevation = CAM_DIST, CAM_AZ, CAM_EL
    cam.lookat[:] = [r.data.qpos[0], r.data.qpos[1], 0.15]
    big, mid, small = font(34), font(26), font(22)
    ff = open_writer(a.out, W, H)

    wp, trail, d_now, a_now, done_at, n = 0, [], 0.0, 0.0, None, 0
    while True:
        t_target = n * a.speed / FPS
        while r.t < t_target - 0.5 * r.model.opt.timestep and done_at is None:
            if r.i % r.inf.decimation == 0:
                vx, vy, wz, d_now, a_now = command_to(r.data, WAYPOINTS[wp])
                if d_now < ARRIVE:
                    wp += 1
                    if wp == len(WAYPOINTS):
                        done_at, wp = r.t, len(WAYPOINTS) - 1
                        r.inf.commands[0:3] = [0.0, 0.0, 0.0]
                        break
                    vx, vy, wz, d_now, a_now = command_to(r.data, WAYPOINTS[wp])
                r.inf.commands[0:3] = [vx, vy, wz]
                trail.append((float(r.data.qpos[0]), float(r.data.qpos[1])))
            r.step()
            if r.t > MAX_SIM_S or r.fell_at is not None:
                done_at = r.t
        if done_at is not None and t_target > done_at + TAIL_S * a.speed:
            break
        cam.lookat[:] = cam.lookat + FOLLOW * (np.array([r.data.qpos[0], r.data.qpos[1], 0.15]) - cam.lookat)
        renderer.update_scene(r.data, cam)
        frame = Image.new("RGB", (W, H), SURFACE)
        frame.paste(Image.fromarray(renderer.render()), (0, 0))
        d = ImageDraw.Draw(frame)
        d.rectangle([0, 0, VW, 52], fill=SURFACE)
        d.text((20, 10), "道案内役: 次の目標点へ向かう指令を、毎回計算して出す", font=small, fill=INK)
        d.text((VW - 20, 10), f"{r.t:5.1f} s (×{a.speed:g} 速)", font=small, fill=INK2, anchor="ra")
        d.text((MAP[0], 60), "上から見た地図", font=mid, fill=INK)
        d.rectangle(MAP, outline=INK2, width=2)
        gx0 = MAP[0]
        for v in np.arange(0.0, 1.21, 0.3):
            (px, _), (_, py) = to_map(v, 0), to_map(0, v)
            d.line([(px, MAP[1]), (px, MAP[3])], fill="#e4e4e0")
            d.line([(MAP[0], py), (MAP[2], py)], fill="#e4e4e0")
        if len(trail) > 1:
            d.line([to_map(*q) for q in trail], fill=COLORS[1], width=4)
        for k, (wx, wy) in enumerate(WAYPOINTS):
            cx, cy = to_map(wx, wy)
            active = k == wp and done_at is None
            d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=COLORS[0] if active else "#c7c7c2")
            d.text((cx + 16, cy - 28), str(k + 1), font=small, fill=INK)
        bx, by = to_map(r.data.qpos[0], r.data.qpos[1])
        if done_at is None:
            tx, ty = to_map(*WAYPOINTS[wp])
            d.line([(bx, by), (tx, ty)], fill=COLORS[0], width=2)
        hx, hy = to_map(r.data.qpos[0] + 0.12 * np.cos(yaw(r.data)), r.data.qpos[1] + 0.12 * np.sin(yaw(r.data)))
        d.line([(bx, by), (hx, hy)], fill=INK, width=4)
        d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=COLORS[1], outline="white", width=2)
        yy = MAP[3] + 22
        if done_at is None:
            d.text((MAP[0], yy), f"目標点  {wp + 1} / {len(WAYPOINTS)}", font=mid, fill=INK)
            d.text((MAP[0], yy + 40), f"距離 d  {d_now * 100:3.0f} cm", font=mid, fill=INK)
            d.text((MAP[0], yy + 80), f"向きのずれ α  {np.degrees(a_now):+4.0f}°", font=mid, fill=INK)
        else:
            d.text((MAP[0], yy), "全部の目標点に到着", font=mid, fill=INK)
        ff.stdin.write(frame.tobytes())
        n += 1
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"wrote {a.out}: {n / FPS:.1f} s video; sim {r.t:.1f} s; waypoints reached={wp + 1 if done_at is not None and r.fell_at is None else wp}; fell_at={r.fell_at}")


if __name__ == "__main__":
    main()
