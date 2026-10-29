"""Render one E007 duck-soccer trial to MP4: follow camera, the striker's current plan, and a top-down minimap.

Runs the same control loop as scripts/eval_soccer.py (run_trial), so the video shows exactly the trial in
docs/data/e007_soccer.csv. Video time = sim time / --speed. After a goal (or fall / time-out) the last
frame is held for HOLD seconds.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground:scripts .venv/bin/python scripts/render_soccer.py \
      --onnx checkpoints/e005/2026_10_28_173132_129761280.onnx --ball 0.3 0.6 --seconds 90 --speed 2 \
      --out videos/soccer_held_out_03_06.mp4

Guarantees: deterministic; writes only --out; prints goal_at and fell_at.
"""
import argparse
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import soccer_scene as S
from eval_soccer import run_trial
from navigator import yaw
from render_compare import COLORS, FONT, FPS, INK, SURFACE

W, H = 1280, 720
CAM_DIST, CAM_AZ, CAM_EL, FOLLOW = 1.7, 115.0, -28.0, 0.05
HOLD = 2.0  # s
MODE_LABEL = {"around": "ボールの横へ回り込む", "line_up": "ボールの真後ろへ", "push": "ゴールへ押す"}
MAP = (W - 330, 70, W - 20, 300)          # minimap box in pixels
MAP_X, MAP_Y = (-0.9, 2.0), (-0.75, 0.75)  # world extent shown in the minimap, m


def to_map(x, y):
    x0, y0, x1, y1 = MAP
    return (x0 + (x - MAP_X[0]) / (MAP_X[1] - MAP_X[0]) * (x1 - x0),
            y1 - (y - MAP_Y[0]) / (MAP_Y[1] - MAP_Y[0]) * (y1 - y0))


def draw_minimap(draw, duck_trail, ball_trail, duck_xy, heading, ball, target):
    draw.rounded_rectangle(MAP, radius=8, fill=SURFACE)
    g, hw, dp = S.GOAL_X, S.GOAL_HALF_W, S.GOAL_DEPTH
    draw.line([to_map(g, hw), to_map(g + dp, hw), to_map(g + dp, -hw), to_map(g, -hw)], fill="#52514e", width=2)
    draw.line([to_map(g, -hw), to_map(g, hw)], fill="#d62020", width=3)
    if len(duck_trail) > 1:
        draw.line([to_map(*p) for p in duck_trail], fill=COLORS[0], width=2)
    if len(ball_trail) > 1:
        draw.line([to_map(*p) for p in ball_trail], fill=COLORS[1], width=3)
    tx, ty = to_map(*target)
    draw.line([to_map(*duck_xy), (tx, ty)], fill=COLORS[0], width=1)
    draw.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], outline=COLORS[0], width=2)
    bx, by = to_map(*ball)
    rb = S.BALL_R / (MAP_X[1] - MAP_X[0]) * (MAP[2] - MAP[0])
    draw.ellipse([bx - rb, by - rb, bx + rb, by + rb], fill=COLORS[1])
    dx, dy = to_map(*duck_xy)
    hx, hy = to_map(duck_xy[0] + 0.15 * np.cos(heading), duck_xy[1] + 0.15 * np.sin(heading))
    draw.line([(dx, dy), (hx, hy)], fill=INK, width=3)
    draw.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=COLORS[0], outline="white", width=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--ball", type=float, nargs=2, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--speed", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    model = S.build_model()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H  # in-memory only; rendering, not physics
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.azimuth, cam.elevation = CAM_DIST, CAM_AZ, CAM_EL
    cam.lookat[:] = [(a.ball[0]) / 2, a.ball[1] / 2, 0.12]
    big, mid, small = (ImageFont.truetype(FONT, s) for s in (34, 22, 16))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", a.out], stdin=subprocess.PIPE)
    st = {"n": 0, "duck": [], "ball": [], "frame": None}

    def render(r, info, final=False):
        d = r.data
        duck, ball = S.feet_xy(r.model, d), S.ball_xy(d)
        st["duck"].append(tuple(duck))
        st["ball"].append(tuple(ball))
        focus = np.array([(duck[0] + ball[0]) / 2, (duck[1] + ball[1]) / 2, 0.12])
        cam.lookat[:] = cam.lookat + FOLLOW * (focus - cam.lookat)
        renderer.update_scene(d, cam)
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, W, 56], fill=SURFACE)
        draw.text((20, 12), "ダックサッカー(追加の学習なし: 歩行方策 + 道案内役)", font=mid, fill=INK)
        draw.text((W - 20, 12), f"{r.t:5.1f} s  (×{a.speed:g} 速)", font=mid, fill=INK, anchor="ra")
        if info["goal_at"] is not None:
            draw.text((W / 2, H / 2 - 40), "GOAL!", font=ImageFont.truetype(FONT, 120), fill=COLORS[3],
                      anchor="mm", stroke_width=6, stroke_fill="black")
        elif r.fell_at is not None:
            draw.text((W / 2, H / 2 - 40), "転倒", font=ImageFont.truetype(FONT, 100), fill="white",
                      anchor="mm", stroke_width=6, stroke_fill="black")
        else:
            label = MODE_LABEL[info["mode"]]
            box_w = int(draw.textlength(label, font=big)) + 36
            draw.rounded_rectangle([20, 72, 20 + box_w, 126], radius=10, fill=COLORS[0])
            draw.text((38, 80), label, font=big, fill="white")
        draw_minimap(draw, st["duck"], st["ball"], duck, yaw(d), ball, info["target"])
        draw.text((20, H - 30), "ボール直径 24 cm・50 g / ゴール幅 60 cm / 方策は歩くことしか学習していない",
                  font=small, fill="white", stroke_width=2, stroke_fill="black")
        st["frame"] = frame
        ff.stdin.write(frame.tobytes())
        st["n"] += 1

    def on_step(r, info):
        if r.t >= st["n"] * a.speed / FPS:
            render(r, info)

    res = run_trial(a.onnx, model, tuple(a.ball), a.seconds, on_step)
    for _ in range(int(HOLD * FPS)):
        ff.stdin.write(st["frame"].tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"wrote {a.out}: {st['n'] / FPS + HOLD:.1f} s video, goal_at={res['goal_at']}, fell_at={res['fell_at']}")


if __name__ == "__main__":
    main()
