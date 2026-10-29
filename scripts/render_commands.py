"""Render two policies side by side under the same sequence of joystick commands (forward, back, turn, side step).

Left: E003 baseline (50M steps). Right: E005 (129.8M steps). Each panel follows its own robot; below it the
total distance walked along the commanded axis and the total angle turned are shown, so the difference is a
number as well as a picture. 1280x720, 30 fps, no audio (captions are burned into the picture).

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground:scripts .venv/bin/python scripts/render_commands.py \
      --left checkpoints/baseline_e003/2026_10_28_034228_50462720.onnx --left_label "E003 ベースライン(5,000 万ステップ)" \
      --right checkpoints/e005/2026_10_28_173132_129761280.onnx --right_label "E005(1.3 億ステップ)" \
      --seconds_each 4 --out videos/e003_vs_e005_commands.mp4

Guarantees: deterministic (no noise); commands are switched at fixed times and the same for both panels; the
policy clock is not changed (phase_rate = 1); writes only --out.
"""
import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from duck_rollout import DuckRollout
from navigator import yaw
from render_compare import COLORS, FPS, INK, INK2, SURFACE
from video_cards import font, open_writer

W, H = 1280, 720
HEAD, PW, PH = 96, 640, 520
CAM_DIST, CAM_AZ, CAM_EL, FOLLOW = 1.2, 135.0, -20.0, 0.06
SEQUENCE = [("前へ歩く", (0.15, 0.0, 0.0)), ("後ろへ下がる", (-0.15, 0.0, 0.0)),
            ("その場で左に回る", (0.0, 0.0, 1.0)), ("左へカニ歩き", (0.0, 0.2, 0.0))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--left_label", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--right_label", required=True)
    p.add_argument("--seconds_each", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rolls = [DuckRollout(a.left, SEQUENCE[0][1]), DuckRollout(a.right, SEQUENCE[0][1])]
    renderers, cams = [], []
    for r in rolls:
        r.model.vis.global_.offwidth, r.model.vis.global_.offheight = PW, PH  # rendering only
        renderers.append(mujoco.Renderer(r.model, PH, PW))
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.model, cam)
        cam.distance, cam.azimuth, cam.elevation = CAM_DIST, CAM_AZ, CAM_EL
        cam.lookat[:] = [r.data.qpos[0], r.data.qpos[1], 0.15]
        cams.append(cam)
    labels = [(a.left_label, COLORS[0]), (a.right_label, COLORS[1])]
    big, mid, small = font(34), font(26), font(22)
    ff = open_writer(a.out, W, H)

    cmd_start = [(r.data.qpos[0], r.data.qpos[1]) for r in rolls]
    yaw_prev = [yaw(r.data) for r in rolls]
    yaw_sum = [0.0, 0.0]
    yaw_at_start = [0.0, 0.0]
    k_prev = 0
    total = a.seconds_each * len(SEQUENCE)
    for n in range(int(total * FPS)):
        t = n / FPS
        k = min(int(t // a.seconds_each), len(SEQUENCE) - 1)
        if k != k_prev:
            cmd_start = [(r.data.qpos[0], r.data.qpos[1]) for r in rolls]
            yaw_at_start = list(yaw_sum)
            k_prev = k
        for i, r in enumerate(rolls):
            r.inf.commands[0:3] = list(SEQUENCE[k][1])
            while r.t < t - 0.5 * r.model.opt.timestep:
                r.step()
                if r.i % r.inf.decimation == 0:
                    yw = yaw(r.data)
                    yaw_sum[i] += (yw - yaw_prev[i] + np.pi) % (2 * np.pi) - np.pi
                    yaw_prev[i] = yw
        frame = Image.new("RGB", (W, H), SURFACE)
        d = ImageDraw.Draw(frame)
        name, cmd = SEQUENCE[k]
        d.text((30, 18), "同じ指令を出したときの違い", font=mid, fill=INK2)
        d.text((30, 50), f"指令: {name}", font=big, fill=INK)
        d.text((W - 30, 22), f"{t:4.1f} s", font=mid, fill=INK2, anchor="ra")
        for i, r in enumerate(rolls):
            cams[i].lookat[:] = cams[i].lookat + FOLLOW * (np.array([r.data.qpos[0], r.data.qpos[1], 0.15]) - cams[i].lookat)
            renderers[i].update_scene(r.data, cams[i])
            panel = Image.fromarray(renderers[i].render())
            frame.paste(panel, (i * PW, HEAD))
            label, color = labels[i]
            d.rectangle([i * PW, HEAD, i * PW + PW - 1, HEAD + PH - 1], outline=color, width=4)
            d.rectangle([i * PW + 4, HEAD + 4, i * PW + 4 + int(d.textlength(label, font=small)) + 24, HEAD + 40], fill=color)
            d.text((i * PW + 16, HEAD + 8), label, font=small, fill="white")
            moved = np.hypot(r.data.qpos[0] - cmd_start[i][0], r.data.qpos[1] - cmd_start[i][1])
            turned = np.degrees(yaw_sum[i] - yaw_at_start[i])
            if cmd[2] != 0.0:
                stat = f"回った角度  {turned:+.0f}°"
            else:
                stat = f"進んだ距離  {moved:.2f} m"
            d.text((i * PW + PW // 2, HEAD + PH + 46), stat, font=big, fill=INK, anchor="mm")
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"wrote {a.out}: {total:.0f} s; falls: {[r.fell_at for r in rolls]}")


if __name__ == "__main__":
    main()
