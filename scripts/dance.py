"""Make the duck dance to music with a trained walking policy and no extra training, and render an MP4 with audio.

A choreography JSON (configs/dance_*.json) lists moves and a sequence of (move, beats). Each move is a
joystick command (vx, vy, wz), which the policy was trained to follow, plus optional sinusoidal head
motions written straight to the head servos (head commands carry no reward in training, so the policy
does not track them). The gait clock is sped up so one gait cycle lasts `beats_per_gait_cycle` beats,
and its phase 0 is put on the first beat. Video time equals music time; the video ends one beat after
the last move (or at the end of the music) and the audio fades out over the last FADE seconds.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/dance.py \
      --config configs/dance_fluffing_a_duck.json --onnx checkpoints/e004/2026_10_28_153255_50462720.onnx \
      --label "E004" --out videos/dance_fluffing_e004.mp4

Guarantees: deterministic; every config key is required; head amplitudes must stay inside the joint
range and under the servo speed limit, else it raises; prints fell_at (None = no fall). Writes only --out.
"""
import argparse
import json
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duck_rollout import DuckRollout
from render_compare import COLORS, FONT, FPS, INK, SURFACE

W, H = 1280, 720
GAIT_PERIOD = 0.54  # s, period of the trained gait (27 control steps at 50 Hz)
MAX_SERVO_VEL = 5.24  # rad/s, from Open_Duck_Playground joystick config
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
CAM_DIST, CAM_AZ, CAM_EL, FOLLOW = 1.1, 135.0, -18.0, 0.04
FADE = 1.5  # s, audio fade-out at the end


def load_config(path):
    with open(path) as f:
        c = json.load(f)
    beat = 60.0 / c["bpm"]
    timeline, t = [], c["first_beat"]
    for name, beats in c["sequence"]:
        move = c["moves"][name]
        if len(move["cmd"]) != 3:
            raise ValueError(f"move {name}: cmd must be [vx, vy, wz]")
        for joint, (amp, period_beats) in move["head"].items():
            if joint not in HEAD_JOINTS:
                raise ValueError(f"move {name}: unknown head joint {joint}")
            peak_vel = 2 * np.pi * amp / (period_beats * beat)
            if peak_vel > MAX_SERVO_VEL:
                raise ValueError(f"move {name}: {joint} peak {peak_vel:.2f} rad/s > {MAX_SERVO_VEL}")
        timeline.append((t, t + beats * beat, name, move))
        t += beats * beat
    return c, beat, timeline


def yaw_deg(d):
    q = d.qpos[3:7]
    return np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    c, beat, timeline = load_config(a.config)
    end = timeline[-1][1] + beat
    phase_rate = GAIT_PERIOD / (c["beats_per_gait_cycle"] * beat)
    r = DuckRollout(a.onnx, (0.0, 0.0, 0.0), phase_rate=phase_rate)
    m, d = r.model, r.data
    steps_per_period = r.inf.PRM.nb_steps_in_period
    # imitation_i advances phase_rate per control step (50 Hz); make it 0 at the first beat.
    r.inf.imitation_i = (-c["first_beat"] * 50 * phase_rate) % steps_per_period
    names = list(r.inf.actuator_names)
    head = {j: names.index(j) for j in HEAD_JOINTS}
    default = r.inf.default_actuator
    for _, _, name, move in timeline:
        for joint, (amp, _) in move["head"].items():
            lo, hi = m.jnt_range[m.joint(joint).id]
            if not (lo <= default[head[joint]] - amp and default[head[joint]] + amp <= hi):
                raise ValueError(f"move {name}: {joint} ±{amp} leaves joint range [{lo:.2f}, {hi:.2f}]")

    m.vis.global_.offwidth, m.vis.global_.offheight = W, H  # in-memory only; rendering, not physics
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.azimuth, cam.elevation = CAM_DIST, CAM_AZ, CAM_EL
    cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.17]
    big, mid, small = (ImageFont.truetype(FONT, s) for s in (30, 22, 16))

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", c["music"], "-map", "0:v", "-map", "1:a",
                           "-af", f"afade=t=out:st={end - FADE:.3f}:d={FADE}",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "192k",
                           "-shortest", a.out], stdin=subprocess.PIPE)
    current = ("pose", timeline[0][3])
    marks = []  # (segment index, x, y, yaw) at each segment start, for the per-move summary
    yaw_prev, yaw_total = yaw_deg(d), 0.0  # unwrapped heading, so turns past 180 deg are counted in full
    for n in range(int(end * FPS)):
        t_frame = (n + 1) / FPS
        while r.t < t_frame - 0.5 * m.opt.timestep:
            seg = next((s for s in timeline if s[0] <= r.t < s[1]), None)
            idx = timeline.index(seg) if seg else -1
            if not marks or marks[-1][0] != idx:
                marks.append((idx, float(d.qpos[0]), float(d.qpos[1]), yaw_total))
            current = (seg[2], seg[3]) if seg else ("pose", c["moves"]["pose"])
            move = current[1]
            r.inf.commands[0:3] = list(move["cmd"])
            r.step()
            yaw_now = yaw_deg(d)
            yaw_total += (yaw_now - yaw_prev + 180) % 360 - 180
            yaw_prev = yaw_now
            seg_start = seg[0] if seg else 0.0
            for joint, (amp, period_beats) in move["head"].items():
                d.ctrl[head[joint]] = default[head[joint]] + amp * np.sin(
                    2 * np.pi * (r.t - seg_start) / (period_beats * beat))
        target = np.array([d.qpos[0], d.qpos[1], 0.17])
        cam.lookat[:] = cam.lookat + FOLLOW * (target - cam.lookat)
        renderer.update_scene(d, cam)
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        k = int((r.t - c["first_beat"]) // beat) if r.t >= c["first_beat"] else -1
        on_beat = k >= 0 and (r.t - c["first_beat"]) % beat < 0.12
        draw.rectangle([0, 0, W, 56], fill=SURFACE)
        draw.text((20, 10), f"Open Duck Mini v2 ダンス(学習なし・{a.label} の歩行方策)", font=mid, fill=INK)
        draw.text((W - 20, 10), f"{c['bpm']:.0f} BPM  {r.t:5.1f} s", font=mid, fill=INK, anchor="ra")
        draw.rounded_rectangle([20, 72, 20 + 34 * 8 + 300, 124], radius=10, fill=COLORS[0])
        draw.text((36, 80), current[1]["label"], font=big, fill="white")
        for b in range(4):
            x0 = 330 + b * 34
            lit = k >= 0 and k % 4 == b
            fill = COLORS[3] if lit and on_beat else ("white" if lit else COLORS[0])
            draw.ellipse([x0, 86, x0 + 24, 110], fill=fill, outline="white", width=2)
        state = "転倒" if r.fell_at is not None else ""
        if state:
            draw.text((W - 20, 80), state, font=big, fill="white", anchor="ra", stroke_width=2, stroke_fill="black")
        draw.text((20, H - 30), c["music_credit"], font=small, fill="white", stroke_width=2, stroke_fill="black")
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    marks.append((None, float(d.qpos[0]), float(d.qpos[1]), yaw_total))
    for (idx, x, y, yw), (_, x2, y2, yw2) in zip(marks, marks[1:]):
        if idx is not None and idx >= 0:
            print(f"  {timeline[idx][2]:<12} moved ({x2 - x:+.2f}, {y2 - y:+.2f}) m, "
                  f"turned {yw2 - yw:+.0f} deg")
    print(f"wrote {a.out}: {end:.1f} s, phase_rate={phase_rate:.3f}, fell_at={r.fell_at}, "
          f"final xy=({r.dx:.2f}, {d.qpos[1]:.2f})")


if __name__ == "__main__":
    main()
